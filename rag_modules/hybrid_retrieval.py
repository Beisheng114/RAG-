"""
混合检索模块
基于双层检索范式：实体级 + 主题级检索
结合图结构检索和向量检索，使用Round-robin轮询策略
"""

import hashlib
import json
import logging
import os
import pickle
import re
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from neo4j import GraphDatabase
from .graph_indexing import GraphIndexingModule

logger = logging.getLogger(__name__)


def _strip_chinese_query_fillers(text: str) -> str:
    """去掉问句尾巴，便于词表命中（不影响向量检索，仅供规则关键词）。"""
    q = text.strip()
    for pat in (
        r"怎么办\??$",
        r"如何处理\??$",
        r"如何解决\??$",
        r"什么原因\??$",
        r"什么原因造成的\??$",
        r"怎么处理\??$",
        r"如何排除\??$",
        r"如何\??$",
        r"吗\??$",
        r"[？?！!。]+$",
    ):
        q = re.sub(pat, "", q).strip()
    return q


def _dedupe_terms_prefer_longer(terms: List[str]) -> List[str]:
    """长词优先，去掉被长词包含的短词（如保留「直流电机」去掉「电机」）。"""
    out: List[str] = []
    for t in sorted((x for x in terms if x), key=len, reverse=True):
        if any(t != o and t in o for o in out):
            continue
        out.append(t)
    return out


@dataclass
class RetrievalResult:
    """检索结果数据结构"""
    content: str
    neo4j_node_id: str
    entity_type: str
    relevance_score: float
    retrieval_level: str  # 'low' or 'high'
    metadata: Dict[str, Any]

class HybridRetrievalModule:
    """
    混合检索模块
    核心特点：
    1. 双层检索范式（实体级 + 主题级）
    2. 关键词提取和匹配
    3. 图结构+向量检索结合
    4. 一跳邻居扩展
    5. Round-robin轮询合并策略
    """
    def __init__(self, config, vector_module, data_module, llm_client):
        self.config = config
        self.vector_module = vector_module
        self.data_module = data_module
        self.llm_client = llm_client
        self.driver = None
        self.bm25_retriever = None
        
        # 添加Ollama支持
        self.llm_provider = config.llm_provider
        self.ollama_base_url = config.ollama_base_url.rstrip('/')
        self.ollama_model = config.ollama_model
        
        self.graph_indexing = GraphIndexingModule(config, llm_client)
        self.graph_indexed = False
        
        # 添加缓存
        self.node_info_cache = {}
        self.query_cache = {}
        self.max_query_cache_size = 256

        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=4)
    def initialize(self, chunks: List[Document]):
        """初始化检索系统"""
        logger.info("初始化混合检索模块...")
        
        # 连接Neo4j
        self.driver = GraphDatabase.driver(
            self.config.neo4j_uri, 
            auth=(self.config.neo4j_user, self.config.neo4j_password)
        )
        
        # 初始化BM25检索器（优先加载持久化索引；未命中或数据变化时重建并落盘，
        # 避免每次启动全量重建的内存/CPU开销，也支持知识库更新后自动失效）
        if chunks:
            self.bm25_retriever = self._load_or_build_bm25(chunks)
            logger.info(f"BM25检索器初始化完成，文档数量: {len(chunks)}")
        
        # 初始化图索引
        self._build_graph_index()

    # ---------- BM25 索引持久化 ----------

    BM25_CACHE_DIR = "bm25_cache"
    BM25_CACHE_FILE = os.path.join(BM25_CACHE_DIR, "bm25_retriever.pkl")

    @staticmethod
    def _bm25_signature(chunks: List) -> str:
        """基于全部块内容生成知识库指纹，用于校验缓存有效性

        覆盖增删改各种变化（包括块数不变但内容替换的情况）
        """
        hasher = hashlib.md5()
        for c in chunks:
            content = str(getattr(c, "page_content", c))
            hasher.update(content.encode("utf-8"))
            hasher.update(b"\x00")
        return hasher.hexdigest()

    def _load_or_build_bm25(self, chunks: List):
        """加载持久化BM25索引；缓存不存在/失效/损坏时重建并保存"""
        signature = self._bm25_signature(chunks)
        try:
            if os.path.exists(self.BM25_CACHE_FILE):
                with open(self.BM25_CACHE_FILE, "rb") as f:
                    payload = pickle.load(f)
                if payload.get("signature") == signature:
                    retriever = payload.get("retriever")
                    logger.info(f"BM25索引缓存命中，跳过重建（块数: {len(chunks)}）")
                    return retriever
                logger.info("BM25索引缓存已失效（知识库数据变化），将重建")
        except Exception as e:
            logger.warning(f"BM25索引缓存加载失败，将重建: {e}")

        retriever = BM25Retriever.from_documents(chunks)
        self._save_bm25_cache(retriever, signature)
        return retriever

    def _save_bm25_cache(self, retriever, signature: str) -> None:
        """持久化BM25索引（失败仅告警，不影响主链路）"""
        try:
            os.makedirs(self.BM25_CACHE_DIR, exist_ok=True)
            with open(self.BM25_CACHE_FILE, "wb") as f:
                pickle.dump({"signature": signature, "retriever": retriever}, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"BM25索引已持久化: {self.BM25_CACHE_FILE}")
        except Exception as e:
            logger.warning(f"BM25索引持久化失败（不影响使用）: {e}")

    def invalidate_bm25_cache(self) -> None:
        """删除BM25索引缓存，下次初始化时强制重建（知识库增量更新/重建后调用）"""
        try:
            if os.path.exists(self.BM25_CACHE_FILE):
                os.remove(self.BM25_CACHE_FILE)
                logger.info("BM25索引缓存已清除，将在下次初始化时重建")
        except Exception as e:
            logger.warning(f"清除BM25索引缓存失败: {e}")
        
    def _build_graph_index(self):
        """构建图索引"""
        if self.graph_indexed:
            return

        logger.info("开始构建图索引...")

        try:
            equipment_categories = getattr(self.data_module, 'equipment_categories', [])
            equipments = self.data_module.equipments
            components = self.data_module.components
            faults = getattr(self.data_module, 'faults', [])
            fault_phenomenons = self.data_module.fault_phenomenons
            fault_reasons = self.data_module.fault_reasons
            maintenance_actions = getattr(self.data_module, 'maintenance_actions', [])
            safety_notices = getattr(self.data_module, 'safety_notices', [])
            knowledge_sources = getattr(self.data_module, 'knowledge_sources', [])

            self.graph_indexing.create_entity_key_values(
                equipment_categories=equipment_categories,
                equipments=equipments,
                components=components,
                faults=faults,
                fault_phenomenons=fault_phenomenons,
                fault_reasons=fault_reasons,
                maintenance_actions=maintenance_actions,
                safety_notices=safety_notices,
                knowledge_sources=knowledge_sources
            )

            relationships = self._extract_relationships_from_graph()
            self.graph_indexing.create_relation_key_values(relationships)

            self.graph_indexing.deduplicate_entities_and_relations()

            self.graph_indexed = True
            stats = self.graph_indexing.get_statistics()
            logger.info(f"图索引构建完成: {stats}")

        except Exception as e:
            logger.error(f"构建图索引失败: {e}")
            
    def _extract_relationships_from_graph(self) -> List[Tuple[str, str, str]]:
        """从Neo4j图中提取关系"""
        relationships = []
        
        try:
            with self.driver.session() as session:
                query = """
                MATCH (source)-[r]->(target)
                RETURN 
                    COALESCE(source.equipment_id, source.component_id, source.fault_id, 
                             source.phenomenon_id, source.cause_id, source.action_id,
                             source.notice_id, source.category_id, source.source_id, source.name) as source_id,
                    type(r) as relation_type,
                    COALESCE(target.equipment_id, target.component_id, target.fault_id,
                             target.phenomenon_id, target.cause_id, target.action_id,
                             target.notice_id, target.category_id, target.source_id, target.name) as target_id
                LIMIT 1000
                """
                result = session.run(query)
                
                for record in result:
                    if record["source_id"] and record["target_id"]:
                        relationships.append((
                            record["source_id"],
                            record["relation_type"],
                            record["target_id"]
                        ))
                    
        except Exception as e:
            logger.error(f"提取图关系失败: {e}")
            
        return relationships



    def extract_query_keywords(self, query: str, retry_count: int = 1) -> Tuple[List[str], List[str]]:
#         """
#         提取查询关键词：实体级 + 主题级
        
#         Args:
#             query: 用户查询文本
#             retry_count: 重试次数
            
#         Returns:
#             (实体级关键词列表, 主题级关键词列表)
#         """
#         prompt = f"""
# 作为船舶故障维修知识助手，请分析以下查询并提取关键词，分为两个层次：
# 查询：{query}
# 提取规则：
# 1. 实体级关键词：具体的设备名称、部件、故障现象、故障原因等有形实体
#    - 例如：主机、发电机、冷却泵、轴承、过热、异响、泄漏、振动
#    - 对于抽象查询，推测相关的具体设备/部件

# 2. 主题级关键词：抽象概念、维修主题、故障类型、维修特点等
#    - 例如：故障诊断、预防性维护、紧急维修、电气故障、机械故障、液压系统
#    - 排除动作词：推荐、介绍、维修、怎么做等

# 示例：
# 查询："主机过热故障如何处理" 
# {{
#     "entity_keywords": ["主机", "冷却泵", "温度传感器", "润滑油", "散热器"],
#     "topic_keywords": ["过热", "故障诊断", "冷却系统", "温度控制", "预防性维护"]
# }}

# 查询："发电机异响怎么排查"
# {{
#     "entity_keywords": ["发电机", "轴承", "齿轮箱", "联轴器", "润滑油"],
#     "topic_keywords": ["异响", "故障排查", "机械故障", "振动分析", "润滑系统"]
# }}

# 请严格按照JSON格式返回，不要包含多余的文字：
# {{
#     "entity_keywords": ["实体1", "实体2", ...],
#     "topic_keywords": ["主题1", "主题2", ...]
# }}
# """
        
#         for attempt in range(retry_count):
#             try:
#                 if self.llm_provider == "ollama":
#                     # 使用Ollama原生API
#                     url = f"{self.ollama_base_url}/api/chat"
#                     payload = {
#                         "model": self.ollama_model,
#                         "messages": [{"role": "user", "content": prompt}],
#                         "stream": False,
#                         "options": {
#                             "temperature": 0.1,
#                             "num_predict": 500
#                         }
#                     }
#                     response = requests.post(url, json=payload, timeout=60)
#                     response.raise_for_status()
#                     result = response.json()
#                     content = result["message"]["content"].strip()
#                 else:
#                     # 使用vLLM OpenAI兼容API
#                     response = self.llm_client.chat.completions.create(
#                         model=self.config.llm_model,
#                         messages=[{"role": "user", "content": prompt}],
#                         temperature=0.1,
#                         max_tokens=500
#                     )
#                     content = response.choices[0].message.content.strip()
                
#                 # 检查内容是否为空
#                 if not content:
#                     logger.warning(f"LLM返回空内容 (尝试 {attempt + 1}/{retry_count})")
#                     if attempt < retry_count - 1:
#                         continue
#                     return self._fallback_keyword_extraction(query)
#                 if "<|think|>" in content:
#                     # 移除<|think|>...</|think|>标签及其内容
#                     content = content.split("<|think|>")[-1].split("</|think>")[-1].strip()
                
#                 # 尝试从内容中提取JSON部分
#                 import re
#                 json_match = re.search(r'\{[\s\S]*\}', content)
#                 if json_match:
#                     content = json_match.group(0)
#                 else:
#                     logger.warning(f"未找到JSON内容 (尝试 {attempt + 1}/{retry_count}): {content[:100]}")
#                     if attempt < retry_count - 1:
#                         continue
#                     return self._fallback_keyword_extraction(query)
                
#                 result = json.loads(content)
#                 entity_keywords = result.get("entity_keywords", [])
#                 topic_keywords = result.get("topic_keywords", [])
                
#                 # 验证结果
#                 if not entity_keywords and not topic_keywords:
#                     if attempt < retry_count - 1:
#                         logger.warning(f"关键词提取结果为空，重试 {attempt + 1}/{retry_count}")
#                         continue
                
#                 logger.info(f"关键词提取完成 - 实体级: {entity_keywords}, 主题级: {topic_keywords}")
#                 return entity_keywords, topic_keywords
                
#             except json.JSONDecodeError as e:
#                 logger.error(f"JSON解析失败 (尝试 {attempt + 1}/{retry_count}): {e}")
#                 logger.debug(f"原始内容: {content[:200] if 'content' in locals() else 'N/A'}")
#                 if attempt == retry_count - 1:
#                     return self._fallback_keyword_extraction(query)
#             except Exception as e:
#                 logger.error(f"关键词提取失败 (尝试 {attempt + 1}/{retry_count}): {e}")
#                 if attempt == retry_count - 1:
#                     return self._fallback_keyword_extraction(query)
        
        return self._fallback_keyword_extraction(query)
    
    def _fallback_keyword_extraction(self, query: str) -> Tuple[List[str], List[str]]:
        """
        降级方案：基于规则的关键词提取
        
        Args:
            query: 用户查询文本
            
        Returns:
            (实体级关键词列表, 主题级关键词列表)
        """
        logger.warning("使用降级方案提取关键词")
        
        # 船舶维修领域实体词库
        entity_keywords_set = {
            "主机", "发电机", "冷却泵", "轴承", "滤芯", "传感器", "探头", "阀门",
            "散热器", "润滑油", "液压系统", "电气系统", "控制系统", "齿轮箱", "联轴器",
            "活塞", "曲轴", "缸套", "喷油器", "增压器", "冷却器", "换热器", "分离器",
            "副机", "扫气泵", "消防泵", "舱底泵", "压载泵", "空压机", "干燥机", "储气罐",
            "舵机", "锚机", "绞缆机", "起货机", "推进器", "螺旋桨", "艉轴", "中间轴",
            "密封件", "油封", "密封圈", "垫片", "法兰", "管路", "软管", "硬管",
            "蓄电池", "充电机", "变频器", "变压器", "配电柜", "接触器", "继电器",
            "指示灯", "电缆", "接线盒", "开关", "保险丝", "电机", "直流电机", "交流电机",
            "换向器", "电刷", "碳刷", "启动马达",
            "滤器", "油滤器", "水滤器", "空气滤器", "分油机", "油水分离器", "油渣柜",
            "膨胀水箱", "热交换器", "冷凝器", "蒸发器", "节流阀", "止回阀", "球阀",
            "蝶阀", "闸阀", "减压阀", "安全阀", "压力表", "温度表", "流量计",
            "舱口盖", "舷窗", "舵叶", "螺旋桨轴", "推力轴承",
            "导航设备", "测深仪", "报警装置"
        }

        # 船舶维修领域主题词库
        topic_keywords_set = {
            "过热", "异响", "泄漏", "振动", "故障", "诊断", "维修", "维护", "检查",
            "多原因", "可能原因", "多种原因", "并发故障",
            "预防性", "紧急", "机械", "电气", "液压", "冷却", "润滑", "控制", "温度",
            "压力", "流量", "磨损", "腐蚀", "老化", "断裂", "堵塞", "短路", "断路",
            "失压", "过载", "欠压", "缺相", "跳闸", "打火", "冒烟", "烧蚀", "受潮",
            "进水", "结冰", "结垢", "生锈", "变形", "裂纹", "脱落", "松动", "偏移",
            "卡滞", "抱死", "打滑", "空转", "效率低", "功率不足", "启动困难", "停机",
            "保养", "检修", "拆装", "更换", "调试", "紧固", "清洁", "除锈",
            "防腐", "补焊", "抛光", "检测", "测试", "校验", "排故", "抢修", "巡检",
            "定期", "隐患", "失效", "异常", "渗漏", "滴漏",
            "油压", "水温", "转速", "电流", "电压", "电阻", "绝缘", "接地",
            "通讯", "干扰", "失灵", "无响应", "性能下降", "密封失效",
            "励磁", "建压", "无电压", "反电压",
            "火花", "换向", "空载", "负载"
        }
        
        norm = _strip_chinese_query_fillers(query)
        scan_text = norm if norm else query

        # 中文：按词表长度降序匹配，优先长词（如「直流电机」早于「电机」）
        entity_from_vocab: List[str] = []
        for t in sorted(entity_keywords_set, key=len, reverse=True):
            if t and (t in scan_text or t in query):
                entity_from_vocab.append(t)
        entity_from_vocab = _dedupe_terms_prefer_longer(entity_from_vocab)

        topic_from_vocab: List[str] = []
        for t in sorted(topic_keywords_set, key=len, reverse=True):
            if t and (t in scan_text or t in query):
                topic_from_vocab.append(t)
        topic_from_vocab = _dedupe_terms_prefer_longer(topic_from_vocab)

        # 空格分词：仅当命中词表或是简短拉丁标识时才纳入，避免整句中文被当成「实体」
        raw_tokens = scan_text.split() if scan_text else []
        entity_keywords: List[str] = []
        topic_keywords: List[str] = []
        _ascii_token = re.compile(r"^[A-Za-z][A-Za-z0-9._-]{0,63}$")

        for kw in raw_tokens:
            clean_kw = kw.strip("，。？！、；：\"\"''（）()")
            if not clean_kw:
                continue
            if clean_kw in entity_keywords_set:
                entity_keywords.append(clean_kw)
            elif clean_kw in topic_keywords_set:
                topic_keywords.append(clean_kw)
            elif _ascii_token.match(clean_kw):
                entity_keywords.append(clean_kw)

        def _dedupe_preserve(seq: List[str]) -> List[str]:
            seen = set()
            out = []
            for x in seq:
                if x and x not in seen:
                    seen.add(x)
                    out.append(x)
            return out

        entity_keywords = _dedupe_terms_prefer_longer(
            _dedupe_preserve(entity_from_vocab + entity_keywords)
        )
        topic_keywords = _dedupe_terms_prefer_longer(
            _dedupe_preserve(topic_from_vocab + topic_keywords)
        )

        # 仍无任何命中：用清洗后片段作弱主题锚点，不设整条问句为实体
        if not entity_keywords and not topic_keywords and scan_text:
            snippet = scan_text[:24] if len(scan_text) > 24 else scan_text
            topic_keywords = [snippet]
        
        logger.info(f"降级方案提取 - 实体级: {entity_keywords}, 主题级: {topic_keywords}")
        return entity_keywords, topic_keywords

    def _bm25_get_relevant(self, query: str, k: int) -> List[Document]:
        """LangChain 0.2+ 使用 invoke；旧版使用 get_relevant_documents。"""
        if not self.bm25_retriever or not query:
            return []
        retriever = self.bm25_retriever
        prev_k = getattr(retriever, "k", None)
        try:
            if prev_k is not None:
                retriever.k = k
            getter = getattr(retriever, "get_relevant_documents", None)
            if callable(getter):
                docs = getter(query)
            else:
                out = retriever.invoke(query)
                docs = list(out) if isinstance(out, list) else ([out] if out is not None else [])
            return docs[:k] if docs else []
        except TypeError:
            try:
                out = retriever.invoke({"query": query})
                docs = list(out) if isinstance(out, list) else ([out] if out is not None else [])
                return docs[:k] if docs else []
            except Exception as e:
                logger.warning("BM25 检索失败 (invoke dict): %s", e)
                return []
        except Exception as e:
            logger.warning("BM25 检索失败: %s", e)
            return []
        finally:
            if prev_k is not None:
                retriever.k = prev_k
    
    def entity_level_retrieval(self, entity_keywords: List[str], top_k: int = 5) -> List[RetrievalResult]:
        """
        实体级检索：专注于具体实体和关系
        使用图索引的键值对结构进行检索
        """
        results = []
        
        # 1. 使用图索引进行实体检索
        for keyword in entity_keywords:
            # 检索匹配的实体
            entities = self.graph_indexing.get_entities_by_key(keyword)
            
            for entity in entities:
                # 获取邻居信息
                neighbors = self._get_node_neighbors(entity.metadata["neo4j_node_id"], max_neighbors=2)
                
                # 构建增强内容
                enhanced_content = entity.value_content
                if neighbors:
                    enhanced_content += f"\n相关信息: {', '.join(neighbors)}"
                
                results.append(RetrievalResult(
                    content=enhanced_content,
                    neo4j_node_id=entity.metadata["neo4j_node_id"],
                    entity_type=entity.entity_type,
                    relevance_score=0.9,  # 精确匹配得分较高
                    retrieval_level="entity",
                    metadata={
                        "entity_name": entity.entity_name,
                        "entity_type": entity.entity_type,
                        "index_keys": entity.index_keys,
                        "matched_keyword": keyword
                    }
                ))
        
        # 2. 如果图索引结果不足，使用Neo4j进行补充检索
        if len(results) < top_k:
            neo4j_results = self._neo4j_entity_level_search(entity_keywords, top_k - len(results))
            results.extend(neo4j_results)
            
        # 3. 按相关性排序并返回
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        logger.info(f"实体级检索完成，返回 {len(results)} 个结果")
        return results[:top_k]
    
    def _neo4j_entity_level_search(self, keywords: List[str], limit: int) -> List[RetrievalResult]:
        """Neo4j补充检索"""
        results = []
        
        try:
            with self.driver.session() as session:
                cypher_query = """
                UNWIND $keywords as keyword
                MATCH (node)
                WHERE node:Equipment OR node:Component OR node:FaultPhenomenon OR node:FaultReason
                AND (node.name CONTAINS keyword OR node.description CONTAINS keyword)
                RETURN 
                    COALESCE(node.equipment_id, node.component_id, node.phenomenon_id, node.cause_id) as node_id,
                    node.name as name,
                    node.description as description,
                    labels(node) as labels,
                    CASE 
                        WHEN node.name CONTAINS keyword THEN 1.0
                        WHEN node.description CONTAINS keyword THEN 0.7
                        ELSE 0.5
                    END as score
                ORDER BY score DESC
                LIMIT $limit
                """
                
                try:
                    result = session.run(cypher_query, {
                        "keywords": keywords,
                        "limit": limit
                    })
                except Exception:
                    fallback_query = """
                    UNWIND $keywords as keyword
                    MATCH (e:Equipment)
                    WHERE e.type CONTAINS keyword
                       OR e.name CONTAINS keyword
                    WITH e, keyword
                    OPTIONAL MATCH (e)-[:consists_of]->(c:Component)
                    WITH e, keyword, collect(c.name)[0..3] as components
                    RETURN
                        e.equipment_id as node_id,
                        e.name as name,
                        e.type as equipment_type,
                        components,
                        keyword as matched_keyword
                    ORDER BY e.name
                    LIMIT $limit
                    """
                    result = session.run(fallback_query, {
                        "keywords": keywords,
                        "limit": limit
                    })
                
                for record in result:
                    content_parts = []
                    if record["name"]:
                        content_parts.append(f"设备/部件: {record['name']}")
                    if record["description"]:
                        content_parts.append(f"描述: {record['description']}")
                    
                    results.append(RetrievalResult(
                        content='\n'.join(content_parts),
                        neo4j_node_id=record["node_id"],
                        entity_type=record["labels"][0] if record["labels"] else "Unknown",
                        relevance_score=float(record["score"]) * 0.7,  # 补充检索得分较低
                        retrieval_level="entity",
                        metadata={
                            "name": record["name"],
                            "labels": record["labels"],
                            "source": "neo4j_fallback"
                        }
                    ))
                    
        except Exception as e:
            logger.error(f"Neo4j补充检索失败: {e}")
            
        return results
    
    def topic_level_retrieval(self, topic_keywords: List[str], top_k: int = 5) -> List[RetrievalResult]:
        """
        主题级检索：专注于广泛主题和概念
        使用图索引的关系键值对结构进行主题检索
        """
        results = []
        
        # 1. 使用图索引进行关系/主题检索
        for keyword in topic_keywords:
            # 检索匹配的关系
            relations = self.graph_indexing.get_relations_by_key(keyword)
            
            for relation in relations:
                # 获取相关实体信息
                source_entity = self.graph_indexing.entity_kv_store.get(relation.source_entity)
                target_entity = self.graph_indexing.entity_kv_store.get(relation.target_entity)
                
                if source_entity and target_entity:
                    # 构建丰富的主题内容
                    content_parts = [
                        f"主题: {keyword}",
                        relation.value_content,
                        f"相关设备: {source_entity.entity_name}",
                        f"相关信息: {target_entity.entity_name}"
                    ]
                    
                    # 添加源实体的详细信息
                    if source_entity.entity_type in ["Equipment", "Component", "FaultPhenomenon", "FaultReason"]:
                        newline = '\n'
                        content_parts.append(f"设备详情: {source_entity.value_content.split(newline)[0]}")
                    
                    # Use target_entity as doc id so distinct causes/components are not
                    # collapsed under the same source (e.g. multiple CAUSED_BY from one phenomenon).
                    results.append(RetrievalResult(
                        content='\n'.join(content_parts),
                        neo4j_node_id=relation.target_entity,
                        entity_type=target_entity.entity_type,
                        relevance_score=0.95,  # 主题匹配得分
                        retrieval_level="topic",
                        metadata={
                            "relation_id": relation.relation_id,
                            "relation_type": relation.relation_type,
                            "source_entity_id": relation.source_entity,
                            "target_entity_id": relation.target_entity,
                            "source_name": source_entity.entity_name,
                            "target_name": target_entity.entity_name,
                            "matched_keyword": keyword,
                            "index_keys": relation.index_keys
                        }
                    ))
        
        # 2. 使用实体的分类信息进行主题检索
        for keyword in topic_keywords:
            entities = self.graph_indexing.get_entities_by_key(keyword)
            for entity in entities:
                if entity.entity_type in ["Equipment", "Component", "FaultPhenomenon", "FaultReason"]:
                    # 构建分类主题内容
                    content_parts = [
                        f"主题分类: {keyword}",
                        entity.value_content
                    ]
                    
                    results.append(RetrievalResult(
                        content='\n'.join(content_parts),
                        neo4j_node_id=entity.metadata["neo4j_node_id"],
                        entity_type=entity.entity_type,
                        relevance_score=0.85,  # 分类匹配得分
                        retrieval_level="topic",
                        metadata={
                            "entity_name": entity.entity_name,
                            "entity_type": entity.entity_type,
                            "matched_keyword": keyword,
                            "source": "equipment_type_match"
                        }
                    ))
        
        # 3. 如果结果不足，使用Neo4j进行补充检索
        if len(results) < top_k:
            neo4j_results = self._neo4j_topic_level_search(topic_keywords, top_k - len(results))
            results.extend(neo4j_results)
            
        # 4. 按相关性排序并返回
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        logger.info(f"主题级检索完成，返回 {len(results)} 个结果")
        return results[:top_k]
    
    def _neo4j_topic_level_search(self, keywords: List[str], limit: int) -> List[RetrievalResult]:
        """Neo4j主题级检索补充"""
        results = []
        
        try:
            with self.driver.session() as session:
                cypher_query = """
                UNWIND $keywords as keyword
                CALL db.index.fulltext.queryNodes('idx_fulltext_equipment_topic', keyword) YIELD node AS e, score
                WHERE e:Equipment
                WITH e, keyword, score
                ORDER BY score DESC
                LIMIT $limit
                OPTIONAL MATCH (e)-[:consists_of]->(c:Component)
                WITH e, keyword, collect(c.name)[0..3] as components
                RETURN
                    e.equipment_id as node_id,
                    e.name as name,
                    e.type as equipment_type,
                    components,
                    keyword as matched_keyword
                ORDER BY e.name
                LIMIT $limit
                """
                
                result = session.run(cypher_query, {
                    "keywords": keywords,
                    "limit": limit
                })
                
                for record in result:
                    content_parts = []
                    content_parts.append(f"设备: {record['name']}")

                    if record["equipment_type"]:
                        content_parts.append(f"设备类型: {record['equipment_type']}")

                    if record["components"]:
                        components_str = ', '.join(record["components"][:3])
                        content_parts.append(f"主要部件: {components_str}")

                    results.append(RetrievalResult(
                        content='\n'.join(content_parts),
                        neo4j_node_id=record["node_id"],
                        entity_type="Equipment",
                        relevance_score=0.75,  # 补充检索得分
                        retrieval_level="topic",
                        metadata={
                            "name": record["name"],
                            "equipment_type": record["equipment_type"],
                            "matched_keyword": record["matched_keyword"],
                            "source": "neo4j_fallback"
                        }
                    ))
                    
        except Exception as e:
            logger.error(f"Neo4j主题级检索失败: {e}")
            
        return results
        
    def dual_level_retrieval(self, query: str, top_k: int = 5) -> List[Document]:
        """
        双层检索：结合实体级和主题级检索
        """
        logger.info(f"开始双层检索: {query}")
        
        # 1. 提取关键词
        entity_keywords, topic_keywords = self.extract_query_keywords(query)
        
        # 2. 执行双层检索
        entity_results = self.entity_level_retrieval(entity_keywords, top_k)
        topic_results = self.topic_level_retrieval(topic_keywords, top_k)
        
        # 3. 结果合并和排序
        all_results = entity_results + topic_results
        
        # 4. 去重和重排序（关系级用 relation_id，避免多原因共用同一源节点被合并）
        seen_keys = set()
        unique_results = []

        def _dedup_key(r: RetrievalResult) -> str:
            rid = (r.metadata or {}).get("relation_id")
            if rid:
                return f"rel:{rid}"
            nid = r.neo4j_node_id
            if nid is not None and str(nid).strip():
                return f"node:{nid}"
            return f"txt:{hash(r.content[:400])}"

        for result in sorted(all_results, key=lambda x: x.relevance_score, reverse=True):
            key = _dedup_key(result)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            unique_results.append(result)
        
        # 5. 转换为Document格式
        documents = []
        for result in unique_results[:top_k]:
            # 确保entity_name字段正确设置
            entity_name = result.metadata.get("name") or result.metadata.get("entity_name")
            if not entity_name or entity_name == "未知设备":
                if result.neo4j_node_id:
                    entity_name = self._get_node_name_from_neo4j(result.neo4j_node_id)
                else:
                    entity_name = "未知设备"
            
            source_info = []
            if result.neo4j_node_id:
                source_info = self._get_node_source_info(result.neo4j_node_id)
            
            doc = Document(
                page_content=result.content,
                metadata={
                    "neo4j_node_id": result.neo4j_node_id,
                    "entity_type": result.entity_type,
                    "retrieval_level": result.retrieval_level,
                    "relevance_score": result.relevance_score,
                    "entity_name": entity_name,
                    "search_type": "dual_level",
                    "sources": source_info,
                    **result.metadata
                }
            )
            documents.append(doc)
            
        logger.info(f"双层检索完成，返回 {len(documents)} 个文档")
        return documents
    
    
    def _get_node_neighbors(self, node_id: str, max_neighbors: int = 3) -> List[str]:
        """获取节点的邻居信息（带缓存）"""
        # 检查缓存
        cache_key = f"node_neighbors_{node_id}_{max_neighbors}"
        if cache_key in self.node_info_cache:
            return self.node_info_cache[cache_key]
        
        try:
            with self.driver.session() as session:
                query = """
                MATCH (n)
                WHERE n.category_id = $node_id OR n.equipment_id = $node_id OR n.component_id = $node_id 
                   OR n.fault_id = $node_id OR n.phenomenon_id = $node_id OR n.cause_id = $node_id 
                   OR n.action_id = $node_id OR n.notice_id = $node_id OR n.source_id = $node_id
                MATCH (n)-[r]-(neighbor)
                RETURN neighbor.name as name
                LIMIT $limit
                """
                result = session.run(query, {"node_id": node_id, "limit": max_neighbors})
                neighbors = [record["name"] for record in result if record["name"]]
                
                # 缓存结果
                self.node_info_cache[cache_key] = neighbors
                return neighbors
        except Exception as e:
            logger.error(f"获取邻居节点失败: {e}")
            return []
    
    def _get_node_name_from_neo4j(self, node_id: str) -> str:
        """从Neo4j获取节点名称（带缓存）"""
        # 检查缓存
        cache_key = f"node_name_{node_id}"
        if cache_key in self.node_info_cache:
            return self.node_info_cache[cache_key]
        
        try:
            with self.driver.session() as session:
                query = """
                MATCH (n)
                WHERE n.category_id = $node_id OR n.equipment_id = $node_id OR n.component_id = $node_id 
                   OR n.fault_id = $node_id OR n.phenomenon_id = $node_id OR n.cause_id = $node_id 
                   OR n.action_id = $node_id OR n.notice_id = $node_id OR n.source_id = $node_id
                RETURN n.name as name
                LIMIT 1
                """
                result = session.run(query, {"node_id": node_id})
                record = next(result, None)
                name = record["name"] if record else "未知设备"
                
                # 缓存结果
                self.node_info_cache[cache_key] = name
                return name
        except Exception as e:
            logger.error(f"获取节点名称失败: {e}")
            return "未知设备"
    
    def _get_node_source_info(self, node_id: str) -> List[Dict]:
        """获取节点的来源信息（KnowledgeSource）（带缓存）"""
        # 检查缓存
        cache_key = f"node_source_{node_id}"
        if cache_key in self.node_info_cache:
            return self.node_info_cache[cache_key]
        
        try:
            with self.driver.session() as session:
                # 尝试多种关系类型
                query = """
                MATCH (n)-[r]->(ks:KnowledgeSource)
                WHERE (n.category_id = $node_id OR n.equipment_id = $node_id OR n.component_id = $node_id 
                   OR n.fault_id = $node_id OR n.phenomenon_id = $node_id OR n.cause_id = $node_id 
                   OR n.action_id = $node_id OR n.notice_id = $node_id)
                RETURN ks.source_id as source_id, COALESCE(ks.name, ks.title, '未知来源') as name, ks.type as type, 
                       ks.chapter as chapter, ks.section as section, ks.reliability as reliability
                """
                result = session.run(query, {"node_id": node_id})
                sources = []
                for record in result:
                    sources.append({
                        "source_id": record["source_id"],
                        "name": record["name"],
                        "type": record["type"],
                        "chapter": record["chapter"],
                        "section": record["section"],
                        "reliability": record["reliability"]
                    })
                
                # 如果没有找到关系，尝试通过 source_id 属性直接查找
                if not sources:
                    query = """
                    MATCH (ks:KnowledgeSource {source_id: $source_id})
                    RETURN ks.source_id as source_id, COALESCE(ks.name, ks.title, '未知来源') as name, ks.type as type, 
                           ks.chapter as chapter, ks.section as section, ks.reliability as reliability
                    """
                    # 先获取节点的 source_id 属性
                    get_source_id_query = """
                    MATCH (n)
                    WHERE n.category_id = $node_id OR n.equipment_id = $node_id OR n.component_id = $node_id 
                       OR n.fault_id = $node_id OR n.phenomenon_id = $node_id OR n.cause_id = $node_id 
                       OR n.action_id = $node_id OR n.notice_id = $node_id
                    RETURN n.source_id as source_id
                    """
                    source_id_result = session.run(get_source_id_query, {"node_id": node_id})
                    source_id_record = source_id_result.single()
                    if source_id_record and source_id_record["source_id"]:
                        source_id_value = source_id_record["source_id"]
                        ks_result = session.run(query, {"source_id": source_id_value})
                        for record in ks_result:
                            sources.append({
                                "source_id": record["source_id"],
                                "name": record["name"],
                                "type": record["type"],
                                "chapter": record["chapter"],
                                "section": record["section"],
                                "reliability": record["reliability"]
                            })
                
                # 缓存结果
                self.node_info_cache[cache_key] = sources
                return sources
        except Exception as e:
            logger.error(f"获取来源信息失败: {e}")
            return []
    
    def hybrid_search(self, query: str, top_k: int = 5, mode: str = "rrf") -> List[Document]:
        """
        高级混合检索：支持多路召回 + RRF融合
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            mode: 检索模式
                - "rrf": RRF融合检索（向量+BM25+图，推荐）
                - "semantic_only": 仅语义检索
                - "simple": 简单向量检索
                - "weighted": 向量混合检索
                
        Returns:
            Document对象列表
        """
        logger.info(f"开始高级混合检索: {query} (模式: {mode})")
        
        cache_key = f"hybrid_search_{query}_{top_k}_{mode}"
        if cache_key in self.query_cache:
            logger.info("使用缓存的检索结果")
            return self.query_cache[cache_key]
        
        try:
            k_fetch = top_k * 2

            def _run_vector():
                if mode == "semantic_only":
                    return self.vector_module.similarity_search(query, k=k_fetch)
                if mode == "weighted":
                    return self.vector_module.semantic_keyword_search(
                        query, k=k_fetch, semantic_weight=0.7
                    )
                if mode == "simple":
                    return self.vector_module.vector_keyword_search(query, k=k_fetch)
                return self.vector_module.hybrid_search(query, k=k_fetch, mode="rrf")

            fv = self.executor.submit(_run_vector)
            fb = (
                self.executor.submit(self._bm25_get_relevant, query, k_fetch)
                if self.bm25_retriever
                else None
            )
            fg = self.executor.submit(self.dual_level_retrieval, query, k_fetch)

            vector_docs = fv.result()
            bm25_docs = fb.result() if fb else []
            graph_docs = fg.result()

            # 轻量截断，减少后续融合排序开销
            if vector_docs:
                vector_docs = vector_docs[: max(k_fetch, top_k)]
            if bm25_docs:
                bm25_docs = bm25_docs[: max(k_fetch, top_k)]
            if graph_docs:
                graph_docs = graph_docs[: max(k_fetch, top_k)]

            logger.info(
                "多路召回完成 - 向量: %s, BM25: %s, 图检索: %s",
                len(vector_docs),
                len(bm25_docs),
                len(graph_docs)
            )

            def _to_document(result: Dict[str, Any]) -> Document:
                content = result.get("text", "")
                metadata = result.get("metadata", {})
                node_id = metadata.get("neo4j_node_id")

                entity_name = metadata.get("entity_name")
                if not entity_name or entity_name == "未知设备":
                    if node_id:
                        entity_name = self._get_node_name_from_neo4j(node_id)
                    else:
                        entity_name = "未知设备"

                source_info = []
                if node_id:
                    source_info = self._get_node_source_info(node_id)

                return Document(
                    page_content=content,
                    metadata={
                        "neo4j_node_id": node_id,
                        "entity_name": entity_name,
                        "search_method": "vector",
                        "score": result.get("score", 0.0),
                        "sources": source_info,
                        **metadata
                    }
                )

            vector_documents = [_to_document(doc) for doc in vector_docs]

            def _doc_id(doc: Document) -> str:
                node_id = doc.metadata.get("neo4j_node_id")
                if node_id:
                    return str(node_id)
                return str(hash(doc.page_content[:200]))

            def _rrf_score(rank: int, k: int = 60) -> float:
                return 1.0 / (k + rank)

            fused: Dict[str, Dict[str, Any]] = {}

            def _add_docs(docs: List[Document], channel: str):
                for idx, doc in enumerate(docs, start=1):
                    doc_id = _doc_id(doc)
                    score = _rrf_score(idx)
                    if doc_id not in fused:
                        fused[doc_id] = {
                            "doc": doc,
                            "score": 0.0,
                            "channels": set()
                        }
                    fused[doc_id]["score"] += score
                    fused[doc_id]["channels"].add(channel)

            _add_docs(vector_documents, "vector")
            _add_docs(bm25_docs, "bm25")
            _add_docs(graph_docs, "graph")

            logger.info("RRF融合完成，候选数: %s", len(fused))

            ranked = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
            documents = []
            for item in ranked[:top_k]:
                doc = item["doc"]
                channels = sorted(item["channels"])
                doc.metadata.update({
                    "search_method": "rrf_fusion",
                    "rrf_score": item["score"],
                    "recall_channels": channels
                })
                documents.append(doc)

            self.query_cache[cache_key] = documents
            if len(self.query_cache) > self.max_query_cache_size:
                # 删除最旧的一个缓存键（Python3.7+ dict 保序）
                oldest_key = next(iter(self.query_cache))
                self.query_cache.pop(oldest_key, None)

            logger.info(f"高级混合检索完成，返回 {len(documents)} 个文档")
            return documents
            
        except Exception as e:
            logger.error(f"高级混合检索失败: {e}")
            return []
        
    def close(self):
        """关闭资源连接"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j连接已关闭") 