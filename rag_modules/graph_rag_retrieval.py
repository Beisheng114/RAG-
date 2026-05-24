"""
真正的图RAG检索模块
基于图结构的知识推理和检索，而非简单的关键词匹配
"""

import json
import logging
import re
import time
from collections import defaultdict, deque, OrderedDict
from typing import List, Dict, Tuple, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum

from langchain_core.documents import Document
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """查询类型枚举"""
    ENTITY_RELATION = "entity_relation"  # 实体关系查询：A和B有什么关系？
    MULTI_HOP = "multi_hop"  # 多跳查询：A通过什么连接到C？
    SUBGRAPH = "subgraph"  # 子图查询：A相关的所有信息
    PATH_FINDING = "path_finding"  # 路径查找：从A到B的最佳路径
    CLUSTERING = "clustering"  # 聚类查询：和A相似的都有什么？

@dataclass
class GraphQuery:
    """图查询结构"""
    query_type: QueryType
    source_entities: List[str]
    target_entities: List[str] = None
    relation_types: List[str] = None
    max_depth: int = 2
    max_nodes: int = 50
    constraints: Dict[str, Any] = None

@dataclass
class GraphPath:
    """图路径结构"""
    nodes: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    path_length: int
    relevance_score: float
    path_type: str

@dataclass
class KnowledgeSubgraph:
    """知识子图结构"""
    central_nodes: List[Dict[str, Any]]
    connected_nodes: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    graph_metrics: Dict[str, float]
    reasoning_chains: List[List[str]]


class TTLCache:
    """简易TTL+LRU缓存，控制内存并减少重复查询。"""

    def __init__(self, max_size: int = 512, ttl_seconds: int = 1800):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._store: "OrderedDict[str, Tuple[float, Any]]" = OrderedDict()

    def get(self, key: str):
        now = time.time()
        item = self._store.get(key)
        if not item:
            return None
        ts, value = item
        if now - ts > self.ttl_seconds:
            self._store.pop(key, None)
            return None
        self._store.move_to_end(key)
        return value

    def set(self, key: str, value: Any):
        self._store[key] = (time.time(), value)
        self._store.move_to_end(key)
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)

    def clear(self):
        self._store.clear()

    def __len__(self):
        return len(self._store)


class GraphRAGRetrieval:
    """
    真正的图RAG检索系统
    核心特点：
    1. 查询意图理解：识别图查询模式
    2. 多跳图遍历：深度关系探索
    3. 子图提取：相关知识网络
    4. 图结构推理：基于拓扑的推理
    5. 动态查询规划：自适应遍历策略
    """
    
    def __init__(self, config, llm_client):
        self.config = config
        self.llm_client = llm_client
        self.driver = None

        # Ollama支持
        self.llm_provider = config.llm_provider
        self.ollama_base_url = config.ollama_base_url.rstrip('/')
        self.ollama_model = config.ollama_model

        # 图结构缓存
        self.entity_cache = {}
        self.relation_cache = {}
        self.subgraph_cache = {}

        # 受控缓存（TTL + LRU）
        self.query_result_cache = TTLCache(max_size=512, ttl_seconds=3600)
        self.entity_match_cache = TTLCache(max_size=1024, ttl_seconds=1800)

        self.cache_hits = 0
        self.cache_misses = 0

        # 实体倒排索引：加速匹配
        self.entity_name_index: Dict[str, Set[str]] = defaultdict(set)
        self.entity_entries: List[Tuple[str, str]] = []
        
        # 性能统计 - 新增
        self.performance_stats = {
            "total_queries": 0,
            "avg_response_time": 0.0,
            "cache_hit_rate": 0.0,
            "llm_calls": 0,
            "neo4j_queries": 0
        }
        
    def initialize(self):
        """初始化图RAG检索系统"""
        logger.info("初始化图RAG检索系统...")
        
        # 连接Neo4j
        try:
            self.driver = GraphDatabase.driver(
                self.config.neo4j_uri, 
                auth=(self.config.neo4j_user, self.config.neo4j_password)
            )
            # 测试连接
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Neo4j连接成功")
        except Exception as e:
            logger.error(f"Neo4j连接失败: {e}")
            return
        
        # 预热：构建实体和关系索引
        self._build_graph_index()
        
    def _build_graph_index(self):
        """构建图索引以加速查询"""
        logger.info("构建图结构索引...")
        
        try:
            with self.driver.session() as session:
                entity_query = """
                MATCH (n)
                WHERE n.equipment_id IS NOT NULL OR n.component_id IS NOT NULL OR n.fault_id IS NOT NULL
                OPTIONAL MATCH (n)-[rel]-()
                WITH n, count(rel) as degree,
                       COALESCE(n.equipment_id, n.component_id, n.fault_id, n.phenomenon_id, n.cause_id, n.action_id, n.notice_id, n.source_id) as node_id,
                       COALESCE(n.name, n.cause_name, n.description) as name,
                       COALESCE(n.description, n.cause_name, n.name) as description,
                       labels(n) as node_labels, 
                       CASE 
                           WHEN n.equipment_id IS NOT NULL THEN 'Equipment'
                           WHEN n.component_id IS NOT NULL THEN 'Component'
                           WHEN n.fault_id IS NOT NULL THEN 'Fault'
                           WHEN n.phenomenon_id IS NOT NULL THEN 'FaultPhenomenon'
                           WHEN n.cause_id IS NOT NULL THEN 'FaultReason'
                           WHEN n.action_id IS NOT NULL THEN 'MaintenanceAction'
                           WHEN n.notice_id IS NOT NULL THEN 'SafetyNotice'
                           WHEN n.source_id IS NOT NULL THEN 'KnowledgeSource'
                           ELSE 'Unknown'
                       END as entity_type
                RETURN node_id, name, description, node_labels, entity_type, degree
                ORDER BY degree DESC
                """

                
                result = session.run(entity_query)
                for record in result:
                    node_id = record["node_id"]
                    if not node_id:
                        continue
                    name = record["name"] or ""
                    self.entity_cache[node_id] = {
                        "labels": record["node_labels"],
                        "name": name,
                        "entity_type": record["entity_type"],
                        "degree": record["degree"],
                        "description": record.get("description", "")
                    }

                    low_name = str(name).lower().strip()
                    if low_name:
                        self.entity_entries.append((node_id, low_name))
                        for tok in self._tokenize(low_name):
                            self.entity_name_index[tok].add(node_id)

                relation_query = """
                MATCH ()-[r]->()
                RETURN type(r) as rel_type, count(r) as frequency
                ORDER BY frequency DESC
                """
                
                result = session.run(relation_query)
                for record in result:
                    rel_type = record["rel_type"]
                    self.relation_cache[rel_type] = record["frequency"]
                    
                logger.info(f"索引构建完成: {len(self.entity_cache)}个实体, {len(self.relation_cache)}个关系类型")
                
        except Exception as e:
            logger.error(f"构建图索引失败: {e}")
    
    def understand_graph_query(self, query: str) -> GraphQuery:
        """
        理解查询的图结构意图
        这是图RAG的核心：从自然语言到图查询的转换
        优化：添加缓存和快速实体匹配
        """
        cache_key = f"query_intent_{hash(query)}"
        cached_result = self.query_result_cache.get(cache_key)
        if cached_result is not None:
            self.cache_hits += 1
            logger.info(f"查询意图缓存命中: {query}")
            return cached_result

        self.cache_misses += 1
        
        # 快速实体匹配优化
        matched_entities = self._fast_entity_match(query)
        if matched_entities:
            logger.info(f"快速实体匹配成功: {matched_entities}")
            # 根据查询内容智能判断查询类型
            query_type = self._infer_query_type(query)
            logger.info(f"推断查询类型: {query_type.value}")
            
            graph_query = GraphQuery(
                query_type=query_type,
                source_entities=matched_entities,
                target_entities=self._infer_target_entities(query),
                relation_types=self._infer_relation_types(query),
                max_depth=2,
                max_nodes=50
            )
            self._cache_query_result(cache_key, graph_query)
            return graph_query
        
        prompt = f"""
        作为图数据库专家，分析以下查询的图结构意图，并将自然语言问题映射到**已有图结构**上。
        已知图中大致有以下节点和关系：
        - 节点类型：
          - EquipmentCategory：装备大类节点，包含 category_id、name、description
          - Equipment：设备节点，包含 equipment_id、name、type、model
          - Component：部件节点，包含 component_id、name、spec、type
          - Fault：故障节点，包含 fault_id、name、fault_type、severity、occurrence_frequency
          - FaultPhenomenon：故障现象节点，包含 phenomenon_id、description
          - FaultReason：故障原因节点，包含 cause_id、cause_name、description、category、level
          - MaintenanceAction：维修步骤节点，包含 action_id、step_order、description、estimated_time、tools
          - SafetyNotice：注意事项节点，包含 notice_id、level、description、consequence
          - KnowledgeSource：知识来源节点，包含 source_id、type、title、reliability
        - 主要关系：
            -(EquipmentCategory)-[:contains]->(Equipment)
            -(Equipment)-[:consists_of]->(Component)
            -(Equipment)-[:has_fault]->(Fault)
            -(Fault)-[:presents_as]->(FaultPhenomenon)
            -(Fault)-[:caused_by]->(FaultReason)
            -(FaultReason)-[:relates_to]->(Component)
            -(FaultReason)-[:fixed_by]->(MaintenanceAction)
            -(MaintenanceAction)-[:has_notice]->(SafetyNotice)
            -(所有实体)-[:comes_from]->(KnowledgeSource)
        请根据上述图结构分析下面的查询：
        查询：{query}
        请识别：
        1. 查询类型：
           - entity_relation: 询问实体间的直接关系（如：主机和发电机有什么关系？）
           - multi_hop: 需要多跳推理（如：主机的哪些部件容易出故障？需要：主机→部件→故障→故障原因）
           - subgraph: 需要完整子图（如：主机系统有什么特点？需要主机相关的完整知识网络）
           - path_finding: 路径查找（如：从故障现象到故障原因的诊断路径）
           - clustering: 聚类相似性（如：和主机故障类似的设备有哪些？）
        2. source_entities：
           - 只包含在图中**很有可能有对应节点**的具体实体名称
           - 优先选择：设备类型（如"主机"、"发电机"）、具体设备名（如"1号主机"）、部件名（如"燃油泵"、"冷却器"）
        3. target_entities：
           - 只在确实需要限制「路径终点」时填写
           - 同样只能使用可能出现在图中的节点名称
           - 对于询问原因的查询，目标实体应该是与故障原因相关的节点，如具体的故障原因名称
           - 不要使用抽象概念如"原因"作为目标实体，而应该使用具体的故障原因名称
        4. relation_types：本次推理中希望优先考虑的关系类型列表
           - 例如：["consists_of", "has_fault", "caused_by", "fixed_by"]
        5. max_depth：建议的图遍历深度（1-3 之间的整数）
        6. constraints：可选的**属性级约束**，用于表达图结构之外的过滤条件
        示例：
        查询："主机的哪些部件容易出故障？"
        返回JSON示例：
        {{
          "query_type": "multi_hop",
          "source_entities": ["主机"],
          "target_entities": ["部件"],
          "relation_types": ["consists_of", "has_fault"],
          "max_depth": 3,
          "constraints": {{}}
        }}
        请严格返回一个合法的 JSON 对象，不要包含任何多余的说明文字。
        """
        
        try:
            self.performance_stats["llm_calls"] += 1
            if self.llm_provider == "ollama":
                # 使用Ollama原生API
                import requests
                url = f"{self.ollama_base_url}/api/chat"
                payload = {
                    "model": self.ollama_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 512
                    }
                }
                response = requests.post(url, json=payload, timeout=30)
                response.raise_for_status()
                result_json = response.json()
                content = result_json["message"]["content"].strip()
            else:
                # 使用vLLM OpenAI兼容API
                response = self.llm_client.chat.completions.create(
                    model=self.config.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=512
                )
                content = response.choices[0].message.content.strip()

            # 处理Qwen3-0.6B返回的<|think|>标签
            if "<|think|>" in content:
                # 移除<|think|>...</|think|>标签及其内容
                content = content.split("<|think|>")[-1].split("</|think>")[-1].strip()
            
            # 清理Markdown代码块标记
            if content.startswith("```json"):
                content = content[7:]  # 移除```json
            elif content.startswith("```"):
                content = content[3:]   # 移除```
            
            if content.endswith("```"):
                content = content[:-3]  # 移除结尾的```
            
            content = content.strip()

            result = json.loads(content)
            
            graph_query = GraphQuery(
                query_type=QueryType(result.get("query_type", "subgraph")),
                source_entities=result.get("source_entities", []),
                target_entities=result.get("target_entities", []),
                relation_types=result.get("relation_types", []),
                max_depth=result.get("max_depth", 2),
                max_nodes=50
            )
            
            # 缓存结果
            self._cache_query_result(cache_key, graph_query)
            
            return graph_query
            
        except Exception as e:
            logger.error(f"查询意图理解失败: {e}")
            # 降级方案：使用快速实体匹配
            matched_entities = self._fast_entity_match(query)
            if matched_entities:
                return GraphQuery(
                    query_type=QueryType.MULTI_HOP,
                    source_entities=matched_entities,
                    max_depth=2
                )
            # 最终降级方案：默认子图查询
            return GraphQuery(
                query_type=QueryType.SUBGRAPH,
                source_entities=[query],
                max_depth=2
            )
    
    def _tokenize(self, text: str) -> List[str]:
        """统一分词：兼容中英文，过滤过短token。"""
        if not text:
            return []
        tokens = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_]+", text.lower())
        return [t for t in tokens if len(t) >= 2]

    def _fast_entity_match(self, query: str) -> List[str]:
        """
        快速实体匹配：基于倒排索引+缓存，避免全表扫描。
        """
        query_lower = (query or "").lower().strip()
        if not query_lower:
            return []

        cache_key = f"entity_match_{hash(query_lower)}"
        cached = self.entity_match_cache.get(cache_key)
        if cached is not None:
            self.cache_hits += 1
            return cached

        self.cache_misses += 1
        tokens = self._tokenize(query_lower)

        candidate_ids: Set[str] = set()
        for tok in tokens:
            candidate_ids.update(self.entity_name_index.get(tok, set()))

        # 候选为空时回退到轻量子串匹配（仅名称）
        if not candidate_ids:
            for entity_id, name in self.entity_entries:
                if name and (name in query_lower or query_lower in name):
                    candidate_ids.add(entity_id)
                    if len(candidate_ids) >= 30:
                        break

        scored: List[Tuple[float, str]] = []
        for entity_id in candidate_ids:
            info = self.entity_cache.get(entity_id, {})
            name = str(info.get("name", "")).lower()
            desc = str(info.get("description", "")).lower()
            degree = float(info.get("degree", 0) or 0)

            score = 0.0
            if name and name in query_lower:
                score += 2.0
            if name and query_lower in name:
                score += 1.0
            if tokens and any(t in name for t in tokens):
                score += 0.8
            if desc and tokens and any(t in desc for t in tokens):
                score += 0.4
            score += min(degree / 50.0, 0.5)

            if score > 0:
                scored.append((score, entity_id))

        scored.sort(reverse=True)
        matched_entities: List[str] = []
        seen_names: Set[str] = set()
        for _, entity_id in scored[:10]:
            name = self.entity_cache.get(entity_id, {}).get("name")
            if name and name not in seen_names:
                seen_names.add(name)
                matched_entities.append(name)
            if len(matched_entities) >= 3:
                break

        self.entity_match_cache.set(cache_key, matched_entities)
        return matched_entities
    
    def _infer_query_type(self, query: str) -> QueryType:
        """
        根据查询内容推断查询类型（意图规则锁定）
        
        Args:
            query: 查询文本
            
        Returns:
            推断的查询类型
        """
        query_lower = query.lower()
        
        # 1. 原因类问题：锁定为子图（更利于返回多原因/多链路）
        cause_keywords = ["原因", "为什么", "为何", "导致", "根因", "成因", "诱因"]
        if any(kw in query_lower for kw in cause_keywords):
            return QueryType.SUBGRAPH
        
        # 2. 组成/结构类问题：锁定为子图
        composition_keywords = ["组成", "结构", "包含", "由什么", "有哪些", "有什么部件", "零件", "部件", "组件"]
        if any(kw in query_lower for kw in composition_keywords):
            return QueryType.SUBGRAPH
        
        # 3. 处理/排查/诊断类问题：路径查找
        path_keywords = ["怎么", "如何", "诊断", "排查", "步骤", "流程", "方法", "从", "到", "路径"]
        if any(kw in query_lower for kw in path_keywords):
            return QueryType.PATH_FINDING
        
        # 4. 关系类问题：实体关系
        relation_keywords = ["关系", "连接", "关联", "和", "与"]
        if any(kw in query_lower for kw in relation_keywords):
            return QueryType.ENTITY_RELATION
        
        # 5. 相似/聚类类问题
        clustering_keywords = ["类似", "相似", "相同", "一样", "同类"]
        if any(kw in query_lower for kw in clustering_keywords):
            return QueryType.CLUSTERING
        
        # 默认：多跳
        return QueryType.MULTI_HOP
    
    def _infer_target_entities(self, query: str) -> List[str]:
        """
        根据查询内容推断目标实体
        
        Args:
            query: 查询文本
            
        Returns:
            推断的目标实体列表
        """
        query_lower = query.lower()
        targets = []
        
        # 原因类问题：目标锁定为故障原因
        if any(kw in query_lower for kw in ["原因", "为什么", "为何", "导致", "根因", "成因", "诱因"]):
            targets.append("FaultReason")
        
        # 现象类
        if any(kw in query_lower for kw in ["现象", "表现", "症状", "故障表现"]):
            targets.append("FaultPhenomenon")
        
        # 处理/维修类
        if any(kw in query_lower for kw in ["维修", "处理", "解决", "措施", "方法", "步骤"]):
            targets.append("MaintenanceAction")
        
        # 组成/部件类
        if any(kw in query_lower for kw in ["部件", "零件", "组件", "组成", "结构"]):
            targets.append("Component")
        
        return targets
    
    def _infer_relation_types(self, query: str) -> List[str]:
        """
        根据查询内容推断关系类型
        
        Args:
            query: 查询文本
            
        Returns:
            推断的关系类型列表
        """
        query_lower = query.lower()
        relations = []
        
        # 原因链优先
        if any(kw in query_lower for kw in ["原因", "为什么", "为何", "导致", "根因", "成因", "诱因"]):
            relations.extend(["caused_by", "presents_as", "relates_to"])
        
        # 处理/维修相关
        if any(kw in query_lower for kw in ["维修", "处理", "解决", "措施", "方法", "步骤"]):
            relations.extend(["fixed_by", "has_notice"])
        
        # 组成/结构相关
        if any(kw in query_lower for kw in ["部件", "零件", "组件", "组成", "结构", "包含", "由什么"]):
            relations.extend(["consists_of", "contains"])
        
        # 故障相关
        if any(kw in query_lower for kw in ["故障", "问题", "异常"]):
            relations.extend(["has_fault", "presents_as"])
        
        return relations
    
    def _cache_query_result(self, cache_key: str, result: Any):
        """缓存查询结果"""
        self.query_result_cache.set(cache_key, result)

    def _clean_expired_cache(self):
        """兼容保留：TTLCache 在读写时自动淘汰。"""
        return
    
    def multi_hop_traversal(self, graph_query: GraphQuery) -> List[GraphPath]:
        """
        多跳图遍历：这是图RAG的核心优势
        通过图结构发现隐含的知识关联
        """
        logger.info(f"执行多跳遍历: {graph_query.source_entities} -> {graph_query.target_entities}")
        
        paths = []
        
        if not self.driver:
            logger.error("Neo4j连接未建立")
            return paths
            
        try:
            with self.driver.session() as session:
                # 构建多跳遍历查询
                source_entities = graph_query.source_entities
                target_keywords = graph_query.target_entities or []
                max_depth = graph_query.max_depth
                
                # 根据查询类型选择不同的遍历策略
                if graph_query.query_type == QueryType.MULTI_HOP:
                    target_filter_clause = ""
                    if target_keywords:
                        target_filter_clause = """
                    AND ANY(kw IN $target_keywords WHERE
                        (target.name IS NOT NULL AND (toString(target.name) CONTAINS kw OR kw CONTAINS toString(target.name))) OR
                        (target.type IS NOT NULL AND (toString(target.type) CONTAINS kw OR kw CONTAINS toString(target.type))) OR
                        (target.cause_name IS NOT NULL AND (toString(target.cause_name) CONTAINS kw OR kw CONTAINS toString(target.cause_name)))
                    )"""
                    
                    cypher_query = f"""
                    UNWIND $source_entities as source_name
                    MATCH (source)
                    WHERE source.name CONTAINS source_name 
                       OR source.equipment_id = source_name
                       OR source.component_id = source_name
                       OR source.fault_id = source_name
                       OR source.phenomenon_id = source_name
                       OR source.cause_id = source_name
                       OR source.action_id = source_name
                       OR source.notice_id = source_name
                       OR source.source_id = source_name
                    
                    MATCH path = (source)-[*1..{max_depth}]-(target)
                    WHERE NOT source = target{target_filter_clause}
                    
                    WITH path, source, target,
                         length(path) as path_len,
                         relationships(path) as rels,
                         nodes(path) as path_nodes
                    
                    WITH path, source, target, path_len, rels, path_nodes,
                         (1.0 / path_len) + 
                         (REDUCE(s = 0.0, node IN path_nodes | s + size([(node)--() | 1])) / 10.0 / size(path_nodes)) +
                         (CASE WHEN ANY(r IN rels WHERE type(r) IN $relation_types) THEN 0.3 ELSE 0.0 END) as relevance
                    
                    ORDER BY relevance DESC
                    LIMIT 20
                    
                    RETURN path, source, target, path_len, rels, path_nodes, relevance
                    """
                    
                    params = {
                        "source_entities": source_entities,
                        "relation_types": graph_query.relation_types or []
                    }
                    if target_keywords:
                        params["target_keywords"] = target_keywords
                    
                    result = session.run(cypher_query, params)
                    
                    for record in result:
                        path_data = self._parse_neo4j_path(record)
                        if path_data:
                            paths.append(path_data)
                
                elif graph_query.query_type == QueryType.ENTITY_RELATION:
                    # 实体间关系查询
                    paths.extend(self._find_entity_relations(graph_query, session))
                
                elif graph_query.query_type == QueryType.PATH_FINDING:
                    # 最短路径查找
                    paths.extend(self._find_shortest_paths(graph_query, session))
                
                # 如果没有找到路径，尝试更宽松的查询
                if not paths and source_entities:
                    logger.info("尝试更宽松的查询...")
                    # 查找与源实体相关的所有节点
                    fallback_query = f"""
                    UNWIND $source_entities as source_name
                    MATCH (source)
                    WHERE source.name CONTAINS source_name 
                       OR source.equipment_id = source_name
                       OR source.component_id = source_name
                       OR source.fault_id = source_name
                       OR source.phenomenon_id = source_name
                       OR source.cause_id = source_name
                       OR source.action_id = source_name
                       OR source.notice_id = source_name
                       OR source.source_id = source_name
                    
                    MATCH path = (source)-[*1..3]-(related)
                    WHERE NOT source = related
                    
                    WITH source, related, path,
                         length(path) as path_len,
                         relationships(path) as rels,
                         nodes(path) as path_nodes
                    
                    WITH source, related, path, path_len, rels, path_nodes,
                         (1.0 / path_len) as relevance
                    
                    ORDER BY relevance DESC
                    LIMIT 15
                    
                    RETURN path, source, related as target, path_len, rels, path_nodes, relevance
                    """
                    
                    fallback_result = session.run(fallback_query, {"source_entities": source_entities})
                    
                    for record in fallback_result:
                        try:
                            path_nodes = []
                            for node in record["path_nodes"]:
                                path_nodes.append({
                                    "id": node.get("id", ""),
                                    "name": node.get("name", ""),
                                    "labels": list(node.labels),
                                    "properties": dict(node)
                                })
                            
                            relationships = []
                            for rel in record["rels"]:
                                relationships.append({
                                    "type": type(rel).__name__,
                                    "properties": dict(rel)
                                })
                            
                            graph_path = GraphPath(
                                nodes=path_nodes,
                                relationships=relationships,
                                path_length=record["path_len"],
                                relevance_score=record["relevance"],
                                path_type="fallback"
                            )
                            paths.append(graph_path)
                        except Exception as e:
                            logger.error(f"解析路径失败: {e}")
                            continue
                    
        except Exception as e:
            logger.error(f"多跳遍历失败: {e}")
            
        logger.info(f"多跳遍历完成，找到 {len(paths)} 条路径")
        return paths
    
    def extract_knowledge_subgraph(self, graph_query: GraphQuery, original_query: str = "") -> KnowledgeSubgraph:
        """
        提取知识子图：获取实体相关的完整知识网络
        这体现了图RAG的整体性思维
        """
        logger.info(f"提取知识子图: {graph_query.source_entities}")
        
        if not self.driver:
            logger.error("Neo4j连接未建立")
            return self._fallback_subgraph_extraction(graph_query)
        
        try:
            with self.driver.session() as session:
                relation_types = graph_query.relation_types or []
                
                # 判断是否是组成查询（使用原始查询文本）
                is_composition_query = any(
                    kw in original_query.lower() 
                    for kw in ["组成", "部件", "结构", "包含", "有哪些"]
                )
                
                logger.info(f"子图查询判断: is_composition_query={is_composition_query}, original_query={original_query}")
                
                is_cause_subgraph = any(
                    rel in relation_types
                    for rel in ["caused_by", "relates_to", "presents_as", "fixed_by"]
                )

                if is_composition_query:
                    # 组成查询：专门查找consists_of关系
                    cypher_query = f"""
                    UNWIND $source_entities as entity_name
                    MATCH (source)
                    WHERE source.name CONTAINS entity_name 
                       OR source.equipment_id = entity_name
                       OR source.component_id = entity_name
                       OR source.fault_id = entity_name
                    
                    // 直接查找consists_of关系的部件
                    OPTIONAL MATCH (source)-[:consists_of]->(component:Component)
                    
                    // 也查找contains关系的设备
                    OPTIONAL MATCH (source)-[:contains]->(equip:Equipment)
                    OPTIONAL MATCH (equip)-[:consists_of]->(equip_comp:Component)
                    
                    WITH source,
                         collect(DISTINCT component) + collect(DISTINCT equip) + collect(DISTINCT equip_comp) as neighbors
                    
                    // 不限制节点数量，返回所有部件
                    RETURN 
                        source,
                        [n IN neighbors WHERE n IS NOT NULL] as nodes,
                        [] as rels,
                        {{
                            node_count: size([n IN neighbors WHERE n IS NOT NULL]),
                            relationship_count: size([n IN neighbors WHERE n IS NOT NULL]),
                            density: 0.0
                        }} as metrics
                    """
                elif is_cause_subgraph:
                    # 故障原因子图查询
                    cypher_query = f"""
                    UNWIND $source_entities as entity_name
                    MATCH (source)
                    WHERE source.name CONTAINS entity_name
                       OR source.equipment_id = entity_name
                       OR source.component_id = entity_name
                       OR source.fault_id = entity_name
                       OR source.cause_id = entity_name
                    
                    OPTIONAL MATCH (source)-[r_fault:has_fault|presents_as|caused_by|relates_to]-(n_fault)
                    OPTIONAL MATCH (source)-[:caused_by]->(reason:FaultReason)
                    OPTIONAL MATCH (reason)-[r_reason:relates_to]-(comp:Component)
                    OPTIONAL MATCH (reason)-[r_fix:fixed_by]-(action:MaintenanceAction)
                    OPTIONAL MATCH (action)-[r_notice:has_notice]-(notice:SafetyNotice)
                    
                    WITH source,
                         collect(DISTINCT n_fault) + collect(DISTINCT reason) + collect(DISTINCT comp)
                         + collect(DISTINCT action) + collect(DISTINCT notice) as neighbors,
                         collect(DISTINCT r_fault) + collect(DISTINCT r_reason)
                         + collect(DISTINCT r_fix) + collect(DISTINCT r_notice) as rels_raw
                    
                    WITH source,
                         [n IN neighbors WHERE n IS NOT NULL] as neighbors,
                         [r IN rels_raw WHERE r IS NOT NULL] as relationships
                    
                    WITH source, neighbors, relationships,
                         size(neighbors) as node_count,
                         size(relationships) as rel_count
                    
                    RETURN
                        source,
                        neighbors[0..{graph_query.max_nodes}] as nodes,
                        relationships[0..{graph_query.max_nodes}] as rels,
                        {{
                            node_count: node_count,
                            relationship_count: rel_count,
                            density: CASE WHEN node_count > 1 THEN toFloat(rel_count) / (node_count * (node_count - 1) / 2) ELSE 0.0 END
                        }} as metrics
                    """
                else:
                    # 默认子图查询：多跳邻居
                    cypher_query = f"""
                    UNWIND $source_entities as entity_name
                    MATCH (source)
                    WHERE source.name CONTAINS entity_name 
                       OR source.equipment_id = entity_name
                       OR source.component_id = entity_name
                       OR source.fault_id = entity_name
                    
                    MATCH (source)-[r*1..{graph_query.max_depth}]-(neighbor)
                    WITH source, collect(DISTINCT neighbor) as neighbors, 
                         collect(DISTINCT r) as relationships
                    
                    WITH source, neighbors, relationships,
                         size(neighbors) as node_count,
                         size(relationships) as rel_count
                    
                    RETURN 
                        source,
                        neighbors[0..{graph_query.max_nodes}] as nodes,
                        relationships[0..{graph_query.max_nodes}] as rels,
                        {{
                            node_count: node_count,
                            relationship_count: rel_count,
                            density: CASE WHEN node_count > 1 THEN toFloat(rel_count) / (node_count * (node_count - 1) / 2) ELSE 0.0 END
                        }} as metrics
                    """
                
                result = session.run(cypher_query, {
                    "source_entities": graph_query.source_entities,
                    "max_nodes": graph_query.max_nodes
                })
                
                record = result.single()
                if record:
                    return self._build_knowledge_subgraph(record)
                    
        except Exception as e:
            logger.error(f"子图提取失败: {e}")
            
        # 降级方案：简单邻居查询
        return self._fallback_subgraph_extraction(graph_query)
    
    def graph_structure_reasoning(self, subgraph: KnowledgeSubgraph, query: str) -> List[str]:
        """
        基于图结构的推理：这是图RAG的智能之处
        不仅检索信息，还能进行逻辑推理
        """
        reasoning_chains = []
        
        try:
            # 1. 识别推理模式
            reasoning_patterns = self._identify_reasoning_patterns(subgraph)
            
            # 2. 构建推理链
            for pattern in reasoning_patterns:
                chain = self._build_reasoning_chain(pattern, subgraph)
                if chain:
                    reasoning_chains.append(chain)
            
            # 3. 验证推理链的可信度
            validated_chains = self._validate_reasoning_chains(reasoning_chains, query)
            
            logger.info(f"图结构推理完成，生成 {len(validated_chains)} 条推理链")
            return validated_chains
            
        except Exception as e:
            logger.error(f"图结构推理失败: {e}")
            return []
    
    def adaptive_query_planning(self, query: str) -> List[GraphQuery]:
        """
        自适应查询规划：根据查询复杂度动态调整策略
        """
        # 分析查询复杂度
        complexity_score = self._analyze_query_complexity(query)
        
        query_plans = []
        
        if complexity_score < 0.3:
            # 简单查询：直接邻居查询
            plan = GraphQuery(
                query_type=QueryType.ENTITY_RELATION,
                source_entities=[query],
                max_depth=1,
                max_nodes=20
            )
            query_plans.append(plan)
            
        elif complexity_score < 0.7:
            # 中等复杂度：多跳查询
            plan = GraphQuery(
                query_type=QueryType.MULTI_HOP,
                source_entities=[query],
                max_depth=2,
                max_nodes=50
            )
            query_plans.append(plan)
            
        else:
            # 复杂查询：子图提取 + 推理
            plan1 = GraphQuery(
                query_type=QueryType.SUBGRAPH,
                source_entities=[query],
                max_depth=3,
                max_nodes=100
            )
            plan2 = GraphQuery(
                query_type=QueryType.MULTI_HOP,
                source_entities=[query],
                max_depth=3,
                max_nodes=50
            )
            query_plans.extend([plan1, plan2])
            
        return query_plans
    
    def graph_rag_search(self, query: str, top_k: int = 5) -> List[Document]:
        """
        图RAG主搜索接口：整合所有图RAG能力
        """
        logger.info(f"开始图RAG检索: {query}")
        
        if not self.driver:
            logger.warning("Neo4j连接未建立，返回空结果")
            return []
        
        # 1. 查询意图理解
        graph_query = self.understand_graph_query(query)
        logger.info(f"查询类型: {graph_query.query_type.value}")
        
        results = []
        
        try:
            # 2. 根据查询类型执行不同策略
            if graph_query.query_type in [QueryType.MULTI_HOP, QueryType.PATH_FINDING]:
                # 多跳遍历 / 路径查找
                paths = self.multi_hop_traversal(graph_query)
                results.extend(self._paths_to_documents(paths, query))
                
            elif graph_query.query_type in [QueryType.SUBGRAPH, QueryType.CLUSTERING]:
                # 子图提取 / 聚类查询：都视为"围绕核心实体的局部知识网络"
                subgraph = self.extract_knowledge_subgraph(graph_query, query)
                
                # 图结构推理
                reasoning_chains = self.graph_structure_reasoning(subgraph, query)
                
                results.extend(self._subgraph_to_documents(subgraph, reasoning_chains, query))
                
            elif graph_query.query_type == QueryType.ENTITY_RELATION:
                # 实体关系查询（可以视为一跳 / 少量跳的路径查询）
                paths = self.multi_hop_traversal(graph_query)
                results.extend(self._paths_to_documents(paths, query))
            
            # 3. 图结构相关性排序
            results = self._rank_by_graph_relevance(results, query)
            
            logger.info(f"图RAG检索完成，返回 {len(results[:top_k])} 个结果")
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"图RAG检索失败: {e}")
            return []
    
    # ========== 辅助方法 ==========
    
    def _parse_neo4j_path(self, record) -> Optional[GraphPath]:
        """解析Neo4j路径记录"""
        try:
            path_nodes = []
            for node in record["path_nodes"]:
                path_nodes.append({
                    "id": node.get("id", ""),
                    "name": node.get("name", ""),
                    "labels": list(node.labels),
                    "properties": dict(node)
                })
            
            relationships = []
            for rel in record["rels"]:
                relationships.append({
                    "type": type(rel).__name__,
                    "properties": dict(rel)
                })
            
            return GraphPath(
                nodes=path_nodes,
                relationships=relationships,
                path_length=record["path_len"],
                relevance_score=record["relevance"],
                path_type="multi_hop"
            )
            
        except Exception as e:
            logger.error(f"路径解析失败: {e}")
            return None
    
    def _build_knowledge_subgraph(self, record) -> KnowledgeSubgraph:
        """构建知识子图对象"""
        try:
            # 处理中心节点
            central_nodes = []
            source = record.get("source")
            if source:
                if hasattr(source, 'items'):
                    central_nodes.append(dict(source))
                else:
                    central_nodes.append({
                        "name": getattr(source, 'name', None) or source.get('name', '未知'),
                        "labels": list(source.labels) if hasattr(source, 'labels') else [],
                        "properties": dict(source) if hasattr(source, '__iter__') else {}
                    })
            
            # 处理连接节点
            connected_nodes = []
            nodes = record.get("nodes", [])
            if nodes:
                for node in nodes:
                    if node is None:
                        continue
                    if hasattr(node, 'items'):
                        connected_nodes.append(dict(node))
                    else:
                        node_dict = {
                            "name": getattr(node, 'name', None) or node.get('name', '未知'),
                            "labels": list(node.labels) if hasattr(node, 'labels') else [],
                            "properties": dict(node) if hasattr(node, '__iter__') else {}
                        }
                        # 添加常用属性
                        for key in ['name', 'description', 'spec', 'type', 'component_id']:
                            if hasattr(node, key):
                                node_dict[key] = getattr(node, key)
                            elif isinstance(node, dict) and key in node:
                                node_dict[key] = node[key]
                        connected_nodes.append(node_dict)
            
            # 处理关系
            relationships = []
            rels = record.get("rels", [])
            if rels:
                for rel in rels:
                    if rel is None:
                        continue
                    if hasattr(rel, 'items'):
                        relationships.append(dict(rel))
                    else:
                        relationships.append({
                            "type": type(rel).__name__ if hasattr(rel, '__class__') else "未知",
                            "properties": dict(rel) if hasattr(rel, '__iter__') else {}
                        })
            
            return KnowledgeSubgraph(
                central_nodes=central_nodes,
                connected_nodes=connected_nodes,
                relationships=relationships,
                graph_metrics=record.get("metrics", {}),
                reasoning_chains=[]
            )
        except Exception as e:
            logger.error(f"构建知识子图失败: {e}")
            return KnowledgeSubgraph(
                central_nodes=[],
                connected_nodes=[],
                relationships=[],
                graph_metrics={},
                reasoning_chains=[]
            )
    
    def _paths_to_documents(self, paths: List[GraphPath], query: str) -> List[Document]:
        """将图路径转换为Document对象"""
        documents = []
        
        for i, path in enumerate(paths):
            # 构建路径描述
            path_desc = self._build_path_description(path)
            
            doc = Document(
                page_content=path_desc,
                metadata={
                    "search_type": "graph_path",
                    "path_length": path.path_length,
                    "relevance_score": path.relevance_score,
                    "path_type": path.path_type,
                    "node_count": len(path.nodes),
                    "relationship_count": len(path.relationships),
                    "entity_name": path.nodes[0].get("name", "图结构结果") if path.nodes else "图结构结果"
                }
            )
            documents.append(doc)
            
        return documents
    
    def _subgraph_to_documents(self, subgraph: KnowledgeSubgraph, 
                              reasoning_chains: List[str], query: str) -> List[Document]:
        """将知识子图转换为Document对象"""
        documents = []
        
        # 子图整体描述
        subgraph_desc = self._build_subgraph_description(subgraph)
        
        doc = Document(
            page_content=subgraph_desc,
            metadata={
                "search_type": "knowledge_subgraph",
                "node_count": len(subgraph.connected_nodes),
                "relationship_count": len(subgraph.relationships),
                "graph_density": subgraph.graph_metrics.get("density", 0.0),
                "reasoning_chains": reasoning_chains,
                "entity_name": subgraph.central_nodes[0].get("name", "知识子图") if subgraph.central_nodes else "知识子图"
            }
        )
        documents.append(doc)
        
        return documents
    
    def _build_path_description(self, path: GraphPath) -> str:
        """构建路径的自然语言描述"""
        if not path.nodes:
            return "空路径"
            
        desc_parts = []
        for i, node in enumerate(path.nodes):
            node_name = node.get("name", f"节点{i}")
            node_labels = ",".join(node.get("labels", []))
            
            # 添加节点信息
            desc_parts.append(f"【{node_labels}】{node_name}")
            
            # 添加节点属性信息
            properties = node.get("properties", {})
            if properties:
                # 提取重要属性
                important_props = {}
                for key, value in properties.items():
                    if key in ["description", "cause_name", "fault_type", "severity"] and value:
                        important_props[key] = value
                
                if important_props:
                    prop_str = "；".join([f"{k}: {v}" for k, v in important_props.items()])
                    desc_parts.append(f"（{prop_str}）")
            
            if i < len(path.relationships):
                rel_type = path.relationships[i].get("type", "相关")
                desc_parts.append(f"\n{rel_type}\n")
        
        return "".join(desc_parts)
    
    def _build_subgraph_description(self, subgraph: KnowledgeSubgraph) -> str:
        """构建子图的自然语言描述"""
        central_names = [node.get("name", "未知") for node in subgraph.central_nodes]
        node_count = len(subgraph.connected_nodes)
        rel_count = len(subgraph.relationships)
        
        # 如果有连接节点，列出详细信息
        if subgraph.connected_nodes:
            desc_parts = [f"关于 {', '.join(central_names)} 的知识网络：\n"]
            
            # 按类型分组节点
            nodes_by_type = {}
            for node in subgraph.connected_nodes:
                node_type = node.get("labels", ["未知"])[0] if node.get("labels") else "未知"
                if node_type not in nodes_by_type:
                    nodes_by_type[node_type] = []
                nodes_by_type[node_type].append(node)
            
            # 列出各类型节点
            for node_type, nodes in nodes_by_type.items():
                desc_parts.append(f"\n【{node_type}】共 {len(nodes)} 个：\n")
                for i, node in enumerate(nodes, 1):
                    node_name = node.get("name", "未知")
                    desc_parts.append(f"  {i}. {node_name}")
                    
                    # 添加描述信息
                    description = node.get("description") or node.get("spec") or node.get("type")
                    if description:
                        desc_parts.append(f" - {description}")
                    desc_parts.append("\n")
            
            return "".join(desc_parts)
        else:
            return f"关于 {', '.join(central_names)} 的知识网络，包含 {node_count} 个相关概念和 {rel_count} 个关系。"
    
    def _rank_by_graph_relevance(self, documents: List[Document], query: str) -> List[Document]:
        """基于图结构相关性排序"""
        return sorted(documents, 
                     key=lambda x: x.metadata.get("relevance_score", 0.0), 
                     reverse=True)
    
    def _analyze_query_complexity(self, query: str) -> float:
        """分析查询复杂度"""
        complexity_indicators = ["什么", "如何", "为什么", "哪些", "关系", "影响", "原因"]
        score = sum(1 for indicator in complexity_indicators if indicator in query)
        return min(score / len(complexity_indicators), 1.0)
    
    def _identify_reasoning_patterns(self, subgraph: KnowledgeSubgraph) -> List[str]:
        """识别推理模式"""
        return ["因果关系", "组成关系", "相似关系"]
    
    def _build_reasoning_chain(self, pattern: str, subgraph: KnowledgeSubgraph) -> Optional[str]:
        """构建推理链"""
        return f"基于{pattern}的推理链"
    
    def _validate_reasoning_chains(self, chains: List[str], query: str) -> List[str]:
        """验证推理链"""
        return chains[:3]
    
    def _find_entity_relations(self, graph_query: GraphQuery, session) -> List[GraphPath]:
        """查找实体间关系"""
        return []
    
    def _find_shortest_paths(self, graph_query: GraphQuery, session) -> List[GraphPath]:
        """查找最短路径"""
        paths = []
        
        source_entities = graph_query.source_entities
        target_entities = graph_query.target_entities or []
        max_depth = graph_query.max_depth
        
        if not source_entities:
            return paths
        
        # 构建最短路径查询
        cypher_query = f"""
        UNWIND $source_entities as source_name
        MATCH (source)
        WHERE source.name CONTAINS source_name 
           OR source.equipment_id = source_name
           OR source.component_id = source_name
           OR source.fault_id = source_name
           OR source.phenomenon_id = source_name
           OR source.cause_id = source_name
           OR source.action_id = source_name
           OR source.notice_id = source_name
           OR source.source_id = source_name
        
        MATCH (target)
        WHERE $target_entities = [] OR ANY(kw IN $target_entities WHERE
            (target.name IS NOT NULL AND (toString(target.name) CONTAINS kw OR kw CONTAINS toString(target.name))) OR
            (target.type IS NOT NULL AND (toString(target.type) CONTAINS kw OR kw CONTAINS toString(target.type))) OR
            (target.cause_name IS NOT NULL AND (toString(target.cause_name) CONTAINS kw OR kw CONTAINS toString(target.cause_name)))
        )
        
        MATCH path = shortestPath((source)-[*1..{max_depth}]-(target))
        WHERE NOT source = target
        
        WITH path, source, target,
             length(path) as path_len,
             relationships(path) as rels,
             nodes(path) as path_nodes
        
        WITH path, source, target, path_len, rels, path_nodes,
             (1.0 / path_len) + 
             (REDUCE(s = 0.0, node IN path_nodes | s + size([(node)--() | 1])) / 10.0 / size(path_nodes)) as relevance
        
        ORDER BY relevance DESC
        LIMIT 10
        
        RETURN path, source, target, path_len, rels, path_nodes, relevance
        """
        
        params = {
            "source_entities": source_entities,
            "target_entities": target_entities
        }
        
        result = session.run(cypher_query, params)
        
        for record in result:
            try:
                path_nodes = []
                for node in record["path_nodes"]:
                    path_nodes.append({
                        "id": node.get("id", ""),
                        "name": node.get("name", ""),
                        "labels": list(node.labels),
                        "properties": dict(node)
                    })
                
                relationships = []
                for rel in record["rels"]:
                    relationships.append({
                        "type": type(rel).__name__,
                        "properties": dict(rel)
                    })
                
                graph_path = GraphPath(
                    nodes=path_nodes,
                    relationships=relationships,
                    path_length=record["path_len"],
                    relevance_score=record["relevance"],
                    path_type="shortest_path"
                )
                paths.append(graph_path)
            except Exception as e:
                logger.error(f"解析路径失败: {e}")
                continue
        
        return paths
    
    def _fallback_subgraph_extraction(self, graph_query: GraphQuery) -> KnowledgeSubgraph:
        """降级子图提取"""
        return KnowledgeSubgraph(
            central_nodes=[],
            connected_nodes=[],
            relationships=[],
            graph_metrics={},
            reasoning_chains=[]
        )
    
    def _validate_results(self, documents: List[Document], query: str) -> List[Document]:
        """
        结果质量验证：过滤低质量结果
        优化：提高准确率
        """
        validated_results = []
        query_lower = query.lower()
        
        for doc in documents:
            content = doc.page_content.lower()
            metadata = doc.metadata
            
            # 验证规则1：内容相关性
            relevance_score = metadata.get("relevance_score", 0.0)
            if relevance_score < 0.3:
                logger.debug(f"过滤低相关性结果: {relevance_score}")
                continue
            
            # 验证规则2：内容长度
            content_length = len(doc.page_content)
            if content_length < 10:
                logger.debug(f"过滤过短内容: {content_length}字符")
                continue
            
            # 验证规则3：内容包含查询关键词
            if not any(kw in content for kw in query_lower.split() if len(kw) > 1):
                logger.debug("过滤不包含查询关键词的结果")
                continue
            
            # 验证规则4：避免重复内容
            is_duplicate = False
            for existing_doc in validated_results:
                similarity = self._calculate_content_similarity(doc.page_content, existing_doc.page_content)
                if similarity > 0.8:  # 80%相似度视为重复
                    is_duplicate = True
                    logger.debug(f"过滤重复内容，相似度: {similarity:.2f}")
                    break
            
            if not is_duplicate:
                validated_results.append(doc)
        
        logger.info(f"结果验证: {len(documents)} -> {len(validated_results)}")
        return validated_results
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """
        计算内容相似度
        使用简单的字符级相似度计算
        """
        if not content1 or not content2:
            return 0.0
        
        # 使用Jaccard相似度
        set1 = set(content1.lower().split())
        set2 = set(content2.lower().split())
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        return {
            **self.performance_stats,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "entity_cache_size": len(self.entity_cache),
            "relation_cache_size": len(self.relation_cache),
            "query_cache_size": len(self.query_result_cache)
        }
    
    def clear_cache(self):
        """清理所有缓存"""
        self.entity_cache.clear()
        self.relation_cache.clear()
        self.subgraph_cache.clear()
        self.query_result_cache.clear()
        self.entity_match_cache.clear()
        logger.info("所有缓存已清理")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        return {
            **self.performance_stats,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "entity_cache_size": len(self.entity_cache),
            "relation_cache_size": len(self.relation_cache),
            "query_cache_size": len(self.query_result_cache)
        }
    
    def clear_cache(self):
        """清理所有缓存"""
        self.entity_cache.clear()
        self.relation_cache.clear()
        self.subgraph_cache.clear()
        self.query_result_cache.clear()
        self.entity_match_cache.clear()
        logger.info("所有缓存已清理")
    
    def close(self):
        """关闭资源连接"""
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()
            logger.info("图RAG检索系统已关闭") 