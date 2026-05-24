import logging
import json
import requests
import re
import os
import base64
from typing import Any, Dict, List, Optional, Union, Set
from neo4j import Driver
from neo4j import GraphDatabase
import time

logger = logging.getLogger(__name__)

class GraphDataInsert:
    """
    graph_insert 核心类
    负责：
    1. 案例 → LLM 结构化抽取
    2. 抽取结果 → Cypher 生成
    3. 执行 Cypher（需显式确认）
    4. 数据验证和错误处理
    5. 关系键值对创建
    """

    def __init__(self, llm_client,uri:str, user:str,password:str, config: Any,database: str = "neo4j", use_deepseek: bool = False, llm_provider: str = "vllm", ollama_model: str = "qwen2.5:7b", ollama_base_url: str = "http://localhost:11434"):
        self.llm_client = llm_client
        self.user = user
        self.password = password
        self.config = config
        self.use_deepseek = use_deepseek
        self.llm_provider = llm_provider
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url.rstrip('/')
        self.inserted_nodes = []
        self.inserted_relations = []
        self.driver = None
        self.database = database
        self.uri = self.config.neo4j_uri
        self._connect()
    
    def __del__(self):
        self.close()



    def _connect(self):
        """建立Neo4j连接"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
            )

            logger.info(f"成功连接到Neo4j数据库：{self.uri}")
            # 顺便做个测试

            with self.driver.session() as session:
                result = session.run("Return 1 as test")
                test_result = result.single()
                if test_result:
                    logger.info("测试成功")

        except Exception as e:
            logger.error(f"连接Neo4j失败：{e}")
            raise

    def close(self):
        #关闭数据库连接
        if hasattr(self, "driver") and self.driver:
            self.driver.close()
            logger.info("Neo4j连接已经关闭")

    def validate_case_data(self, data: Dict) -> tuple[bool, str]:
        """
        验证案例数据的有效性
        
        Args:
            data: 解析后的案例数据
            
        Returns:
            (是否有效, 错误信息)
        """
        required_fields = ["船舶装备", "装备零部件", "故障", "故障现象", "故障原因", "维修步骤", "注意事项"]
        
        for field in required_fields:
            if field not in data:
                return False, f"缺少必填字段: {field}"
        
        if not data["船舶装备"]:
            return False, "船舶装备不能为空"
        
        if not data["装备零部件"]:
            return False, "装备零部件不能为空"
        
        if not data["故障"]:
            return False, "故障不能为空"
        
        if not data["故障现象"]:
            return False, "故障现象不能为空"
        
        if not data["故障原因"]:
            return False, "故障原因不能为空"
        
        if not data["维修步骤"] or len(data["维修步骤"]) == 0:
            return False, "维修步骤不能为空"
        
        for step in data["维修步骤"]:
            if "步骤序号" not in step or "操作内容" not in step:
                return False, "维修步骤缺少必要字段"
        
        return True, ""

    def _is_mainly_chinese_text(self, text: str, threshold: float = 0.3) -> bool:
        """
        粗略判断文本是否以中文为主。
        说明：这是轻量启发式判断，不做复杂语种识别，足以用于“是否先翻译为中文”的分流。
        """
        if not text or not text.strip():
            return True

        # 只统计中英文/数字字符，避免标点噪声影响
        token_chars = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", text)
        if not token_chars:
            return True

        zh_chars = re.findall(r"[\u4e00-\u9fff]", text)
        zh_ratio = len(zh_chars) / max(len(token_chars), 1)
        return zh_ratio >= threshold

    def _translate_to_chinese(self, text: str, retry_count: int = 2, use_deepseek: bool = False) -> str:
        """
        使用当前 LLM 通道把任意语言文本翻译为中文。
        失败时返回原文，避免影响主流程可用性。
        """
        if not text or not text.strip():
            return text

        prompt = f"""
你是船舶维修领域的专业翻译助手。
请将下面文本完整翻译为简体中文，保留原始技术细节、型号、单位、步骤顺序和专业术语。
要求：
1. 只输出翻译后的中文内容，不要输出解释
2. 型号、编码、参数值（如 6S50ME-C、M12、220V）不要擅自改写
3. 如果原文已经是中文，仅做必要轻微润色并原样返回
原文：
{text}
"""

        for attempt in range(retry_count):
            try:
                if use_deepseek:
                    model = "/models/deepseek"
                elif self.llm_provider == "ollama":
                    model = self.ollama_model
                else:
                    model = self.config.llm_model

                logger.info(f"检测到非中文输入，开始翻译为中文 (尝试 {attempt + 1}/{retry_count})，模型: {model}")

                if self.llm_provider == "ollama":
                    url = f"{self.ollama_base_url}/api/chat"
                    payload = {
                        "model": self.ollama_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "options": {
                            "temperature": 0.0,
                            "num_predict": 2048
                        }
                    }
                    response = requests.post(url, json=payload, timeout=90)
                    response.raise_for_status()
                    result = response.json()
                    content = result["message"]["content"].strip()
                else:
                    resp = self.llm_client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=2048
                    )
                    content = resp.choices[0].message.content.strip()

                if content:
                    return content
            except Exception as e:
                logger.warning(f"翻译失败 (尝试 {attempt + 1}/{retry_count}): {e}")

        logger.warning("翻译步骤失败，回退为原文继续处理")
        return text

    def _contains_any_subject(self, text: str, subjects: List[str]) -> bool:
        """判断 text 是否已经包含任一主体词（设备/部件等），用于避免重复加前缀。"""
        t = (text or "").strip()
        if not t:
            return False
        for s in subjects:
            s2 = (s or "").strip()
            if s2 and s2 in t:
                return True
        return False

    def _contextualize_generic_term(
        self,
        term: str,
        subject: str,
        extra_subjects: Optional[List[str]] = None,
        generic_keywords: Optional[List[str]] = None,
    ) -> str:
        """
        将“泛化术语”绑定到主体上，例如：
        - "过热" -> "发电机过热"
        - "温度过高" -> "主机温度过高"

        触发条件（尽量保守）：
        - term 本身是泛化词（命中 generic_keywords）
        - 且 term 未包含主体（subject 或 extra_subjects 中任意一个）
        """
        t = (term or "").strip()
        if not t:
            return t

        subs = [subject] + (extra_subjects or [])
        if self._contains_any_subject(t, subs):
            return t

        if generic_keywords:
            if any(k in t for k in generic_keywords):
                return f"{subject}{t}"
            return t

        # 未提供关键词列表时，不做改写
        return t

    def _infer_system_name(self, equipment: str, equipment_type: str, fault_type: str = "") -> str:
        """
        推断装备所属系统（用于案例未显式给出“所属系统”时兜底）。
        返回值示例：动力系统 / 电力系统 / 液压系统 / 燃油系统 / 润滑系统 / 冷却系统 / 控制系统 / 结构系统 / 通用系统
        """
        text = f"{equipment} {equipment_type} {fault_type}".lower()
        text = text.replace(" ", "")

        # 顺序很重要：更具体的系统优先
        rules = [
            ("电力系统", ["发电机", "配电", "电力", "电气", "电机", "电缆", "蓄电池", "变压器", "断路器"]),
            ("动力系统", ["主机", "主发动机", "推进", "动力", "柴油机", "燃气轮机", "轴系"]),
            ("液压系统", ["液压", "油缸", "液压泵", "阀组", "液压站"]),
            ("燃油系统", ["燃油", "供油", "喷油", "油泵", "油路", "油嘴"]),
            ("润滑系统", ["润滑", "机油", "润滑油", "油压", "油温"]),
            ("冷却系统", ["冷却", "海水泵", "淡水泵", "换热器", "散热", "冷却水"]),
            ("控制系统", ["控制", "自动化", "plc", "传感器", "执行器", "联锁", "仪表"]),
            ("结构系统", ["结构", "船体", "舱壁", "支架", "基座", "焊缝"]),
        ]
        for system_name, keywords in rules:
            if any(k in text for k in keywords):
                return system_name
        return "通用系统"

    # =========================
    # 🔗 已有节点关联功能
    # =========================
    def query_existing_nodes(
        self,
        entity_type: str,
        name: str,
        similarity_threshold: float = 0.8,
        entity_attrs: Optional[Dict[str, Any]] = None,
    ) -> List[Dict]:
        """
        查询数据库中已有的相似节点
        
        Args:
            entity_type: 实体类型（Equipment, Component, Fault, FaultReason等）
            name: 实体名称
            similarity_threshold: 相似度阈值
            entity_attrs: 用于约束查询/合并的属性（不同实体类型可选）
            
        Returns:
            相似节点列表，每个节点包含 id, name, similarity
        """
        if not self.driver:
            logger.error("Neo4j连接未建立")
            return []
        
        try:
            with self.driver.session() as session:
                # 根据实体类型确定标签和ID字段
                type_config = {
                    "EquipmentCategory": {
                        "label": "EquipmentCategory",
                        "id_field": "category_id",
                        "name_field": "name",
                        "attr_fields": [],
                    },
                    "Equipment": {
                        "label": "Equipment",
                        "id_field": "equipment_id",
                        "name_field": "name",
                        "attr_fields": ["type", "model"],
                    },
                    "Component": {
                        "label": "Component",
                        "id_field": "component_id",
                        "name_field": "name",
                        "attr_fields": ["spec", "type"],
                    },
                    "Fault": {
                        "label": "Fault",
                        "id_field": "fault_id",
                        "name_field": "name",
                        "attr_fields": ["fault_type", "severity", "occurrence_frequency"],
                    },
                    "FaultPhenomenon": {
                        "label": "FaultPhenomenon",
                        "id_field": "phenomenon_id",
                        "name_field": "description",
                        "attr_fields": [],
                    },
                    "FaultReason": {
                        "label": "FaultReason",
                        "id_field": "cause_id",
                        "name_field": "cause_name",
                        "attr_fields": ["category", "level"],
                    },
                    "MaintenanceAction": {
                        "label": "MaintenanceAction",
                        "id_field": "action_id",
                        "name_field": "description",
                        "attr_fields": [],
                    },
                    "SafetyNotice": {
                        "label": "SafetyNotice",
                        "id_field": "notice_id",
                        "name_field": "description",
                        "attr_fields": [],
                    },
                }
                
                if entity_type not in type_config:
                    logger.warning(f"未知的实体类型: {entity_type}")
                    return []
                
                config = type_config[entity_type]
                
                # 查询该类型的节点（拉取必要属性用于合并判定）
                select_fields = [
                    f"n.{config['id_field']} as id",
                    f"n.{config['name_field']} as name",
                ]
                for af in config.get("attr_fields", []):
                    select_fields.append(f"n.{af} as {af}")

                query = f"""
                MATCH (n:{config['label']})
                RETURN {', '.join(select_fields)}
                """
                
                result = session.run(query)
                nodes = []
                
                for record in result:
                    node_name = record["name"]
                    if node_name:
                        similarity = self._calculate_similarity(name, node_name)
                        if similarity >= similarity_threshold:
                            # neo4j.Record 在不同驱动版本上对 get() 支持不一，这里用更稳妥的索引方式
                            attrs: Dict[str, Any] = {}
                            for af in config.get("attr_fields", []):
                                try:
                                    attrs[af] = record[af]
                                except Exception:
                                    attrs[af] = None
                            nodes.append({
                                "id": record["id"],
                                "name": node_name,
                                "similarity": similarity,
                                "attrs": attrs,
                            })
                
                # 按相似度排序
                nodes.sort(key=lambda x: x["similarity"], reverse=True)
                
                logger.info(f"查询到 {len(nodes)} 个相似节点 (类型: {entity_type}, 名称: {name})")
                return nodes
                
        except Exception as e:
            logger.error(f"查询已有节点失败: {e}")
            return []
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """
        计算两个字符串的相似度（基于编辑距离和关键词匹配）
        
        Args:
            str1: 字符串1
            str2: 字符串2
            
        Returns:
            相似度 (0-1)
        """
        if not str1 or not str2:
            return 0.0
        
        # 归一化
        str1 = str1.strip().lower()
        str2 = str2.strip().lower()
        
        # 完全匹配
        if str1 == str2:
            return 1.0
        
        # 包含关系
        if str1 in str2 or str2 in str1:
            return 0.9
        
        # 编辑距离相似度
        edit_sim = self._edit_distance_similarity(str1, str2)
        
        # 关键词匹配相似度
        keyword_sim = self._keyword_similarity(str1, str2)
        
        # 综合相似度（加权平均）
        return 0.4 * edit_sim + 0.6 * keyword_sim
    
    def _edit_distance_similarity(self, str1: str, str2: str) -> float:
        """计算编辑距离相似度"""
        m, n = len(str1), len(str2)
        if m == 0 or n == 0:
            return 0.0
        
        # 使用动态规划计算编辑距离
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i-1] == str2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        max_len = max(m, n)
        return 1.0 - (dp[m][n] / max_len)
    
    def _keyword_similarity(self, str1: str, str2: str) -> float:
        """计算关键词匹配相似度"""
        keywords1 = self._extract_keywords(str1)
        keywords2 = self._extract_keywords(str2)
        
        if not keywords1 or not keywords2:
            return 0.0
        
        intersection = keywords1 & keywords2
        union = keywords1 | keywords2
        
        return len(intersection) / len(union)
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """提取核心关键词"""
        stop_words = {
            "的", "了", "和", "与", "或", "及", "以及",
            "非常", "很", "太", "过于", "比较",
            "导致", "引起", "造成", "产生",
            "原因", "现象", "故障", "问题",
            "在", "上", "下", "中", "里",
        }
        
        keywords = set()
        for char in text:
            if char not in stop_words and char.strip():
                keywords.add(char)
        
        # 提取2-3字的关键词组合
        for i in range(len(text) - 1):
            phrase = text[i:i+2]
            if phrase not in stop_words:
                keywords.add(phrase)
        
        return keywords
    
    def _is_unknown_value(self, value: Any) -> bool:
        """判断字段是否属于“未知/无效值”（用于更保守的实体合并策略）"""
        if value is None:
            return True
        v = str(value).strip().lower()
        return v in {"", "未知", "unknown", "null", "none", "无", "nan"}

    def _normalize_attr_value(self, value: Any) -> str:
        """轻量归一化，减少空格/大小写差异带来的误判。"""
        if value is None:
            return ""
        v = str(value).strip().lower()
        # 去掉中英文空白字符
        v = "".join(v.split())
        return v

    def _should_merge_entities(
        self,
        name1: str,
        name2: str,
        similarity: float,
        entity_type: str,
        attrs1: Optional[Dict[str, Any]] = None,
        attrs2: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        判断是否应该合并两个实体
        
        合并条件：
        1. 相似度 > 0.95
        2. 不存在否定词冲突
        3. 核心关键词一致
        
        Args:
            name1: 实体名称1
            name2: 实体名称2
            similarity: 相似度
            entity_type: 实体类型
            attrs1: 待合并实体的属性（例如 Equipment 的 model/type）
            attrs2: 数据库中已有实体的属性
            
        Returns:
            是否应该合并
        """
        if similarity <= 0.95:
            return False

        attrs1 = attrs1 or {}
        attrs2 = attrs2 or {}

        # 针对“同名不同型号”这种高风险实体：强制型号/类型兼容后才允许合并
        if entity_type == "Equipment":
            new_model = attrs1.get("model")
            old_model = attrs2.get("model")
            if not self._is_unknown_value(new_model):
                # 新数据明确给了型号：数据库若也明确给了且不相等，则绝不合并
                if not self._is_unknown_value(old_model):
                    if self._normalize_attr_value(new_model) != self._normalize_attr_value(old_model):
                        return False
                # 数据库是未知型号：允许合并（后续由 cypher 补全未知字段）

            new_type = attrs1.get("type")
            old_type = attrs2.get("type")
            if not self._is_unknown_value(new_type):
                if not self._is_unknown_value(old_type):
                    if self._normalize_attr_value(new_type) != self._normalize_attr_value(old_type):
                        return False
        
        # 检查否定词冲突
        negation_pairs = [
            ("非", ""),
            ("不", ""),
            ("无", ""),
            ("未", ""),
            ("否", ""),
            ("人为", "非人为"),
            ("正常", "异常"),
            ("正常", "故障"),
        ]
        
        for neg_word, pos_word in negation_pairs:
            if (neg_word in name1 and pos_word in name2) or (pos_word in name1 and neg_word in name2):
                return False
        
        # 检查核心关键词一致性
        keywords1 = self._extract_keywords(name1)
        keywords2 = self._extract_keywords(name2)
        
        if not (keywords1 & keywords2):
            return False
        
        intersection = keywords1 & keywords2
        union = keywords1 | keywords2
        if len(intersection) / len(union) < 0.5:
            return False
        
        # 检查冲突动词
        conflict_verbs = [
            ("短路", "接反"),
            ("短路", "断路"),
            ("过热", "过冷"),
            ("卡住", "松动"),
            ("卡住", "脱落"),
        ]
        
        for verb1, verb2 in conflict_verbs:
            if (verb1 in name1 and verb2 in name2) or (verb2 in name1 and verb1 in name2):
                return False
        
        return True
    
    def find_or_create_entity(
        self,
        entity_type: str,
        entity_name: str,
        entity_attrs: Optional[Dict[str, Any]] = None
    ) -> Dict:
        """
        查找已有实体或创建新实体（插入时与库内对齐的主入口）。

        流程概览：
        1. 用 ``query_existing_nodes`` 按类型拉取同名/相似名节点（默认相似度 ≥0.8）；
        2. 通过 ``_should_merge_entities`` 判断是否可与已有节点合并；
        3. 若可合并：返回已有节点的业务 id（如 equipment_id），后续 Cypher 对该 id 做 MERGE，
           新案例会**挂到同一设备/故障/原因节点上**，只新增关系与属性，而不会凭空复制一个孤立实体；
        4. 若不可合并：分配新 id，MERGE 将创建新节点。

        若你希望「名称略不同也必须并到同一节点」，可适当**降低 similarity_threshold**或改进
        ``_calculate_similarity``（例如加嵌入相似度）；若希望幂等导入，也可在 Cypher 侧对
        ``name`` + 标签做 ``MERGE``，与业务 id 策略二选一或组合使用。
        
        Args:
            entity_type: 实体类型
            entity_name: 实体名称
            entity_attrs: 实体属性
            
        Returns:
            包含实体ID和是否新建的字典
        """
        entity_attrs = entity_attrs or {}
        # 查询相似节点
        similar_nodes = self.query_existing_nodes(
            entity_type,
            entity_name,
            similarity_threshold=0.8,
            entity_attrs=entity_attrs,
        )
        
        # 检查是否应该合并
        for node in similar_nodes:
            if self._should_merge_entities(
                entity_name,
                node["name"],
                node["similarity"],
                entity_type,
                attrs1=entity_attrs,
                attrs2=node.get("attrs") or {},
            ):
                logger.info(f"找到可合并的已有节点: {node['name']} (相似度: {node['similarity']:.3f})")
                return {
                    "id": node["id"],
                    "name": node["name"],
                    "is_new": False,
                    "similarity": node["similarity"],
                    "attrs": node.get("attrs") or {},
                }
        
        # 没有找到可合并的节点，创建新节点
        timestamp = int(time.time())
        
        type_id_prefix = {
            "EquipmentCategory": "ZB",
            "Equipment": "SB",
            "Component": "BJ",
            "Fault": "GZ",
            "FaultPhenomenon": "XX",
            "FaultReason": "YY",
            "MaintenanceAction": "WX",
            "SafetyNotice": "ZY",
        }
        
        prefix = type_id_prefix.get(entity_type, "XX")
        new_id = f"{prefix}{timestamp}"
        
        return {
            "id": new_id,
            "name": entity_name,
            "is_new": True,
            "similarity": 0.0,
            "attrs": entity_attrs,
        }

    def _is_image_file(self, path: str) -> bool:
        ext = os.path.splitext(path.lower())[1]
        return ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def _is_pdf_file(self, path: str) -> bool:
        return os.path.splitext(path.lower())[1] == ".pdf"

    def _looks_like_file_path(self, value: str) -> bool:
        if not value:
            return False
        candidate = value.strip().strip('"').strip("'")
        return os.path.exists(candidate) and os.path.isfile(candidate)

    def _extract_json_from_text(self, content: str) -> Dict:
        if not content:
            raise ValueError("空响应，无法解析 JSON")
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            m = re.search(r"\{[\s\S]*\}", content)
            if not m:
                raise
            return json.loads(m.group(0))

    def _analyze_case_from_images_with_ollama(self, image_paths: List[str], retry_count: int = 2) -> Dict:
        """多模态路径：使用 Ollama 直接做 OCR + 结构化总结。"""
        valid_paths = [p for p in image_paths if p and os.path.exists(p)]
        if not valid_paths:
            raise ValueError("未提供有效图片路径")

        max_images = 8
        used_paths = valid_paths[:max_images]
        images_b64 = []
        for p in used_paths:
            with open(p, "rb") as f:
                images_b64.append(base64.b64encode(f.read()).decode("utf-8"))

        prompt = """
你是船舶装备维修领域的资深工程师。
请先对图片中的文本进行OCR识别，再结合图文内容做结构化抽取。
要求：
1) 只输出合法 JSON，不要输出解释。
2) 不得编造；缺失信息用“未知/无/[]”。
3) 字段必须完整，结构如下：
{
  "船舶装备": "",
  "所属系统": "",
  "装备类型": "",
  "装备型号": "",
  "装备零部件": [],
  "故障": "",
  "故障类型": "",
  "故障严重等级": "高/中/低",
  "故障发生频率": "高频/中频/低频",
  "故障现象": [],
  "故障原因": [],
  "诊断依据": {
    "检测方法": [],
    "关键数据": [],
    "定位结论": ""
  },
  "工况环境": {
    "运行阶段": "",
    "负载状态": "",
    "环境影响": ""
  },
  "维修步骤": [
    {
      "步骤序号": 1,
      "操作内容": "",
      "所需工具": "",
      "所需备件": "",
      "操作时长": "",
      "关键检查点": ""
    }
  ],
  "维修结果与验证": {
    "是否恢复": "是/否/未知",
    "验证方式": [],
    "遗留问题": ""
  },
  "注意事项": [],
  "知识来源": {
    "书名": "源于用户上传图像OCR提取",
    "章节": "",
    "小节": ""
  }
}
""".strip()

        url = f"{self.ollama_base_url}/api/chat"
        for attempt in range(retry_count):
            try:
                payload = {
                    "model": self.ollama_model,
                    "messages": [{"role": "user", "content": prompt, "images": images_b64}],
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 2048},
                }
                resp = requests.post(url, json=payload, timeout=180)
                resp.raise_for_status()
                content = resp.json().get("message", {}).get("content", "").strip()
                data = self._extract_json_from_text(content)
                ok, err = self.validate_case_data(data)
                if ok:
                    return data
                if attempt == retry_count - 1:
                    raise ValueError(f"多模态抽取数据验证失败: {err}")
            except Exception as e:
                if attempt == retry_count - 1:
                    raise
                logger.warning(f"多模态抽取失败，重试中 ({attempt + 1}/{retry_count}): {e}")

        raise RuntimeError("多模态抽取失败")

    def _looks_like_file_path(self, text: str) -> bool:
        t = (text or "").strip().strip('"').strip("'")
        if not t:
            return False
        if any(t.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".pdf"]):
            return True
        return os.path.exists(t)

    def _is_image_file(self, path: str) -> bool:
        p = (path or "").strip().strip('"').strip("'")
        return os.path.isfile(p) and any(p.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"])

    def _is_pdf_file(self, path: str) -> bool:
        p = (path or "").strip().strip('"').strip("'")
        return os.path.isfile(p) and p.lower().endswith(".pdf")

    def _analyze_case_from_images_with_ollama(self, image_paths: List[str], retry_count: int = 2) -> Dict:
        if not image_paths:
            raise ValueError("未提供图片文件")

        prompt = """
你是船舶装备维修领域的信息抽取助手。
请基于图片中的文本（OCR）进行结构化抽取，并严格仅输出 JSON。
字段要求与下述 schema 一致，缺失内容请用“未知/无/[]”填充，不要编造。

{
  "船舶装备": "",
  "所属系统": "",
  "装备类型": "",
  "装备型号": "",
  "装备零部件": [],
  "故障": "",
  "故障类型": "",
  "故障严重等级": "",
  "故障发生频率": "",
  "故障现象": [],
  "故障原因": [],
  "诊断依据": {"检测方法": [], "关键数据": [], "定位结论": ""},
  "工况环境": {"运行阶段": "", "负载状态": "", "环境影响": ""},
  "维修步骤": [{"步骤序号": 1, "操作内容": "", "所需工具": "", "所需备件": "", "操作时长": "", "关键检查点": ""}],
  "维修结果与验证": {"是否恢复": "", "验证方式": [], "遗留问题": ""},
  "注意事项": [],
  "知识来源": {"书名": "源于用户上传资料", "章节": "", "小节": ""}
}
"""

        for attempt in range(retry_count):
            try:
                b64_images = []
                for p in image_paths:
                    with open(p, "rb") as f:
                        b64_images.append(base64.b64encode(f.read()).decode("utf-8"))

                payload = {
                    "model": self.ollama_model,
                    "messages": [{"role": "user", "content": prompt, "images": b64_images}],
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 2048}
                }
                resp = requests.post(f"{self.ollama_base_url}/api/chat", json=payload, timeout=180)
                resp.raise_for_status()
                content = (resp.json().get("message", {}) or {}).get("content", "").strip()

                if not content:
                    continue

                if '<|think|>' in content:
                    content = content.split('<|think|>')[-1].split('</|think>')[-1].strip()

                try:
                    data = json.loads(content)
                except Exception:
                    m = re.search(r'\{[\s\S]*\}', content)
                    if not m:
                        continue
                    data = json.loads(m.group(0))

                is_valid, error_msg = self.validate_case_data(data)
                if not is_valid:
                    logger.warning(f"多模态抽取结果校验失败: {error_msg}")
                    continue
                return data
            except Exception as e:
                logger.warning(f"多模态抽取失败 (尝试 {attempt + 1}/{retry_count}): {e}")

        raise RuntimeError("多模态OCR结构化抽取失败")

    # =========================
    # 1️⃣ LLM 结构化抽取
    # =========================
    def analyze_case(self, case: str, retry_count: int = 3, use_deepseek: bool = False, image_paths: Optional[List[str]] = None) -> Dict:
        """
        使用LLM分析案例并提取结构化信息
        
        Args:
            case: 案例文本
            retry_count: 重试次数
            
        Returns:
            解析后的结构化数据
            
        Raises:
            RuntimeError: 当LLM解析失败时
        """
        # 多模态优先：显式传入 image_paths 时优先走
        if self.llm_provider == "ollama" and image_paths:
            valid_imgs = [p for p in image_paths if self._is_image_file(p)]
            if valid_imgs:
                return self._analyze_case_from_images_with_ollama(valid_imgs, retry_count=max(1, retry_count))

        # 兼容旧接口：case 可能直接是图片路径
        case_str = (case or "").strip()
        if self.llm_provider == "ollama" and self._looks_like_file_path(case_str):
            p = case_str.strip('"').strip("'")
            if self._is_image_file(p):
                return self._analyze_case_from_images_with_ollama([p], retry_count=max(1, retry_count))
            if self._is_pdf_file(p):
                raise ValueError("当前接口暂未实现PDF直读，请先转为图片后传入 image_paths")

        # 预处理：若输入不是中文为主，则先翻译成中文后再结构化抽取
        preprocessed_case = case
        if not self._is_mainly_chinese_text(case):
            preprocessed_case = self._translate_to_chinese(case, use_deepseek=use_deepseek)

        prompt = f"""
你是船舶装备维修领域的资深工程师，精通从维修记录中进行“结构化抽取”，并能将结果用于 Neo4j 图数据库入库。
请从【船舶维修案例】中抽取信息，严格按 JSON 输出（只输出 JSON，不要输出任何解释/Markdown）。

【抽取目标】
- 抽取“设备/部件/故障/现象/原因/步骤/注意事项/来源”，并补充关键的“诊断依据、工况、验证结果、备件与工具信息”。
- 绝对禁止凭空编造：案例里没有明确提及的内容，一律填入“未知/无/[]”，不要猜测。

【案例原文】
{preprocessed_case}

【输出 JSON Schema（字段必须全部出现，允许用 未知/无/[] 占位）】
{{
  "船舶装备": "<设备名称，尽量用名词短语；未知则写'未知设备'>",
  "所属系统": "<动力系统/电力系统/液压系统/燃油系统/润滑系统/冷却系统/控制系统/结构系统/通用系统；未知可写'未知'>",
  "装备类型": "<动力设备/电气设备/液压设备/...；未知写'未知'>",
  "装备型号": "<型号/规格/系列号；没有提及写'未知'>",

  "装备零部件": ["<部件1>", "<部件2>"],

  "故障": "<故障名称（名词短语），例如'启动失败/过热/无法供油'；未知写'未知故障'>",
  "故障类型": "<机械/电气/液压/控制/结构/润滑/冷却/...；未知写'机械'>",
  "故障严重等级": "<高/中/低；若未提及，用'中'>",
  "故障发生频率": "<高频/中频/低频；若未提及，用'中频'>",

  "故障现象": ["<现象1>", "<现象2>"],
  "故障原因": ["<原因1>", "<原因2>"],

  "诊断依据": {{
    "检测方法": ["<测量/检查/试验方法，未知用[]>"],
    "关键数据": ["<如温度/压力/电流/报警码等；未知用[]>"],
    "定位结论": "<如'故障集中在XX部位/XX回路'；未知写'未知'>"
  }},

  "工况环境": {{
    "运行阶段": "<航行/靠泊/启机/停机/切换/负载变化等；未知写'未知'>",
    "负载状态": "<轻载/重载/满载/变负载；未知写'未知'>",
    "环境影响": "<海况/温度/盐雾/振动等；未知写'未知'>"
  }},

  "维修步骤": [
    {{
      "步骤序号": 1,
      "操作内容": "<必须是可执行动作，不要写结论句>",
      "所需工具": "<工具/仪表；未知可空字符串>",
      "所需备件": "<备件/材料；未知可空字符串>",
      "操作时长": "<如10分钟/2小时；未知可空字符串>",
      "关键检查点": "<该步完成后要检查/确认的点；未知写'未知'>"
    }}
  ],

  "维修结果与验证": {{
    "是否恢复": "<是/否/未知>",
    "验证方式": ["<试车/复测/运行观察等；未知用[]>"],
    "遗留问题": "<若有遗留风险/建议复检；未知写'未知'>"
  }},

  "注意事项": ["<安全/操作风险提示；没有写[]>"],

  "知识来源": {{
    "书名": "<手册/规程/记录来源；不可知则写'源于用户输入的案例文本'>",
    "章节": "<章节；未知写''>",
    "小节": "<小节；未知写''>"
  }}
}}

【关键约束（非常重要）】
1) 只输出 JSON；不要输出任何额外文字。
2) 数组字段必须输出 JSON 数组（例如 装备零部件/故障现象/故障原因/注意事项/检测方法/关键数据/验证方式）。
3) 不要把多个条目用逗号塞进一个字符串里（要拆成数组）。
4) “未知/无/[]”是允许且推荐的占位方式；不要为了不空而编造。
5) 维修步骤至少 1 条；每条必须包含“步骤序号”和“操作内容”。
6) 严重等级只能是 高/中/低；发生频率只能是 高频/中频/低频。
"""

        for attempt in range(retry_count):
            try:
                # 选择模型
                if use_deepseek:
                    model = "/models/deepseek"
                elif self.llm_provider == "ollama":
                    model = self.ollama_model
                else:
                    model = self.config.llm_model
                logger.info(f"开始调用 LLM 分析案例 (尝试 {attempt + 1}/{retry_count})，使用模型: {model}")
                
                if self.llm_provider == "ollama":
                    # 使用Ollama原生API
                    url = f"{self.ollama_base_url}/api/chat"
                    payload = {
                        "model": self.ollama_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "options": {
                            "temperature": 0.1,
                            "num_predict": 1024
                        }
                    }
                    response = requests.post(url, json=payload, timeout=60)
                    response.raise_for_status()
                    result = response.json()
                    content = result["message"]["content"].strip()
                else:
                    # 使用vLLM OpenAI兼容API
                    resp = self.llm_client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                        max_tokens=1024
                    )
                    content = resp.choices[0].message.content.strip()
                
                logger.debug(f"LLM 返回内容: {content[:200]}...")  # 只打印前200个字符
                
                # 检查内容是否为空
                if not content:
                    logger.error(f"LLM 返回空内容 (尝试 {attempt + 1}/{retry_count})")
                    if attempt == retry_count - 1:
                        raise RuntimeError("LLM 返回空内容，无法解析")
                    continue
                
                # 移除<|think|>标签内容
                if '<|think|>' in content:
                    logger.debug("移除<|think|>标签内容")
                    # 移除<|think|>...</|think|>标签及其内容
                    content = content.split('<|think|>')[-1].split('</|think>')[-1].strip()
                    logger.debug(f"处理后内容: {content[:200]}...")
                
                # 尝试解析JSON
                try:
                    data = json.loads(content)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON解析失败 (尝试 {attempt + 1}/{retry_count}): {e}")
                    logger.error(f"原始内容: {content}")
                    if attempt == retry_count - 1:
                        # 尝试从内容中提取JSON部分
                        import re
                        json_match = re.search(r'\{[\s\S]*\}', content)
                        if json_match:
                            try:
                                data = json.loads(json_match.group(0))
                                logger.warning("从非标准响应中提取到JSON")
                            except:
                                raise RuntimeError(f"LLM解析失败，无法解析JSON: {e}")
                        else:
                            raise RuntimeError(f"LLM解析失败，无法解析JSON: {e}")
                    else:
                        continue
                
                # 验证数据
                is_valid, error_msg = self.validate_case_data(data)
                if not is_valid:
                    if attempt < retry_count - 1:
                        logger.warning(f"数据验证失败: {error_msg}, 重试 {attempt + 1}/{retry_count}")
                        continue
                    raise ValueError(f"数据验证失败: {error_msg}")
                
                logger.info("LLM案例解析成功")
                return data

            except json.JSONDecodeError as e:
                logger.error(f"JSON解析失败 (尝试 {attempt + 1}/{retry_count}): {e}")
                if attempt == retry_count - 1:
                    raise RuntimeError(f"LLM解析失败，无法解析JSON: {e}")
            except Exception as e:
                logger.exception(f"LLM 案例解析失败 (尝试 {attempt + 1}/{retry_count})")
                if attempt == retry_count - 1:
                    raise RuntimeError(f"LLM 解析失败: {e}")


    def build_cypher(self, data: Dict, case_key: Optional[str] = None) -> List[str]:
        """
        构建Cypher语句（适配kg_generator_v2.py的实体类型和关系定义）
        
        实体类型：
        - EquipmentCategory: 装备大类
        - Equipment: 设备
        - Component: 部件
        - Fault: 故障
        - FaultPhenomenon: 故障现象
        - FaultReason: 故障原因
        - MaintenanceAction: 维修步骤
        - SafetyNotice: 注意事项
        - KnowledgeSource: 知识来源
        
        关系类型：
        - CONTAINS: EquipmentCategory -> Equipment
        - CONSISTS_OF: Equipment -> Component
        - HAS_FAULT: Equipment -> Fault
        - PRESENTS_AS: Fault -> FaultPhenomenon
        - CAUSED_BY: Fault -> FaultReason
        - RELATES_TO: FaultReason -> Component
        - FIXED_BY: FaultReason -> MaintenanceAction
        - HAS_NOTICE: MaintenanceAction -> SafetyNotice
        - BELONGS_TO: Equipment -> EquipmentCategory
        - COMES_FROM: 所有实体 -> KnowledgeSource
        
        Args:
            data: 解析后的案例数据
            
        Returns:
            Cypher语句列表
        """
        cyphers = []

        equipment = data["船舶装备"]
        system_name = data.get("所属系统", "未知")
        equipment_type = data.get("装备类型", "未知")
        equipment_model = data.get("装备型号", "未知")
        
        parts_data = data["装备零部件"]
        fault_name = data.get("故障", "未知故障")
        fault_type = data.get("故障类型", "机械")
        severity = data.get("故障严重等级", "中")
        frequency = data.get("故障发生频率", "中频")
        phenomena_data = data["故障现象"]
        causes_data = data["故障原因"]
        
        if isinstance(parts_data, dict):
            parts = list(parts_data.keys())
        else:
            parts = self._split(parts_data)
        
        if isinstance(phenomena_data, dict):
            phenomena = list(phenomena_data.keys())
        else:
            phenomena = self._split(phenomena_data)
        
        if isinstance(causes_data, dict):
            causes = list(causes_data.keys())
        else:
            causes = self._split(causes_data)

        # 兜底系统归属：若未明确给出，则基于设备名/装备类型/故障类型推断
        if not system_name or str(system_name).strip() in {"未知", "无"}:
            system_name = self._infer_system_name(equipment, equipment_type, fault_type)

        # ---- 关键：为“泛化故障/现象”补充主体，避免跨设备/部件误合并或语义污染 ----
        generic_phenomenon_keywords = [
            "过热", "温度过高", "高温", "温度升高",
            "异响", "噪音", "振动",
            "泄漏", "渗漏",
            "压力低", "压力不足", "压力过低", "压力过高",
            "流量低", "流量不足",
            "电流过大", "电流过高", "电压过低", "电压过高",
            "报警", "报警码",
        ]
        generic_fault_keywords = [
            "启动失败", "无法启动", "起动失败",
            "过热", "温度过高",
            "停机", "自动停机", "跳闸",
            "功率不足", "无输出", "输出异常",
        ]

        # 主体：优先用设备名；若现象已包含某部件名则不再加设备前缀
        subject_equipment = equipment
        subject_parts = parts if isinstance(parts, list) else []

        fault_name = self._contextualize_generic_term(
            fault_name,
            subject=subject_equipment,
            extra_subjects=subject_parts,
            generic_keywords=generic_fault_keywords,
        )
        phenomena = [
            self._contextualize_generic_term(
                fp,
                subject=subject_equipment,
                extra_subjects=subject_parts,
                generic_keywords=generic_phenomenon_keywords,
            )
            for fp in phenomena
        ]
        
        notices = data.get("注意事项", [])
        knowledge_source = data.get("知识来源", {})

        # 使用 uuid 生成“本次案例”的唯一标识，避免同一秒内多次导入导致节点 id 冲突（从而覆盖不同型号数据）
        if case_key is None:
            import uuid
            case_key = uuid.uuid4().hex

        source_id = f"KS{case_key}"
        book_name = knowledge_source.get("书名", "未知")
        chapter = knowledge_source.get("章节", "")
        section = knowledge_source.get("小节", "")
        
        # 1. 创建知识来源节点
        cyphers.append(
            f"""
            MERGE (ks:KnowledgeSource {{source_id: '{source_id}'}})
            SET ks.type = '手册', ks.title = '{book_name}',
                ks.chapter = '{chapter}', ks.section = '{section}',
                ks.reliability = '中';
            """.strip()
        )

        # 2. 创建装备大类节点
        category_id = f"ZB{case_key}"
        category_name = equipment_type
        cyphers.append(
            f"""
            MERGE (ec:EquipmentCategory {{category_id: '{category_id}'}})
            SET ec.name = '{category_name}', ec.description = '{category_name}相关设备',
                ec.system_name = '{system_name}',
                ec.source_id = '{source_id}';
            """.strip()
        )

        # 3. 创建设备节点
        equipment_id = f"SB{case_key}"
        cyphers.append(
            f"""
            MERGE (e:Equipment {{equipment_id: '{equipment_id}'}})
            SET e.name = '{equipment}', e.system_name = '{system_name}',
                e.type = '{equipment_type}', e.model = '{equipment_model}',
                e.source_id = '{source_id}';
            """.strip()
        )
        
        # 4. 创建装备大类与设备的关系
        cyphers.append(
            f"""
            MATCH (ec:EquipmentCategory {{category_id: '{category_id}'}})
            MATCH (e:Equipment {{equipment_id: '{equipment_id}'}})
            MERGE (ec)-[:CONTAINS]->(e);
            """.strip()
        )

        # 5. 创建部件节点及关系
        for i, p in enumerate(parts):
            component_id = f"BJ{case_key}{i+1:03d}"
            cyphers.append(
                f"""
                MERGE (c:Component {{component_id: '{component_id}'}})
                SET c.name = '{p}', c.spec = '未知', c.type = '部件',
                    c.source_id = '{source_id}'
                MERGE (e:Equipment {{equipment_id: '{equipment_id}'}})
                MERGE (e)-[:CONSISTS_OF]->(c);
                """.strip()
            )

        # 6. 创建故障节点及关系
        fault_id = f"GZ{case_key}"
        cyphers.append(
            f"""
            MERGE (f:Fault {{fault_id: '{fault_id}'}})
            SET f.name = '{fault_name}', f.fault_type = '{fault_type}',
                f.severity = '{severity}', f.occurrence_frequency = '{frequency}',
                f.description = '{fault_name}', f.source_id = '{source_id}'
            MERGE (e:Equipment {{equipment_id: '{equipment_id}'}})
            MERGE (e)-[:HAS_FAULT]->(f);
            """.strip()
        )

        # 7. 创建故障现象节点及关系
        for i, fp in enumerate(phenomena):
            phenomenon_id = f"XX{case_key}{i+1:03d}"
            cyphers.append(
                f"""
                MERGE (fp:FaultPhenomenon {{phenomenon_id: '{phenomenon_id}'}})
                SET fp.description = '{fp}', fp.source_id = '{source_id}'
                MERGE (f:Fault {{fault_id: '{fault_id}'}})
                MERGE (f)-[:PRESENTS_AS]->(fp);
                """.strip()
            )

        # 8. 创建故障原因节点及关系
        cause_ids = []
        for i, cause in enumerate(causes):
            cause_id = f"YY{case_key}{i+1:03d}"
            cause_ids.append(cause_id)
            cause_category = "老化"
            cyphers.append(
                f"""
                MERGE (fr:FaultReason {{cause_id: '{cause_id}'}})
                SET fr.cause_name = '{cause}', fr.description = '{cause}',
                    fr.category = '{cause_category}', fr.level = 0,
                    fr.source_id = '{source_id}'
                MERGE (f:Fault {{fault_id: '{fault_id}'}})
                MERGE (f)-[:CAUSED_BY]->(fr);
                """.strip()
            )
            
            # 9. 创建故障原因与部件的关系
            if parts:
                component_id = f"BJ{case_key}001"
                cyphers.append(
                    f"""
                    MATCH (fr:FaultReason {{cause_id: '{cause_id}'}})
                    MATCH (c:Component {{component_id: '{component_id}'}})
                    MERGE (fr)-[:RELATES_TO]->(c);
                    """.strip()
                )

        # 10. 创建维修步骤节点及关系（修改为 FaultReason -> MaintenanceAction）
        for step in data["维修步骤"]:
            action_id = f"WX{case_key}{step['步骤序号']:03d}"
            duration = step.get('操作时长', '')
            tools = step.get('所需工具', '')
            
            # 维修步骤关联到第一个故障原因
            if cause_ids:
                first_cause_id = cause_ids[0]
                cyphers.append(
                    f"""
                    MERGE (ma:MaintenanceAction {{action_id: '{action_id}'}})
                    SET ma.step_order = {step['步骤序号']},
                        ma.description = '{step['操作内容']}',
                        ma.estimated_time = '{duration}',
                        ma.tools = '{tools}',
                        ma.source_id = '{source_id}'
                    WITH ma
                    MERGE (fr:FaultReason {{cause_id: '{first_cause_id}'}})
                    MERGE (fr)-[:FIXED_BY]->(ma);
                    """.strip()
                )

        # 11. 创建注意事项节点及关系
        if notices:
            for i, n in enumerate(notices):
                notice_id = f"ZY{case_key}{i+1:03d}"
                cyphers.append(
                    f"""
                    MERGE (sn:SafetyNotice {{notice_id: '{notice_id}'}})
                    SET sn.level = '中', sn.description = '{n}',
                        sn.consequence = '可能造成设备损坏或人员伤害',
                        sn.source_id = '{source_id}'
                    WITH sn
                    MATCH (ma:MaintenanceAction)
                    WHERE ma.action_id STARTS WITH 'WX{case_key}'
                    MERGE (ma)-[:HAS_NOTICE]->(sn);
                    """.strip()
                )

        # 12. 将知识来源关系扩展到本案例所有核心实体
        cyphers.append(
            f"""
            MATCH (ks:KnowledgeSource {{source_id: '{source_id}'}})
            MATCH (ec:EquipmentCategory {{category_id: '{category_id}'}})
            MERGE (ec)-[:COMES_FROM]->(ks);
            """.strip()
        )
        cyphers.append(
            f"""
            MATCH (ks:KnowledgeSource {{source_id: '{source_id}'}})
            MATCH (e:Equipment {{equipment_id: '{equipment_id}'}})
            MERGE (e)-[:COMES_FROM]->(ks);
            """.strip()
        )
        cyphers.append(
            f"""
            MATCH (ks:KnowledgeSource {{source_id: '{source_id}'}})
            MATCH (c:Component)
            WHERE c.component_id STARTS WITH 'BJ{case_key}'
            MERGE (c)-[:COMES_FROM]->(ks);
            """.strip()
        )
        cyphers.append(
            f"""
            MATCH (ks:KnowledgeSource {{source_id: '{source_id}'}})
            MATCH (f:Fault {{fault_id: '{fault_id}'}})
            MERGE (f)-[:COMES_FROM]->(ks);
            """.strip()
        )
        cyphers.append(
            f"""
            MATCH (ks:KnowledgeSource {{source_id: '{source_id}'}})
            MATCH (fp:FaultPhenomenon)
            WHERE fp.phenomenon_id STARTS WITH 'XX{case_key}'
            MERGE (fp)-[:COMES_FROM]->(ks);
            """.strip()
        )
        cyphers.append(
            f"""
            MATCH (ks:KnowledgeSource {{source_id: '{source_id}'}})
            MATCH (fr:FaultReason)
            WHERE fr.cause_id STARTS WITH 'YY{case_key}'
            MERGE (fr)-[:COMES_FROM]->(ks);
            """.strip()
        )
        cyphers.append(
            f"""
            MATCH (ks:KnowledgeSource {{source_id: '{source_id}'}})
            MATCH (ma:MaintenanceAction)
            WHERE ma.action_id STARTS WITH 'WX{case_key}'
            MERGE (ma)-[:COMES_FROM]->(ks);
            """.strip()
        )
        cyphers.append(
            f"""
            MATCH (ks:KnowledgeSource {{source_id: '{source_id}'}})
            MATCH (sn:SafetyNotice)
            WHERE sn.notice_id STARTS WITH 'ZY{case_key}'
            MERGE (sn)-[:COMES_FROM]->(ks);
            """.strip()
        )

        return cyphers

    def create_relation_key_values(self, data: Dict, case_key: Optional[str] = None) -> List[Dict]:
        """
        创建关系键值对，用于图索引（适配kg_generator_v2.py的实体类型和关系定义）
        
        Args:
            data: 解析后的案例数据
            
        Returns:
            关系键值对列表
        """
        relations = []
        
        equipment = data["船舶装备"]
        system_name = data.get("所属系统", "未知")
        equipment_type = data.get("装备类型", "未知")
        parts = self._split(data["装备零部件"])
        fault_name = data.get("故障", "未知故障")
        fault_type = data.get("故障类型", "机械")
        phenomena = self._split(data["故障现象"])
        causes = self._split(data["故障原因"])
        notices = data.get("注意事项", [])

        if not system_name or str(system_name).strip() in {"未知", "无"}:
            system_name = self._infer_system_name(equipment, equipment_type, fault_type)

        # 同步 build_cypher 的唯一标识生成方式，避免同秒导入导致索引键冲突
        if case_key is None:
            import uuid
            case_key = uuid.uuid4().hex

        category_id = f"ZB{case_key}"
        equipment_id = f"SB{case_key}"
        fault_id = f"GZ{case_key}"
        source_id = f"KS{case_key}"
        
        # 1. 装备大类包含设备的关系
        relations.append({
            "source_entity": category_id,
            "target_entity": equipment_id,
            "relation_type": "CONTAINS",
            "index_keys": ["CONTAINS", "装备大类", "设备分类", system_name, f"{equipment_type}_设备", equipment],
            "value_content": f"系统 {system_name} 下装备大类 {equipment_type} 包含设备: {equipment}"
        })
        
        # 2. 设备包含部件的关系
        for i, p in enumerate(parts):
            component_id = f"BJ{case_key}{i+1:03d}"
            relations.append({
                "source_entity": equipment_id,
                "target_entity": component_id,
                "relation_type": "CONSISTS_OF",
                "index_keys": ["CONSISTS_OF", system_name, "设备组成", "部件配置", f"{equipment}_部件", p],
                "value_content": f"{system_name}中 {equipment} 包含部件: {p}"
            })
        
        # 3. 设备发生故障的关系
        relations.append({
            "source_entity": equipment_id,
            "target_entity": fault_id,
            "relation_type": "HAS_FAULT",
            "index_keys": ["HAS_FAULT", system_name, "设备故障", f"{equipment}_故障", "维修问题"],
            "value_content": f"{system_name}中 {equipment} 存在故障: {fault_name}"
        })
        
        # 4. 故障表现为现象的关系
        for i, fp in enumerate(phenomena):
            phenomenon_id = f"XX{case_key}{i+1:03d}"
            relations.append({
                "source_entity": fault_id,
                "target_entity": phenomenon_id,
                "relation_type": "PRESENTS_AS",
                "index_keys": ["PRESENTS_AS", "故障现象", f"{fault_name}_现象", "故障诊断"],
                "value_content": f"故障 {fault_name} 表现为: {fp}"
            })
        
        # 5. 故障由原因导致的关系
        cause_ids = []
        for i, cause in enumerate(causes):
            cause_id = f"YY{case_key}{i+1:03d}"
            cause_ids.append(cause_id)
            relations.append({
                "source_entity": fault_id,
                "target_entity": cause_id,
                "relation_type": "CAUSED_BY",
                "index_keys": ["CAUSED_BY", "故障原因", "故障分析", f"{fault_name}_原因", "故障诊断"],
                "value_content": f"故障 {fault_name} 由 {cause} 导致"
            })
            
            # 6. 故障原因关联部件的关系
            if parts:
                component_id = f"BJ{case_key}001"
                relations.append({
                    "source_entity": cause_id,
                    "target_entity": component_id,
                    "relation_type": "RELATES_TO",
                    "index_keys": ["RELATES_TO", "故障部件", f"{cause}_部件", "部件关联"],
                    "value_content": f"故障原因 {cause} 关联部件 {parts[0]}"
                })
        
        # 7. 故障原因通过维修步骤修复的关系（修改为 FaultReason -> MaintenanceAction）
        for step in data["维修步骤"]:
            action_id = f"WX{case_key}{step['步骤序号']:03d}"
            
            # 维修步骤关联到第一个故障原因
            if cause_ids:
                first_cause_id = cause_ids[0]
                relations.append({
                    "source_entity": first_cause_id,
                    "target_entity": action_id,
                    "relation_type": "FIXED_BY",
                    "index_keys": ["FIXED_BY", "维修步骤", "故障维修", f"{causes[0]}_维修", "维修方案"],
                    "value_content": f"故障原因 {causes[0]} 需要维修步骤: {step['操作内容']}"
                })
            
            # 8. 维修步骤有注意事项的关系
            if notices:
                for i, n in enumerate(notices):
                    notice_id = f"ZY{case_key}{i+1:03d}"
                    relations.append({
                        "source_entity": action_id,
                        "target_entity": notice_id,
                        "relation_type": "HAS_NOTICE",
                        "index_keys": ["HAS_NOTICE", "注意事项", "安全注意", "维修安全", "操作规范"],
                        "value_content": f"维修步骤 {step['操作内容']} 需要注意: {n}"
                    })
        
        # 9. 设备来自知识来源的关系
        relations.append({
            "source_entity": equipment_id,
            "target_entity": source_id,
            "relation_type": "COMES_FROM",
            "index_keys": ["COMES_FROM", "知识来源", "数据溯源", "知识管理"],
            "value_content": f"设备 {equipment} 的知识来源: {data.get('知识来源', {}).get('书名', '未知')}"
        })
        
        return relations

    def build_cypher_with_linking(self, data: Dict, enable_linking: bool = True) -> List[str]:
        """
        构建Cypher语句（支持与已有节点关联）
        
        Args:
            data: 解析后的案例数据
            enable_linking: 是否启用已有节点关联功能
            
        Returns:
            Cypher语句列表
        """
        cyphers = []
        
        equipment = data["船舶装备"]
        system_name = data.get("所属系统", "未知")
        equipment_type = data.get("装备类型", "未知")
        equipment_model = data.get("装备型号", "未知")
        
        parts_data = data["装备零部件"]
        fault_name = data.get("故障", "未知故障")
        fault_type = data.get("故障类型", "机械")
        severity = data.get("故障严重等级", "中")
        frequency = data.get("故障发生频率", "中频")
        phenomena_data = data["故障现象"]
        causes_data = data["故障原因"]
        
        if isinstance(parts_data, dict):
            parts = list(parts_data.keys())
        else:
            parts = self._split(parts_data)
        
        if isinstance(phenomena_data, dict):
            phenomena = list(phenomena_data.keys())
        else:
            phenomena = self._split(phenomena_data)
        
        if isinstance(causes_data, dict):
            causes = list(causes_data.keys())
        else:
            causes = self._split(causes_data)

        if not system_name or str(system_name).strip() in {"未知", "无"}:
            system_name = self._infer_system_name(equipment, equipment_type, fault_type)

        # ---- 关键：为“泛化故障/现象”补充主体，避免跨设备/部件误合并 ----
        generic_phenomenon_keywords = [
            "过热", "温度过高", "高温", "温度升高",
            "异响", "噪音", "振动",
            "泄漏", "渗漏",
            "压力低", "压力不足", "压力过低", "压力过高",
            "流量低", "流量不足",
            "电流过大", "电流过高", "电压过低", "电压过高",
            "报警", "报警码",
        ]
        generic_fault_keywords = [
            "启动失败", "无法启动", "起动失败",
            "过热", "温度过高",
            "停机", "自动停机", "跳闸",
            "功率不足", "无输出", "输出异常",
        ]
        subject_equipment = equipment
        subject_parts = parts if isinstance(parts, list) else []

        fault_name = self._contextualize_generic_term(
            fault_name,
            subject=subject_equipment,
            extra_subjects=subject_parts,
            generic_keywords=generic_fault_keywords,
        )
        phenomena = [
            self._contextualize_generic_term(
                fp,
                subject=subject_equipment,
                extra_subjects=subject_parts,
                generic_keywords=generic_phenomenon_keywords,
            )
            for fp in phenomena
        ]
        
        notices = data.get("注意事项", [])
        knowledge_source = data.get("知识来源", {})
        
        timestamp = int(time.time())
        source_id = f"KS{timestamp}"
        book_name = knowledge_source.get("书名", "未知")
        chapter = knowledge_source.get("章节", "")
        section = knowledge_source.get("小节", "")
        
        # 1. 创建知识来源节点
        cyphers.append(
            f"""
            MERGE (ks:KnowledgeSource {{source_id: '{source_id}'}})
            SET ks.type = '手册', ks.title = '{book_name}',
                ks.chapter = '{chapter}', ks.section = '{section}',
                ks.reliability = '中';
            """.strip()
        )
        
        # 2. 查找或创建装备大类节点
        if enable_linking:
            category_result = self.find_or_create_entity("EquipmentCategory", equipment_type)
            category_id = category_result["id"]
            is_new_category = category_result["is_new"]
        else:
            category_id = f"ZB{timestamp}"
            is_new_category = True
        
        if is_new_category:
            cyphers.append(
                f"""
                MERGE (ec:EquipmentCategory {{category_id: '{category_id}'}})
                SET ec.name = '{equipment_type}', ec.description = '{equipment_type}相关设备',
                    ec.system_name = '{system_name}',
                    ec.source_id = '{source_id}';
                """.strip()
            )
        
        # 3. 查找或创建设备节点
        if enable_linking:
            equipment_result = self.find_or_create_entity(
                "Equipment",
                equipment,
                {
                    "type": equipment_type,
                    "model": equipment_model,
                },
            )
            equipment_id = equipment_result["id"]
            is_new_equipment = equipment_result["is_new"]
        else:
            equipment_id = f"SB{timestamp}"
            is_new_equipment = True
        
        if is_new_equipment:
            cyphers.append(
                f"""
                MERGE (e:Equipment {{equipment_id: '{equipment_id}'}})
                SET e.name = '{equipment}', e.system_name = '{system_name}',
                    e.type = '{equipment_type}', e.model = '{equipment_model}',
                    e.source_id = '{source_id}';
                """.strip()
            )
        else:
            # 若数据库该设备已有但型号/类型为“未知”，用新数据补全（避免同名误合并，同时提高幂等导入质量）
            existing_attrs = equipment_result.get("attrs") or {}
            existing_model = existing_attrs.get("model")
            existing_type = existing_attrs.get("type")
            should_update_model = (not self._is_unknown_value(equipment_model)) and self._is_unknown_value(existing_model)
            should_update_type = (not self._is_unknown_value(equipment_type)) and self._is_unknown_value(existing_type)
            if should_update_model or should_update_type:
                cyphers.append(
                    f"""
                    MATCH (e:Equipment {{equipment_id: '{equipment_id}'}})
                    SET e.type = CASE
                            WHEN e.type IS NULL OR e.type = '' OR toLower(e.type) = '未知' THEN '{equipment_type}'
                            ELSE e.type
                        END,
                        e.system_name = CASE
                            WHEN e.system_name IS NULL OR e.system_name = '' OR toLower(e.system_name) = '未知' THEN '{system_name}'
                            ELSE e.system_name
                        END,
                        e.model = CASE
                            WHEN e.model IS NULL OR e.model = '' OR toLower(e.model) = '未知' THEN '{equipment_model}'
                            ELSE e.model
                        END,
                        e.source_id = COALESCE(e.source_id, '{source_id}');
                    """.strip()
                )
        
        # 4. 创建装备大类与设备的关系
        cyphers.append(
            f"""
            MATCH (ec:EquipmentCategory {{category_id: '{category_id}'}})
            MATCH (e:Equipment {{equipment_id: '{equipment_id}'}})
            MERGE (ec)-[:CONTAINS]->(e);
            """.strip()
        )
        
        # 5. 查找或创建部件节点
        component_ids = []
        for i, p in enumerate(parts):
            if enable_linking:
                component_result = self.find_or_create_entity("Component", p)
                component_id = component_result["id"]
                is_new_component = component_result["is_new"]
            else:
                component_id = f"BJ{timestamp}{i+1:03d}"
                is_new_component = True
            
            component_ids.append(component_id)
            
            if is_new_component:
                cyphers.append(
                    f"""
                    MERGE (c:Component {{component_id: '{component_id}'}})
                    SET c.name = '{p}', c.spec = '未知', c.type = '部件',
                        c.source_id = '{source_id}';
                    """.strip()
                )
            
            cyphers.append(
                f"""
                MATCH (e:Equipment {{equipment_id: '{equipment_id}'}})
                MATCH (c:Component {{component_id: '{component_id}'}})
                MERGE (e)-[:CONSISTS_OF]->(c);
                """.strip()
            )
        
        # 6. 查找或创建故障节点
        if enable_linking:
            fault_result = self.find_or_create_entity("Fault", fault_name)
            fault_id = fault_result["id"]
            is_new_fault = fault_result["is_new"]
        else:
            fault_id = f"GZ{timestamp}"
            is_new_fault = True
        
        if is_new_fault:
            cyphers.append(
                f"""
                MERGE (f:Fault {{fault_id: '{fault_id}'}})
                SET f.name = '{fault_name}', f.fault_type = '{fault_type}',
                    f.severity = '{severity}', f.occurrence_frequency = '{frequency}',
                    f.description = '{fault_name}', f.source_id = '{source_id}';
                """.strip()
            )
        
        cyphers.append(
            f"""
            MATCH (e:Equipment {{equipment_id: '{equipment_id}'}})
            MATCH (f:Fault {{fault_id: '{fault_id}'}})
            MERGE (e)-[:HAS_FAULT]->(f);
            """.strip()
        )
        
        # 7. 查找或创建故障现象节点
        for i, fp in enumerate(phenomena):
            if enable_linking:
                phenomenon_result = self.find_or_create_entity("FaultPhenomenon", fp)
                phenomenon_id = phenomenon_result["id"]
                is_new_phenomenon = phenomenon_result["is_new"]
            else:
                phenomenon_id = f"XX{timestamp}{i+1:03d}"
                is_new_phenomenon = True
            
            if is_new_phenomenon:
                cyphers.append(
                    f"""
                    MERGE (fp:FaultPhenomenon {{phenomenon_id: '{phenomenon_id}'}})
                    SET fp.description = '{fp}', fp.source_id = '{source_id}';
                    """.strip()
                )
            
            cyphers.append(
                f"""
                MATCH (f:Fault {{fault_id: '{fault_id}'}})
                MATCH (fp:FaultPhenomenon {{phenomenon_id: '{phenomenon_id}'}})
                MERGE (f)-[:PRESENTS_AS]->(fp);
                """.strip()
            )
        
        # 8. 查找或创建故障原因节点
        cause_ids = []
        for i, cause in enumerate(causes):
            if enable_linking:
                cause_result = self.find_or_create_entity("FaultReason", cause)
                cause_id = cause_result["id"]
                is_new_cause = cause_result["is_new"]
            else:
                cause_id = f"YY{timestamp}{i+1:03d}"
                is_new_cause = True
            
            cause_ids.append(cause_id)
            
            if is_new_cause:
                cyphers.append(
                    f"""
                    MERGE (fr:FaultReason {{cause_id: '{cause_id}'}})
                    SET fr.cause_name = '{cause}', fr.description = '{cause}',
                        fr.category = '老化', fr.level = 0,
                        fr.source_id = '{source_id}';
                    """.strip()
                )
            
            cyphers.append(
                f"""
                MATCH (f:Fault {{fault_id: '{fault_id}'}})
                MATCH (fr:FaultReason {{cause_id: '{cause_id}'}})
                MERGE (f)-[:CAUSED_BY]->(fr);
                """.strip()
            )
            
            # 9. 创建故障原因与部件的关系
            if component_ids:
                first_component_id = component_ids[0]
                cyphers.append(
                    f"""
                    MATCH (fr:FaultReason {{cause_id: '{cause_id}'}})
                    MATCH (c:Component {{component_id: '{first_component_id}'}})
                    MERGE (fr)-[:RELATES_TO]->(c);
                    """.strip()
                )
        
        # 10. 查找或创建维修步骤节点
        for step in data["维修步骤"]:
            if enable_linking:
                action_result = self.find_or_create_entity("MaintenanceAction", step['操作内容'])
                action_id = action_result["id"]
                is_new_action = action_result["is_new"]
            else:
                action_id = f"WX{timestamp}{step['步骤序号']:03d}"
                is_new_action = True
            
            duration = step.get('操作时长', '')
            tools = step.get('所需工具', '')
            
            if is_new_action:
                cyphers.append(
                    f"""
                    MERGE (ma:MaintenanceAction {{action_id: '{action_id}'}})
                    SET ma.step_order = {step['步骤序号']},
                        ma.description = '{step['操作内容']}',
                        ma.estimated_time = '{duration}',
                        ma.tools = '{tools}',
                        ma.source_id = '{source_id}';
                    """.strip()
                )
            
            # 维修步骤关联到第一个故障原因
            if cause_ids:
                first_cause_id = cause_ids[0]
                cyphers.append(
                    f"""
                    MATCH (fr:FaultReason {{cause_id: '{first_cause_id}'}})
                    WITH fr
                    MATCH (ma:MaintenanceAction {{action_id: '{action_id}'}})
                    MERGE (fr)-[:FIXED_BY]->(ma);
                    """.strip()
                )
        
        # 11. 查找或创建注意事项节点
        if notices:
            for i, n in enumerate(notices):
                if enable_linking:
                    notice_result = self.find_or_create_entity("SafetyNotice", n)
                    notice_id = notice_result["id"]
                    is_new_notice = notice_result["is_new"]
                else:
                    notice_id = f"ZY{timestamp}{i+1:03d}"
                    is_new_notice = True
                
                if is_new_notice:
                    cyphers.append(
                        f"""
                        MERGE (sn:SafetyNotice {{notice_id: '{notice_id}'}})
                        SET sn.level = '中', sn.description = '{n}',
                            sn.consequence = '可能造成设备损坏或人员伤害',
                            sn.source_id = '{source_id}';
                        """.strip()
                    )
                
                cyphers.append(
                    f"""
                    MATCH (ma:MaintenanceAction)
                    WHERE ma.action_id STARTS WITH 'WX{timestamp}'
                    WITH ma
                    MATCH (sn:SafetyNotice {{notice_id: '{notice_id}'}})
                    MERGE (ma)-[:HAS_NOTICE]->(sn);
                    """.strip()
                )
        
        # 12. 将知识来源关系扩展到本案例所有核心实体
        cyphers.append(
            f"""
            MATCH (ks:KnowledgeSource {{source_id: '{source_id}'}})
            MATCH (ec:EquipmentCategory {{category_id: '{category_id}'}})
            MERGE (ec)-[:COMES_FROM]->(ks);
            """.strip()
        )
        cyphers.append(
            f"""
            MATCH (ks:KnowledgeSource {{source_id: '{source_id}'}})
            MATCH (e:Equipment {{equipment_id: '{equipment_id}'}})
            MERGE (e)-[:COMES_FROM]->(ks);
            """.strip()
        )
        cyphers.append(
            f"""
            MATCH (ks:KnowledgeSource {{source_id: '{source_id}'}})
            MATCH (c:Component)
            WHERE c.component_id STARTS WITH 'BJ{timestamp}'
            MERGE (c)-[:COMES_FROM]->(ks);
            """.strip()
        )
        cyphers.append(
            f"""
            MATCH (ks:KnowledgeSource {{source_id: '{source_id}'}})
            MATCH (f:Fault {{fault_id: '{fault_id}'}})
            MERGE (f)-[:COMES_FROM]->(ks);
            """.strip()
        )
        cyphers.append(
            f"""
            MATCH (ks:KnowledgeSource {{source_id: '{source_id}'}})
            MATCH (fp:FaultPhenomenon)
            WHERE fp.phenomenon_id STARTS WITH 'XX{timestamp}'
            MERGE (fp)-[:COMES_FROM]->(ks);
            """.strip()
        )
        cyphers.append(
            f"""
            MATCH (ks:KnowledgeSource {{source_id: '{source_id}'}})
            MATCH (fr:FaultReason)
            WHERE fr.cause_id STARTS WITH 'YY{timestamp}'
            MERGE (fr)-[:COMES_FROM]->(ks);
            """.strip()
        )
        cyphers.append(
            f"""
            MATCH (ks:KnowledgeSource {{source_id: '{source_id}'}})
            MATCH (ma:MaintenanceAction)
            WHERE ma.action_id STARTS WITH 'WX{timestamp}'
            MERGE (ma)-[:COMES_FROM]->(ks);
            """.strip()
        )
        cyphers.append(
            f"""
            MATCH (ks:KnowledgeSource {{source_id: '{source_id}'}})
            MATCH (sn:SafetyNotice)
            WHERE sn.notice_id STARTS WITH 'ZY{timestamp}'
            MERGE (sn)-[:COMES_FROM]->(ks);
            """.strip()
        )
        
        return cyphers

    # =========================
    # 3️⃣ 执行 Cypher（确认后）
    # =========================
    def execute(self, cyphers: List[str], dry_run: bool = False) -> Dict[str, Any]:
        """
        执行Cypher语句
        
        Args:
            cyphers: Cypher语句列表
            dry_run: 是否只预览不执行
            
        Returns:
            执行结果字典，包含成功/失败信息
        """
        result = {
            "success": False,
            "executed_count": 0,
            "failed_count": 0,
            "errors": [],
            "inserted_nodes": [],
            "inserted_relations": [],
            "cyphers": cyphers
        }
        
        if dry_run:
            result["success"] = True
            result["message"] = "预览模式，未实际执行"
            logger.info("预览模式: 生成了 {} 条 Cypher 语句".format(len(cyphers)))
            return result
        
        try:
            with self.driver.session() as session:
                logger.info("开始执行 Cypher 语句，共 {} 条".format(len(cyphers)))
                for i, c in enumerate(cyphers, 1):
                    try:
                        logger.debug(f"执行语句 ({i}/{len(cyphers)}): {c[:100]}...")
                        session.run(c)
                        result["executed_count"] += 1
                        
                        if "MERGE" in c or "CREATE" in c:
                            if "EquipmentCategory" in c:
                                result["inserted_nodes"].append("EquipmentCategory")
                            elif "Equipment" in c and "EquipmentCategory" not in c:
                                result["inserted_nodes"].append("Equipment")
                            elif "Component" in c:
                                result["inserted_nodes"].append("Component")
                            elif "Fault" in c and "FaultPhenomenon" not in c and "FaultReason" not in c:
                                result["inserted_nodes"].append("Fault")
                            elif "FaultPhenomenon" in c:
                                result["inserted_nodes"].append("FaultPhenomenon")
                            elif "FaultReason" in c:
                                result["inserted_nodes"].append("FaultReason")
                            elif "MaintenanceAction" in c:
                                result["inserted_nodes"].append("MaintenanceAction")
                            elif "SafetyNotice" in c:
                                result["inserted_nodes"].append("SafetyNotice")
                            elif "KnowledgeSource" in c:
                                result["inserted_nodes"].append("KnowledgeSource")
                        
                        if "-[:" in c:
                            relation_type = c.split("-[:")[1].split("]")[0]
                            result["inserted_relations"].append(relation_type)
                            
                    except Exception as e:
                        result["failed_count"] += 1
                        error_msg = f"执行失败 ({i}/{len(cyphers)}): {str(e)}"
                        result["errors"].append(error_msg)
                        logger.error(error_msg)
                
                # 去重并统计
                result["inserted_nodes"] = list(set(result["inserted_nodes"]))
                result["inserted_relations"] = list(set(result["inserted_relations"]))
                
                result["success"] = result["failed_count"] == 0
                result["message"] = f"执行完成: {result['executed_count']} 成功, {result['failed_count']} 失败"
                result["node_count"] = len(result["inserted_nodes"])
                result["relation_count"] = len(result["inserted_relations"])
                
                logger.info(f"执行结果: {result['message']}")
                logger.info(f"插入的节点类型: {result['inserted_nodes']}")
                logger.info(f"插入的关系类型: {result['inserted_relations']}")
                
        except Exception as e:
            result["success"] = False
            result["message"] = f"数据库连接失败: {str(e)}"
            logger.exception("执行Cypher失败")
        
        return result

    # =========================
    # 4️⃣ 完整流程
    # =========================
    def insert_case(self, case: str, dry_run: bool = False, use_deepseek: bool = None, parsed_data: dict = None, image_paths: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        完整的案例插入流程
        
        Args:
            case: 案例文本
            dry_run: 是否只预览不执行
            use_deepseek: 是否使用Deepseek提取数据
            parsed_data: 用户提供的解析数据（如果有）
            
        Returns:
            执行结果字典
        """
        result = {
            "success": False,
            "case": case,
            "parsed_data": None,
            "cyphers": [],
            "execution": None,
            "relation_key_values": [],
            "case_key": None,
            "neo4j_node_ids": [],
            "message": "",
            "statistics": {}
        }
        
        try:
            # 为本次导入生成唯一 case_key，保证 build_cypher 与 create_relation_key_values 使用同一套 id
            import uuid
            case_key = uuid.uuid4().hex
            result["case_key"] = case_key

            # 确定是否使用Deepseek
            current_use_deepseek = use_deepseek if use_deepseek is not None else self.use_deepseek
            
            # 1. LLM解析（如果用户没有提供解析数据）
            if parsed_data is None:
                logger.info(f"开始解析案例... {'使用Deepseek' if current_use_deepseek else '使用默认模型'}")
                parsed_data = self.analyze_case(case, use_deepseek=current_use_deepseek, image_paths=image_paths)
                logger.info(f"解析成功，提取了 {len(parsed_data.get('维修步骤', []))} 个维修步骤")
            else:
                logger.info("使用用户提供的解析数据")
            
            result["parsed_data"] = parsed_data

            # 计算本次新增/更新的 neo4j_node_ids，用于向量索引增量刷新
            try:
                parts = self._split(parsed_data.get("装备零部件", "无"))
                phenomena = self._split(parsed_data.get("故障现象", "无"))
                causes = self._split(parsed_data.get("故障原因", "无"))

                notices_raw = parsed_data.get("注意事项", [])
                if isinstance(notices_raw, str):
                    notices = self._split(notices_raw)
                elif isinstance(notices_raw, list):
                    notices = [str(x).strip() for x in notices_raw if str(x).strip()]
                else:
                    notices = []

                steps = parsed_data.get("维修步骤", []) if isinstance(parsed_data.get("维修步骤", []), list) else []

                category_id = f"ZB{case_key}"
                equipment_id = f"SB{case_key}"
                fault_id = f"GZ{case_key}"
                source_id = f"KS{case_key}"

                component_ids = [f"BJ{case_key}{i+1:03d}" for i in range(len(parts))]
                phenomenon_ids = [f"XX{case_key}{i+1:03d}" for i in range(len(phenomena))]
                cause_ids = [f"YY{case_key}{i+1:03d}" for i in range(len(causes))]

                action_ids = []
                for idx, step in enumerate(steps):
                    step_no = step.get("步骤序号", idx + 1) if isinstance(step, dict) else (idx + 1)
                    try:
                        step_no_int = int(step_no)
                    except Exception:
                        step_no_int = idx + 1
                    action_ids.append(f"WX{case_key}{step_no_int:03d}")

                notice_ids = [f"ZY{case_key}{i+1:03d}" for i in range(len(notices))]

                result["neo4j_node_ids"] = (
                    [category_id, equipment_id, fault_id, source_id]
                    + component_ids
                    + phenomenon_ids
                    + cause_ids
                    + action_ids
                    + notice_ids
                )
            except Exception as e:
                logger.warning(f"计算 neo4j_node_ids 失败，增量更新可能不完整: {e}")
            
            # 2. 构建Cypher
            logger.info("构建Cypher语句...")
            cyphers = self.build_cypher(parsed_data, case_key=case_key)
            result["cyphers"] = cyphers
            logger.info(f"构建完成，生成了 {len(cyphers)} 条 Cypher 语句")
            
            # 3. 创建关系键值对
            logger.info("创建关系键值对...")
            relation_key_values = self.create_relation_key_values(parsed_data, case_key=case_key)
            result["relation_key_values"] = relation_key_values
            logger.info(f"创建完成，生成了 {len(relation_key_values)} 个关系键值对")
            
            # 4. 执行Cypher
            if not dry_run:
                logger.info("执行Cypher语句...")
                execution_result = self.execute(cyphers, dry_run=False)
                result["execution"] = execution_result
                result["success"] = execution_result["success"]
                result["message"] = execution_result["message"]
                
                # 统计信息
                result["statistics"] = {
                    "nodes_inserted": execution_result.get("node_count", 0),
                    "relations_inserted": execution_result.get("relation_count", 0),
                    "cypher_statements": len(cyphers),
                    "relation_key_values": len(relation_key_values),
                    "maintenance_steps": len(parsed_data.get("维修步骤", []))
                }
            else:
                result["success"] = True
                result["message"] = "预览模式，未实际执行"
                
                # 统计信息
                result["statistics"] = {
                    "cypher_statements": len(cyphers),
                    "relation_key_values": len(relation_key_values),
                    "maintenance_steps": len(parsed_data.get("维修步骤", []))
                }
            
            logger.info(f"案例插入{'成功' if result['success'] else '失败'}")
            if result['success']:
                logger.info(f"插入统计: {result['statistics']}")
            
        except Exception as e:
            result["success"] = False
            result["message"] = f"插入失败: {str(e)}"
            logger.exception("案例插入失败")
        
        return result
        


    def insertbycase(self, case: str, parsed_data: dict = None) -> Dict[str, Any]:
        """
        从已有的 JSON 字符串直接插入案例流程
        
        Args:
            case: JSON 字符串（必须为标准 JSON 格式）
            parsed_data: 该参数在本方法中无效，留空即可
            
        Returns:
            执行结果字典
        """
        result = {
            "success": False,
            "case": case,
            "parsed_data": None,
            "cyphers": [],
            "execution": None,
            "relation_key_values": [],
            "case_key": None,
            "neo4j_node_ids": [],
            "message": "",
            "statistics": {}
        }

        try:
            # 为本次导入生成唯一 case_key，保证 build_cypher 与 create_relation_key_values 使用同一套 id
            import uuid
            case_key = uuid.uuid4().hex
            result["case_key"] = case_key

            # 1. 直接解析 JSON
            parsed_data = json.loads(case)
            result["parsed_data"] = parsed_data

            # 计算本次新增/更新的 neo4j_node_ids，用于向量索引增量刷新
            try:
                parts = self._split(parsed_data.get("装备零部件", "无"))
                phenomena = self._split(parsed_data.get("故障现象", "无"))
                causes = self._split(parsed_data.get("故障原因", "无"))

                notices_raw = parsed_data.get("注意事项", [])
                if isinstance(notices_raw, str):
                    notices = self._split(notices_raw)
                elif isinstance(notices_raw, list):
                    notices = [str(x).strip() for x in notices_raw if str(x).strip()]
                else:
                    notices = []

                steps = parsed_data.get("维修步骤", []) if isinstance(parsed_data.get("维修步骤", []), list) else []

                category_id = f"ZB{case_key}"
                equipment_id = f"SB{case_key}"
                fault_id = f"GZ{case_key}"
                source_id = f"KS{case_key}"

                component_ids = [f"BJ{case_key}{i+1:03d}" for i in range(len(parts))]
                phenomenon_ids = [f"XX{case_key}{i+1:03d}" for i in range(len(phenomena))]
                cause_ids = [f"YY{case_key}{i+1:03d}" for i in range(len(causes))]

                action_ids = []
                for idx, step in enumerate(steps):
                    step_no = step.get("步骤序号", idx + 1) if isinstance(step, dict) else (idx + 1)
                    try:
                        step_no_int = int(step_no)
                    except Exception:
                        step_no_int = idx + 1
                    action_ids.append(f"WX{case_key}{step_no_int:03d}")

                notice_ids = [f"ZY{case_key}{i+1:03d}" for i in range(len(notices))]

                result["neo4j_node_ids"] = (
                    [category_id, equipment_id, fault_id, source_id]
                    + component_ids
                    + phenomenon_ids
                    + cause_ids
                    + action_ids
                    + notice_ids
                )
            except Exception as e:
                logger.warning(f"计算 neo4j_node_ids 失败，增量更新可能不完整: {e}")
        except json.JSONDecodeError as e:
            result["message"] = f"提供的 case 不是合法 JSON: {e}"
            logger.error(result["message"])
            return result

        # 2. 构建 Cypher
        logger.info("构建 Cypher 语句...")
        cyphers = self.build_cypher(parsed_data, case_key=case_key)
        result["cyphers"] = cyphers
        logger.info(f"构建完成，生成了 {len(cyphers)} 条 Cypher 语句")

        # 3. 创建关系键值对
        logger.info("创建关系键值对...")
        relation_key_values = self.create_relation_key_values(parsed_data, case_key=case_key)
        result["relation_key_values"] = relation_key_values
        logger.info(f"创建完成，生成了 {len(relation_key_values)} 个关系键值对")

        try:
            # 4. 执行 Cypher
            
            logger.info("执行 Cypher 语句...")
            execution_result = self.execute(cyphers, dry_run=False)
            result["execution"] = execution_result
            result["success"] = execution_result["success"]
            result["message"] = execution_result["message"]

                # 统计信息
            result["statistics"] = {
                    "nodes_inserted": execution_result.get("node_count", 0),
                    "relations_inserted": execution_result.get("relation_count", 0),
                    "cypher_statements": len(cyphers),
                    "relation_key_values": len(relation_key_values),
                    "maintenance_steps": len(parsed_data.get("维修步骤", []))
                }
            

            logger.info(f"案例插入{'成功' if result['success'] else '失败'}")
            if result["success"]:
                logger.info(f"插入统计: {result['statistics']}")

        except Exception as e:
            result["success"] = False
            result["message"] = f"插入失败: {str(e)}"
            logger.exception("案例插入失败")

        return result
        

    # =========================
    # 工具函数
    # =========================
    @staticmethod
    def _split(text: Union[str, dict, list]) -> List[str]:
        if text == "无":
            return []
        if isinstance(text, list):
            return [str(i).strip() for i in text if str(i).strip()]
        if isinstance(text, dict):
            return list(text.keys())
        return [i.strip() for i in text.split(",") if i.strip()]
