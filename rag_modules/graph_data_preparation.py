"""

图数据库 数据准备模块
"""
import logging
from dataclasses import dataclass
from typing import List, Dict, Any

from langchain_core.documents import Document
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

@dataclass
class GraphNode:
    """ 图节点 数据结构"""
    node_id: str
    labels:List[str]
    name:str
    properties:Dict[str,Any]


@dataclass
class GraphRelation:
    """图关系数据结构"""
    start_node_id: str
    end_node_id: str
    relation_type: str
    properties: Dict[str,Any]


class GraphDataPreparationModule:
    """图数据库数据准备模块 - 从Neo4j读取数据并转换为文档"""

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        """
            初始化图数据库连接

            Args:
                uri: Neo4j连接URI
                user: 用户名
                password: 密码
                database: 数据库名称
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = None
        self.documents: List[Document] = []
        self.chunks: List[Document] = []
        self.equipment_categories: List[GraphNode] = []
        self.equipments: List[GraphNode] = []
        self.components: List[GraphNode] = []
        self.faults: List[GraphNode] = []
        self.fault_phenomenons: List[GraphNode] = []
        self.fault_reasons: List[GraphNode] = []
        self.maintenance_actions: List[GraphNode] = []
        self.safety_notices: List[GraphNode] = []
        self.knowledge_sources: List[GraphNode] = []

        self._connect()

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


    def load_graph_data(self) -> Dict[str, Any]:
        with self.driver.session() as session:
            equipment_categories_query = """
            MATCH (ec:EquipmentCategory)
            RETURN ec.category_id as nodeId, labels(ec) as labels, ec.name as name,
                   properties(ec) as properties
            ORDER BY ec.category_id
            """
            result = session.run(equipment_categories_query)
            self.equipment_categories = []
            for record in result:
                node = GraphNode(
                    node_id=record["nodeId"],
                    labels=record["labels"],
                    name=record["name"],
                    properties=record["properties"]
                )
                self.equipment_categories.append(node)
            logger.info(f"加载了{len(self.equipment_categories)}个装备大类节点")

            equipments_query = """
            MATCH (e:Equipment)
            RETURN e.equipment_id as nodeId, labels(e) as labels, e.name as name,
                   properties(e) as properties
            ORDER BY e.equipment_id
            """
            result = session.run(equipments_query)
            self.equipments = []
            for record in result:
                properties = dict(record["properties"])
                node = GraphNode(
                    node_id=record["nodeId"],
                    labels=record["labels"],
                    name=record["name"],
                    properties=properties
                )
                self.equipments.append(node)
            logger.info(f"加载了{len(self.equipments)}个设备节点")

            components_query = """
            MATCH (c:Component)
            RETURN c.component_id as nodeId, labels(c) as labels, c.name as name,
                   properties(c) as properties
            ORDER BY c.component_id
            """
            result = session.run(components_query)
            self.components = []
            for record in result:
                node = GraphNode(
                    node_id=record["nodeId"],
                    labels=record["labels"],
                    name=record["name"],
                    properties=record["properties"]
                )
                self.components.append(node)
            logger.info(f"加载了{len(self.components)}个部件节点")

            faults_query = """
            MATCH (f:Fault)
            RETURN f.fault_id as nodeId, labels(f) as labels, f.name as name,
                   properties(f) as properties
            ORDER BY f.fault_id
            """
            result = session.run(faults_query)
            self.faults = []
            for record in result:
                node = GraphNode(
                    node_id=record["nodeId"],
                    labels=record["labels"],
                    name=record["name"],
                    properties=record["properties"]
                )
                self.faults.append(node)
            logger.info(f"加载了{len(self.faults)}个故障节点")

            fault_phenomenons_query = """
            MATCH (fp:FaultPhenomenon)
            RETURN fp.phenomenon_id as nodeId, labels(fp) as labels, fp.description as name,
                   properties(fp) as properties
            ORDER BY fp.phenomenon_id
            """
            result = session.run(fault_phenomenons_query)
            self.fault_phenomenons = []
            for record in result:
                node = GraphNode(
                    node_id=record["nodeId"],
                    labels=record["labels"],
                    name=record["name"],
                    properties=record["properties"]
                )
                self.fault_phenomenons.append(node)
            logger.info(f"加载了{len(self.fault_phenomenons)}个故障现象节点")

            fault_reasons_query = """
            MATCH (fr:FaultReason)
            RETURN fr.cause_id as nodeId, labels(fr) as labels, fr.cause_name as name,
                   properties(fr) as properties
            ORDER BY fr.cause_id
            """
            result = session.run(fault_reasons_query)
            self.fault_reasons = []
            for record in result:
                node = GraphNode(
                    node_id=record["nodeId"],
                    labels=record["labels"],
                    name=record["name"],
                    properties=record["properties"]
                )
                self.fault_reasons.append(node)
            logger.info(f"加载了{len(self.fault_reasons)}个故障原因节点")

            maintenance_actions_query = """
            MATCH (ma:MaintenanceAction)
            RETURN ma.action_id as nodeId, labels(ma) as labels, ma.description as name,
                   properties(ma) as properties
            ORDER BY ma.action_id
            """
            result = session.run(maintenance_actions_query)
            self.maintenance_actions = []
            for record in result:
                node = GraphNode(
                    node_id=record["nodeId"],
                    labels=record["labels"],
                    name=record["name"],
                    properties=record["properties"]
                )
                self.maintenance_actions.append(node)
            logger.info(f"加载了{len(self.maintenance_actions)}个维修步骤节点")

            safety_notices_query = """
            MATCH (sn:SafetyNotice)
            RETURN sn.notice_id as nodeId, labels(sn) as labels, sn.description as name,
                   properties(sn) as properties
            ORDER BY sn.notice_id
            """
            result = session.run(safety_notices_query)
            self.safety_notices = []
            for record in result:
                node = GraphNode(
                    node_id=record["nodeId"],
                    labels=record["labels"],
                    name=record["name"],
                    properties=record["properties"]
                )
                self.safety_notices.append(node)
            logger.info(f"加载了{len(self.safety_notices)}个注意事项节点")

            knowledge_sources_query = """
            MATCH (ks:KnowledgeSource)
            RETURN ks.source_id as nodeId, labels(ks) as labels, ks.title as name,
                   properties(ks) as properties
            ORDER BY ks.source_id
            """
            result = session.run(knowledge_sources_query)
            self.knowledge_sources = []
            for record in result:
                node = GraphNode(
                    node_id=record["nodeId"],
                    labels=record["labels"],
                    name=record["name"],
                    properties=record["properties"]
                )
                self.knowledge_sources.append(node)
            logger.info(f"加载了{len(self.knowledge_sources)}个知识来源节点")

            return {
                'equipment_categories': len(self.equipment_categories),
                'equipments': len(self.equipments),
                'components': len(self.components),
                'faults': len(self.faults),
                'fault_phenomenons': len(self.fault_phenomenons),
                'fault_reasons': len(self.fault_reasons),
                'maintenance_actions': len(self.maintenance_actions),
                'safety_notices': len(self.safety_notices),
                'knowledge_sources': len(self.knowledge_sources)
            }



    def build_documents(self) -> List[Document]:
        """
        构建设备文档，集成相关的部件和故障信息
        将图数据转换为结构化文档
        Returns:
            结构化的设备文档列表
        """
        logger.info("正在构建设备文档...")

        documents = []

        with self.driver.session() as session:
            for equipment in self.equipments:
                try:
                    equipment_id = equipment.node_id
                    equipment_name = equipment.name

                    components_query = """
                    MATCH (e:Equipment {equipment_id: $equipment_id})-[:consists_of]->(c:Component)
                    RETURN c.name as name, c.type as type,
                           c.spec as spec
                    ORDER BY c.name
                    """

                    components_result = session.run(components_query, {"equipment_id": equipment_id})
                    components_info = []
                    for comp_record in components_result:
                        component_text = f"{comp_record['name']}"
                        if comp_record.get("type"):
                            component_text += f" ({comp_record['type']})"
                        if comp_record.get("spec"):
                            component_text += f" [规格: {comp_record['spec']}]"
                        components_info.append(component_text)

                    faults_query = """
                    MATCH (e:Equipment {equipment_id: $equipment_id})-[:has_fault]->(f:Fault)
                    RETURN f.name as name, f.fault_type as fault_type,
                           f.severity as severity, f.occurrence_frequency as frequency,
                           f.description as description
                    ORDER BY f.name
                    """

                    faults_result = session.run(faults_query, {"equipment_id": equipment_id})
                    faults_info = []
                    for fault_record in faults_result:
                        fault_text = f"故障: {fault_record['name']}"
                        if fault_record.get("fault_type"):
                            fault_text += f" [{fault_record['fault_type']}]"
                        if fault_record.get("description"):
                            fault_text += f"\n描述: {fault_record['description']}"
                        if fault_record.get("severity"):
                            fault_text += f"\n严重等级: {fault_record['severity']}"
                        if fault_record.get("frequency"):
                            fault_text += f"\n发生频率: {fault_record['frequency']}"
                        faults_info.append(fault_text)

                    content_parts = [f"# {equipment_name}"]

                    if equipment.properties.get("type"):
                        content_parts.append(f"\n设备类型: {equipment.properties['type']}")

                    if equipment.properties.get("model"):
                        content_parts.append(f"型号: {equipment.properties['model']}")

                    if components_info:
                        content_parts.append("\n## 组成部件")
                        for i, component in enumerate(components_info, 1):
                            content_parts.append(f"{i}. {component}")

                    if faults_info:
                        content_parts.append("\n## 常见故障")
                        for i, fault in enumerate(faults_info, 1):
                            content_parts.append(f"\n### {i}. {fault}")

                    full_content = "\n".join(content_parts)

                    doc = Document(
                        page_content=full_content,
                        metadata={
                            "neo4j_node_id": equipment_id,
                            "neo4j_label": "Equipment",
                            "entity_type": "Equipment",
                            "equipment_type": equipment.properties.get("type", "未知"),
                            "doc_type": "entity",
                            "chunk_id": f"{equipment_id}_entity",
                            "parent_id": "",
                            "chunk_index": 0
                        }
                    )

                    documents.append(doc)

                except Exception as e:
                    logger.warning(f"构建设备文档失败 {equipment_name} (ID: {equipment_id}): {e}")
                    continue

        for component in self.components:
            try:
                component_id = component.node_id
                component_name = component.name
                
                content_parts = [f"# 部件: {component_name}"]
                
                if component.properties.get("type"):
                    content_parts.append(f"\n部件类型: {component.properties['type']}")
                if component.properties.get("spec"):
                    content_parts.append(f"规格: {component.properties['spec']}")
                
                with self.driver.session() as session:
                    related_faults_query = """
                    MATCH (c:Component {component_id: $component_id})-[:has_fault]->(f:Fault)
                    RETURN f.name as name, f.description as description
                    """
                    related_faults = session.run(related_faults_query, {"component_id": component_id})
                    fault_texts = []
                    for fault in related_faults:
                        fault_text = fault['name']
                        if fault.get('description'):
                            fault_text += f": {fault['description']}"
                        fault_texts.append(fault_text)
                    
                    if fault_texts:
                        content_parts.append("\n## 相关故障")
                        for ft in fault_texts:
                            content_parts.append(f"- {ft}")
                    
                    parent_query = """
                    MATCH (e:Equipment)-[:consists_of]->(c:Component {component_id: $component_id})
                    RETURN e.name as equipment_name
                    """
                    parent_result = session.run(parent_query, {"component_id": component_id})
                    parent = parent_result.single()
                    if parent:
                        content_parts.append(f"\n所属设备: {parent['equipment_name']}")
                
                full_content = "\n".join(content_parts)
                
                doc = Document(
                    page_content=full_content,
                    metadata={
                        "neo4j_node_id": component_id,
                        "neo4j_label": "Component",
                        "entity_type": "Component",
                        "doc_type": "entity",
                        "chunk_id": f"{component_id}_entity",
                        "parent_id": "",
                        "chunk_index": 0
                    }
                )
                documents.append(doc)
                
            except Exception as e:
                logger.warning(f"构建部件文档失败 {component_name}: {e}")
                continue

        for fault_phenomenon in self.fault_phenomenons:
            try:
                fp_id = fault_phenomenon.node_id
                fp_name = fault_phenomenon.name

                content_parts = [f"# 故障现象: {fp_name}"]

                if fault_phenomenon.properties.get("description"):
                    content_parts.append(f"\n描述: {fault_phenomenon.properties['description']}")

                with self.driver.session() as session:
                    fault_query = """
                    MATCH (f:Fault)-[:presents_as]->(fp:FaultPhenomenon {phenomenon_id: $fp_id})
                    RETURN f.fault_id as fault_id, f.name as name, f.description as description
                    """
                    fault_result = session.run(fault_query, {"fp_id": fp_id})
                    fault_records = list(fault_result)
                    fault_ids = [record["fault_id"] for record in fault_records if record.get("fault_id")]

                    fault_texts = []
                    for fault in fault_records:
                        fault_text = fault.get("name") or "未知故障"
                        if fault.get("description"):
                            fault_text += f": {fault['description']}"
                        fault_texts.append(fault_text)

                    if fault_texts:
                        content_parts.append("\n## 关联故障")
                        for i, ft in enumerate(fault_texts, 1):
                            content_parts.append(f"{i}. {ft}")

                    reasons_query = """
                    MATCH (f:Fault)-[:caused_by*1..3]->(fr:FaultReason)
                    WHERE f.fault_id IN $fault_ids
                    RETURN fr.cause_name as name, fr.description as description, fr.level as level
                    ORDER BY fr.level, fr.cause_name
                    """
                    reasons = session.run(reasons_query, {"fault_ids": fault_ids})
                    reason_texts = []
                    for reason in reasons:
                        reason_text = reason['name']
                        if reason.get('description'):
                            reason_text += f": {reason['description']}"
                        if reason.get('level') is not None:
                            reason_text += f"（层级: {reason['level']}）"
                        reason_texts.append(reason_text)

                    if reason_texts:
                        content_parts.append("\n## 可能原因")
                        for i, rt in enumerate(reason_texts, 1):
                            content_parts.append(f"{i}. {rt}")

                    steps_query = """
                    MATCH (f:Fault)-[:caused_by*1..3]->(fr:FaultReason)-[:fixed_by]->(ma:MaintenanceAction)
                    WHERE f.fault_id IN $fault_ids
                    RETURN ma.step_order as step_order, ma.description as description, ma.tools as tools
                    ORDER BY ma.step_order
                    """
                    steps = session.run(steps_query, {"fault_ids": fault_ids})
                    step_texts = []
                    for step in steps:
                        step_desc = step.get("description") or ""
                        if step.get("step_order") is not None:
                            step_text = f"{step['step_order']}. {step_desc}"
                        else:
                            step_text = step_desc
                        if step.get("tools"):
                            step_text += f"（工具: {step['tools']}）"
                        step_texts.append(step_text)

                    if step_texts:
                        content_parts.append("\n## 维修步骤")
                        for st in step_texts:
                            content_parts.append(st)

                full_content = "\n".join(content_parts)

                doc = Document(
                    page_content=full_content,
                    metadata={
                        "neo4j_node_id": fp_id,
                        "neo4j_label": "FaultPhenomenon",
                        "entity_type": "FaultPhenomenon",
                        "doc_type": "entity",
                        "chunk_id": f"{fp_id}_entity",
                        "parent_id": "",
                        "chunk_index": 0
                    }
                )
                documents.append(doc)

            except Exception as e:
                logger.warning(f"构建故障现象文档失败 {fp_name}: {e}")
                continue

        for fault in self.faults:
            try:
                fault_id = fault.node_id
                fault_name = fault.name

                content_parts = [f"# 故障: {fault_name}"]

                if fault.properties.get("description"):
                    content_parts.append(f"\n描述: {fault.properties['description']}")
                if fault.properties.get("severity"):
                    content_parts.append(f"严重等级: {fault.properties['severity']}")
                if fault.properties.get("fault_type"):
                    content_parts.append(f"故障类型: {fault.properties['fault_type']}")

                with self.driver.session() as session:
                    phenomenon_query = """
                    MATCH (f:Fault {fault_id: $fault_id})-[:presents_as]->(fp:FaultPhenomenon)
                    RETURN fp.description as description
                    ORDER BY fp.description
                    """
                    phenomenon_result = session.run(phenomenon_query, {"fault_id": fault_id})
                    phenomenon_texts = [record["description"] for record in phenomenon_result if record.get("description")]
                    if phenomenon_texts:
                        content_parts.append("\n## 故障现象")
                        for i, text in enumerate(phenomenon_texts, 1):
                            content_parts.append(f"{i}. {text}")

                    reasons_query = """
                    MATCH (f:Fault {fault_id: $fault_id})-[:caused_by*1..3]->(fr:FaultReason)
                    RETURN fr.cause_name as name, fr.description as description, fr.level as level
                    ORDER BY fr.level, fr.cause_name
                    """
                    reasons_result = session.run(reasons_query, {"fault_id": fault_id})
                    reasons_by_level = {}
                    for reason in reasons_result:
                        level = reason.get("level")
                        reason_name = reason.get("name") or "未知原因"
                        reason_desc = reason.get("description")
                        reason_text = reason_name
                        if reason_desc:
                            reason_text += f": {reason_desc}"
                        reasons_by_level.setdefault(level, []).append(reason_text)

                    if reasons_by_level:
                        content_parts.append("\n## 原因树")
                        for level in sorted(reasons_by_level.keys(), key=lambda x: x if x is not None else 999):
                            level_label = f"层级 {level}" if level is not None else "层级 未知"
                            content_parts.append(f"{level_label}：")
                            for reason_text in reasons_by_level[level]:
                                content_parts.append(f"- {reason_text}")

                    steps_query = """
                    MATCH (f:Fault {fault_id: $fault_id})-[:caused_by*1..3]->(fr:FaultReason)-[:fixed_by]->(ma:MaintenanceAction)
                    OPTIONAL MATCH (ma)-[:has_notice]->(sn:SafetyNotice)
                    RETURN ma.action_id as action_id, ma.step_order as step_order, ma.description as description, ma.tools as tools,
                           collect(DISTINCT sn.description) as notices
                    ORDER BY ma.step_order
                    """
                    steps_result = session.run(steps_query, {"fault_id": fault_id})
                    step_lines = []
                    for step in steps_result:
                        step_desc = step.get("description") or ""
                        if step.get("step_order") is not None:
                            step_text = f"{step['step_order']}. {step_desc}"
                        else:
                            step_text = step_desc
                        if step.get("tools"):
                            step_text += f"（工具: {step['tools']}）"
                        notices = [notice for notice in step.get("notices", []) if notice]
                        if notices:
                            step_text += f"；注意事项: {'；'.join(notices)}"
                        step_lines.append(step_text)

                    if step_lines:
                        content_parts.append("\n## 维修步骤")
                        for step_line in step_lines:
                            content_parts.append(step_line)

                full_content = "\n".join(content_parts)

                doc = Document(
                    page_content=full_content,
                    metadata={
                        "neo4j_node_id": fault_id,
                        "neo4j_label": "Fault",
                        "entity_type": "Fault",
                        "doc_type": "entity",
                        "chunk_id": f"{fault_id}_entity",
                        "parent_id": "",
                        "chunk_index": 0
                    }
                )
                documents.append(doc)

            except Exception as e:
                logger.warning(f"构建故障文档失败 {fault_name}: {e}")
                continue

        for fault_reason in self.fault_reasons:
            try:
                fr_id = fault_reason.node_id
                fr_name = fault_reason.name

                content_parts = [f"# 故障原因: {fr_name}"]

                if fault_reason.properties.get("description"):
                    content_parts.append(f"\n描述: {fault_reason.properties['description']}")
                if fault_reason.properties.get("solution"):
                    content_parts.append(f"\n解决方案: {fault_reason.properties['solution']}")

                with self.driver.session() as session:
                    parent_query = """
                    MATCH (fr:FaultReason {cause_id: $fr_id})-[:caused_by]->(parent:FaultReason)
                    RETURN parent.cause_name as name, parent.description as description, parent.level as level
                    ORDER BY parent.level, parent.cause_name
                    """
                    parent_result = session.run(parent_query, {"fr_id": fr_id})
                    parent_texts = []
                    for parent in parent_result:
                        parent_text = parent.get("name") or "未知原因"
                        if parent.get("description"):
                            parent_text += f": {parent['description']}"
                        if parent.get("level") is not None:
                            parent_text += f"（层级: {parent['level']}）"
                        parent_texts.append(parent_text)

                    if parent_texts:
                        content_parts.append("\n## 父原因")
                        for i, text in enumerate(parent_texts, 1):
                            content_parts.append(f"{i}. {text}")

                    child_query = """
                    MATCH (child:FaultReason)-[:caused_by]->(fr:FaultReason {cause_id: $fr_id})
                    RETURN child.cause_name as name, child.description as description, child.level as level
                    ORDER BY child.level, child.cause_name
                    """
                    child_result = session.run(child_query, {"fr_id": fr_id})
                    child_texts = []
                    for child in child_result:
                        child_text = child.get("name") or "未知原因"
                        if child.get("description"):
                            child_text += f": {child['description']}"
                        if child.get("level") is not None:
                            child_text += f"（层级: {child['level']}）"
                        child_texts.append(child_text)

                    if child_texts:
                        content_parts.append("\n## 子原因")
                        for i, text in enumerate(child_texts, 1):
                            content_parts.append(f"{i}. {text}")

                full_content = "\n".join(content_parts)

                doc = Document(
                    page_content=full_content,
                    metadata={
                        "neo4j_node_id": fr_id,
                        "neo4j_label": "FaultReason",
                        "entity_type": "FaultReason",
                        "doc_type": "entity",
                        "chunk_id": f"{fr_id}_entity",
                        "parent_id": "",
                        "chunk_index": 0
                    }
                )
                documents.append(doc)

            except Exception as e:
                logger.warning(f"构建故障原因文档失败 {fr_name}: {e}")
                continue

        for step in self.maintenance_actions:
            try:
                step_id = step.node_id
                step_name = step.name

                content_parts = [f"# 维修步骤: {step_name}"]

                if step.properties.get("description"):
                    content_parts.append(f"\n描述: {step.properties['description']}")
                tools_required = step.properties.get("tools_required") or step.properties.get("tools")
                if tools_required:
                    content_parts.append(f"\n所需工具: {tools_required}")

                with self.driver.session() as session:
                    related_faults_query = """
                    MATCH (f:Fault)-[:caused_by*1..3]->(fr:FaultReason)-[:fixed_by]->(ma:MaintenanceAction {action_id: $step_id})
                    RETURN DISTINCT f.name as name, f.description as description
                    ORDER BY f.name
                    """
                    related_faults = session.run(related_faults_query, {"step_id": step_id})
                    fault_texts = []
                    for fault in related_faults:
                        fault_text = fault.get("name") or "未知故障"
                        if fault.get("description"):
                            fault_text += f": {fault['description']}"
                        fault_texts.append(fault_text)

                    if fault_texts:
                        content_parts.append("\n## 关联故障")
                        for i, text in enumerate(fault_texts, 1):
                            content_parts.append(f"{i}. {text}")

                    related_reasons_query = """
                    MATCH (fr:FaultReason)-[:fixed_by]->(ma:MaintenanceAction {action_id: $step_id})
                    RETURN fr.cause_name as name, fr.description as description, fr.level as level
                    ORDER BY fr.level, fr.cause_name
                    """
                    related_reasons = session.run(related_reasons_query, {"step_id": step_id})
                    reason_texts = []
                    for reason in related_reasons:
                        reason_text = reason.get("name") or "未知原因"
                        if reason.get("description"):
                            reason_text += f": {reason['description']}"
                        if reason.get("level") is not None:
                            reason_text += f"（层级: {reason['level']}）"
                        reason_texts.append(reason_text)

                    if reason_texts:
                        content_parts.append("\n## 关联原因")
                        for i, text in enumerate(reason_texts, 1):
                            content_parts.append(f"{i}. {text}")

                    notices_query = """
                    MATCH (ma:MaintenanceAction {action_id: $step_id})-[:has_notice]->(sn:SafetyNotice)
                    RETURN sn.description as description, sn.level as level, sn.consequence as consequence
                    ORDER BY sn.level
                    """
                    notices = session.run(notices_query, {"step_id": step_id})
                    notice_texts = []
                    for notice in notices:
                        notice_text = notice.get("description") or ""
                        if notice.get("level"):
                            notice_text += f"（风险等级: {notice['level']}）"
                        if notice.get("consequence"):
                            notice_text += f"；后果: {notice['consequence']}"
                        notice_texts.append(notice_text)

                    if notice_texts:
                        content_parts.append("\n## 关联注意事项")
                        for i, text in enumerate(notice_texts, 1):
                            content_parts.append(f"{i}. {text}")

                full_content = "\n".join(content_parts)

                doc = Document(
                    page_content=full_content,
                    metadata={
                        "neo4j_node_id": step_id,
                        "neo4j_label": "MaintenanceAction",
                        "entity_type": "MaintenanceAction",
                        "doc_type": "entity",
                        "chunk_id": f"{step_id}_entity",
                        "parent_id": "",
                        "chunk_index": 0
                    }
                )
                documents.append(doc)

            except Exception as e:
                logger.warning(f"构建维修步骤文档失败 {step_name}: {e}")
                continue

        for notice in self.safety_notices:
            try:
                notice_id = notice.node_id
                notice_name = notice.name
                
                content_parts = [f"# 注意事项: {notice_name}"]
                
                if notice.properties.get("description"):
                    content_parts.append(f"\n描述: {notice.properties['description']}")
                if notice.properties.get("severity"):
                    content_parts.append(f"\n严重程度: {notice.properties['severity']}")
                
                full_content = "\n".join(content_parts)
                
                doc = Document(
                    page_content=full_content,
                    metadata={
                        "neo4j_node_id": notice_id,
                        "neo4j_label": "SafetyNotice",
                        "entity_type": "SafetyNotice",
                        "doc_type": "entity",
                        "chunk_id": f"{notice_id}_entity",
                        "parent_id": "",
                        "chunk_index": 0
                    }
                )
                documents.append(doc)
                
            except Exception as e:
                logger.warning(f"构建注意事项文档失败 {notice_name}: {e}")
                continue

        self.documents = documents
        logger.info(f"成功构建 {len(documents)} 个文档（设备:{len(self.equipments)}, 部件:{len(self.components)}, 故障现象:{len(self.fault_phenomenons)}, 故障原因:{len(self.fault_reasons)}, 维修步骤:{len(self.maintenance_actions)}, 注意事项:{len(self.safety_notices)}）")
        return documents

    def build_documents_for_node_ids(self, node_ids: List[str]) -> List[Document]:
        """
        仅构建指定 neo4j_node_id 对应的文档（用于向量索引增量更新）。

        说明：
        - 这里的 neo4j_node_id 指的是你在入库时写入向量 payload 的业务主键（如 Equipment.equipment_id、Component.component_id 等）
        - 文档内容仍会通过图查询补全其关联信息（例如设备文档会包含其部件/故障），因此不需要为所有关联节点都传入 node_ids。
        """
        node_id_set = set([str(x) for x in (node_ids or []) if x])
        if not node_id_set:
            return []

        # 备份原始列表，避免污染后续 full build
        equipments_bak = self.equipments
        components_bak = self.components
        faults_bak = self.faults
        fault_phenomenons_bak = self.fault_phenomenons
        fault_reasons_bak = self.fault_reasons
        maintenance_actions_bak = self.maintenance_actions
        safety_notices_bak = self.safety_notices
        knowledge_sources_bak = self.knowledge_sources

        try:
            self.equipments = [x for x in equipments_bak if x.node_id in node_id_set]
            self.components = [x for x in components_bak if x.node_id in node_id_set]
            self.faults = [x for x in faults_bak if x.node_id in node_id_set]
            self.fault_phenomenons = [x for x in fault_phenomenons_bak if x.node_id in node_id_set]
            self.fault_reasons = [x for x in fault_reasons_bak if x.node_id in node_id_set]
            self.maintenance_actions = [x for x in maintenance_actions_bak if x.node_id in node_id_set]
            self.safety_notices = [x for x in safety_notices_bak if x.node_id in node_id_set]
            self.knowledge_sources = [x for x in knowledge_sources_bak if x.node_id in node_id_set]

            return self.build_documents()
        finally:
            # 还原列表
            self.equipments = equipments_bak
            self.components = components_bak
            self.faults = faults_bak
            self.fault_phenomenons = fault_phenomenons_bak
            self.fault_reasons = fault_reasons_bak
            self.maintenance_actions = maintenance_actions_bak
            self.safety_notices = safety_notices_bak
            self.knowledge_sources = knowledge_sources_bak

    def chunk_documents(self, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:

        """可以进行优化的分块方法。"""
        """
        对文档进行分块处理

        Args:
            chunk_size: 分块大小
            chunk_overlap: 重叠大小

        Returns:
            分块后的文档列表
        """
        logger.info(f"正在进行文档分块，块大小: {chunk_size}, 重叠: {chunk_overlap}")

        if not self.documents:
            raise ValueError("请先构建文档")

        chunks = []
        chunk_id = 0

        for doc in self.documents:
            content = doc.page_content

            # 简单的按长度分块
            if len(content) <= chunk_size:
                # 内容较短，不需要分块
                chunk = Document(
                    page_content=content,
                    metadata={
                        **doc.metadata,
                        "chunk_id": f"{doc.metadata['neo4j_node_id']}_chunk_{chunk_id}",
                        "parent_id": doc.metadata["neo4j_node_id"],
                        "chunk_index": 0,
                        "doc_type": "chunk"
                    }
                )
                chunks.append(chunk)
                chunk_id += 1
            else:
                # 按章节分块（基于标题）
                sections = content.split('\n## ')
                if len(sections) <= 1:
                    # 没有二级标题，按长度强制分块
                    total_chunks = (len(content) - 1) // (chunk_size - chunk_overlap) + 1

                    for i in range(total_chunks):
                        start = i * (chunk_size - chunk_overlap)
                        end = min(start + chunk_size, len(content))

                        chunk_content = content[start:end]

                        chunk = Document(
                            page_content=chunk_content,
                            metadata={
                                **doc.metadata,
                                "chunk_id": f"{doc.metadata['neo4j_node_id']}_chunk_{chunk_id}",
                                "parent_id": doc.metadata["neo4j_node_id"],
                                "chunk_index": i,
                                "doc_type": "chunk"
                            }
                        )
                        chunks.append(chunk)
                        chunk_id += 1
                else:
                    # 按章节分块
                    for i, section in enumerate(sections):
                        if i == 0:
                            # 第一个部分包含标题
                            chunk_content = section
                        else:
                            # 其他部分添加章节标题
                            chunk_content = f"## {section}"

                        chunk = Document(
                            page_content=chunk_content,
                            metadata={
                                **doc.metadata,
                                "chunk_id": f"{doc.metadata['neo4j_node_id']}_chunk_{chunk_id}",
                                "parent_id": doc.metadata["neo4j_node_id"],
                                "chunk_index": i,
                                "doc_type": "chunk"
                            }
                        )
                        chunks.append(chunk)
                        chunk_id += 1

        self.chunks = chunks
        logger.info(f"文档分块完成，共生成 {len(chunks)} 个块")
        return chunks

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据统计信息

        Returns:
            统计信息字典
        """
        stats = {
            'total_equipment_categories': len(self.equipment_categories),
            'total_equipments': len(self.equipments),
            'total_components': len(self.components),
            'total_faults': len(self.faults),
            'total_fault_phenomenons': len(self.fault_phenomenons),
            'total_fault_reasons': len(self.fault_reasons),
            'total_maintenance_actions': len(self.maintenance_actions),
            'total_safety_notices': len(self.safety_notices),
            'total_knowledge_sources': len(self.knowledge_sources),
            'total_documents': len(self.documents),
            'total_chunks': len(self.chunks)
        }

        if self.documents:
            equipment_types = {}

            for doc in self.documents:
                equipment_type = doc.metadata.get('equipment_type', '未知')
                equipment_types[equipment_type] = equipment_types.get(equipment_type, 0) + 1

            stats.update({
                'equipment_types': equipment_types,
                'avg_content_length': sum(doc.metadata.get('content_length', 0) for doc in self.documents) / len(
                    self.documents),
                'avg_chunk_size': sum(chunk.metadata.get('chunk_size', 0) for chunk in self.chunks) / len(
                    self.chunks) if self.chunks else 0
            })

        return stats

    def __del__(self):
        """析构函数，确保关闭连接"""
        self.close()