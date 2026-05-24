import json
import logging
import requests
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)


@dataclass
class EntityKeyValue:
    """实体键值对"""
    entity_name: str
    index_keys: List[str]  # 索引键列表
    value_content:str   # 详细描述内容
    entity_type:str    # 实体类型 (Equipment, Component, FaultPhenomenon)
    metadata:Dict[str, Any]

@dataclass
class RelationKeyValue:
    """关系键值对"""
    relation_id: str
    index_keys: List[str]  # 多个索引键（可包含全局主题）
    value_content: str     # 关系描述内容
    relation_type: str     # 关系类型
    source_entity: str     # 源实体
    target_entity: str     # 目标实体
    metadata: Dict[str, Any]




"""需要修改"""
class GraphIndexingModule:
    """
        图索引模块
        核心功能：
        1. 为实体创建键值对（名称作为唯一索引键）
        2. 为关系创建键值对（多个索引键，包含全局主题）
        3. 去重和优化图操作
        4. 支持增量更新
        """
    def __init__(self,config,llm_client):
        self.config = config
        self.llm_client = llm_client
        
        # 添加Ollama支持
        self.llm_provider = config.llm_provider
        self.ollama_base_url = config.ollama_base_url.rstrip('/')
        self.ollama_model = config.ollama_model

        # 键值对存储
        self.entity_kv_store: Dict[str, EntityKeyValue] = {}
        self.relation_kv_store: Dict[str, RelationKeyValue] = {}

        # 索引映射：key -> entity/relation IDs
        self.key_to_entities: Dict[str, List[str]] = defaultdict(list)
        self.key_to_relations: Dict[str, List[str]] = defaultdict(list)



    def create_entity_key_values(self, equipment_categories: List[Any] = None, equipments: List[Any] = None, 
                                   components: List[Any] = None, faults: List[Any] = None,
                                   fault_phenomenons: List[Any] = None, fault_reasons: List[Any] = None,
                                   maintenance_actions: List[Any] = None, safety_notices: List[Any] = None,
                                   knowledge_sources: List[Any] = None) -> Dict[str, EntityKeyValue]:
        """
        为实体创建键值对结构
        每个实体使用其名称作为唯一索引键
        """
        logger.info("开始创建实体键值对...")

        if equipment_categories is None:
            equipment_categories = []
        if equipments is None:
            equipments = []
        if components is None:
            components = []
        if faults is None:
            faults = []
        if fault_phenomenons is None:
            fault_phenomenons = []
        if fault_reasons is None:
            fault_reasons = []
        if maintenance_actions is None:
            maintenance_actions = []
        if safety_notices is None:
            safety_notices = []
        if knowledge_sources is None:
            knowledge_sources = []

        for equipment in equipments:
            entity_id = equipment.node_id
            entity_name = equipment.name or f"设备_{entity_id}"

            content_parts = [f"设备名称：{entity_name}"]

            if hasattr(equipment, 'properties'):
                props = equipment.properties
                if props.get('type'):
                    content_parts.append(f"类型: {props['type']}")
                if props.get('model'):
                    content_parts.append(f"型号: {props['model']}")

            entity_kv = EntityKeyValue(
                entity_name=entity_name,
                index_keys=[entity_name],
                value_content='\n'.join(content_parts),
                entity_type="Equipment",
                metadata={
                    "neo4j_node_id": entity_id,
                    "neo4j_label": "Equipment",
                    "entity_type": "Equipment",
                    "properties": getattr(equipment, 'properties', {}),
                }
            )

            self.entity_kv_store[entity_id] = entity_kv
            self.key_to_entities[entity_name].append(entity_id)

        # 处理装备大类实体
        for category in equipment_categories:
            entity_id = category.node_id
            entity_name = category.name or f"装备大类_{entity_id}"

            content_parts = [f"装备大类名称：{entity_name}"]

            if hasattr(category, 'properties'):
                props = category.properties
                if props.get('description'):
                    content_parts.append(f"描述: {props['description']}")

            entity_kv = EntityKeyValue(
                entity_name=entity_name,
                index_keys=[entity_name],
                value_content='\n'.join(content_parts),
                entity_type="EquipmentCategory",
                metadata={
                    "neo4j_node_id": entity_id,
                    "neo4j_label": "EquipmentCategory",
                    "entity_type": "EquipmentCategory",
                    "properties": getattr(category, 'properties', {})
                }
            )

            self.entity_kv_store[entity_id] = entity_kv
            self.key_to_entities[entity_name].append(entity_id)

        # 处理故障实体
        for fault in faults:
            entity_id = fault.node_id
            entity_name = fault.name or f"故障_{entity_id}"

            content_parts = [f"故障名称：{entity_name}"]

            if hasattr(fault, 'properties'):
                props = fault.properties
                if props.get('description'):
                    content_parts.append(f"描述: {props['description']}")

            entity_kv = EntityKeyValue(
                entity_name=entity_name,
                index_keys=[entity_name],
                value_content='\n'.join(content_parts),
                entity_type="Fault",
                metadata={
                    "neo4j_node_id": entity_id,
                    "neo4j_label": "Fault",
                    "entity_type": "Fault",
                    "properties": getattr(fault, 'properties', {})
                }
            )

            self.entity_kv_store[entity_id] = entity_kv
            self.key_to_entities[entity_name].append(entity_id)

        # 处理部件实体
        for component in components:
            entity_id = component.node_id
            entity_name = component.name or f"部件_{entity_id}"

            content_parts = [f"部件名称: {entity_name}"]

            if hasattr(component, 'properties'):
                props = component.properties
                if props.get('component_type'):
                    content_parts.append(f"类别: {props['component_type']}")
                if props.get('specification'):
                    content_parts.append(f"规格: {props['specification']}")
                if props.get('description'):
                    content_parts.append(f"描述: {props['description']}")

            entity_kv = EntityKeyValue(
                entity_name=entity_name,
                index_keys=[entity_name],
                value_content='\n'.join(content_parts),
                entity_type="Component",
                metadata={
                    "neo4j_node_id": entity_id,
                    "neo4j_label": "Component",
                    "entity_type": "Component",
                    "properties": getattr(component, 'properties', {})
                }
            )

            self.entity_kv_store[entity_id] = entity_kv
            self.key_to_entities[entity_name].append(entity_id)

        # 处理故障现象实体
        for fault in fault_phenomenons:
            entity_id = fault.node_id
            entity_name = fault.name or f"故障现象_{entity_id}"

            content_parts = [f"故障现象: {entity_name}"]

            if hasattr(fault, 'properties'):
                props = fault.properties
                if props.get('description'):
                    content_parts.append(f"描述: {props['description']}")
                if props.get('severity'):
                    content_parts.append(f"严重程度: {props['severity']}")
                if props.get('frequency'):
                    content_parts.append(f"发生频率: {props['frequency']}")

            entity_kv = EntityKeyValue(
                entity_name=entity_name,
                index_keys=[entity_name],
                value_content='\n'.join(content_parts),
                entity_type="FaultPhenomenon",
                metadata={
                    "neo4j_node_id": entity_id,
                    "neo4j_label": "FaultPhenomenon",
                    "entity_type": "FaultPhenomenon",
                    "properties": getattr(fault, 'properties', {})
                }
            )

            self.entity_kv_store[entity_id] = entity_kv
            self.key_to_entities[entity_name].append(entity_id)

        # 处理故障原因实体
        for fault_reason in fault_reasons:
            entity_id = fault_reason.node_id
            entity_name = fault_reason.name or f"故障原因_{entity_id}"

            content_parts = [f"故障原因: {entity_name}"]

            if hasattr(fault_reason, 'properties'):
                props = fault_reason.properties
                if props.get('description'):
                    content_parts.append(f"描述: {props['description']}")
                if props.get('fault_reason_type'):
                    content_parts.append(f"类型: {props['fault_reason_type']}")

            entity_kv = EntityKeyValue(
                entity_name=entity_name,
                index_keys=[entity_name],
                value_content='\n'.join(content_parts),
                entity_type="FaultReason",
                metadata={
                    "neo4j_node_id": entity_id,
                    "neo4j_label": "FaultReason",
                    "entity_type": "FaultReason",
                    "properties": getattr(fault_reason, 'properties', {})
                }
            )

            self.entity_kv_store[entity_id] = entity_kv
            self.key_to_entities[entity_name].append(entity_id)

        # 处理维修步骤实体
        for maintenance_step in maintenance_actions:
            entity_id = maintenance_step.node_id
            entity_name = maintenance_step.name or f"维修步骤_{entity_id}"

            content_parts = [f"维修步骤: {entity_name}"]

            if hasattr(maintenance_step, 'properties'):
                props = maintenance_step.properties
                if props.get('step_num'):
                    content_parts.append(f"步骤序号: {props['step_num']}")
                if props.get('tools'):
                    content_parts.append(f"所需工具: {props['tools']}")
                if props.get('duration'):
                    content_parts.append(f"预计时长: {props['duration']}")

            entity_kv = EntityKeyValue(
                entity_name=entity_name,
                index_keys=[entity_name],
                value_content='\n'.join(content_parts),
                entity_type="MaintenanceStep",
                metadata={
                    "neo4j_node_id": entity_id,
                    "neo4j_label": "MaintenanceStep",
                    "entity_type": "MaintenanceStep",
                    "properties": getattr(maintenance_step, 'properties', {})
                }
            )

            self.entity_kv_store[entity_id] = entity_kv
            self.key_to_entities[entity_name].append(entity_id)

        # 处理注意事项实体
        for attention in safety_notices:
            entity_id = attention.node_id
            entity_name = attention.name or f"注意事项_{entity_id}"

            content_parts = [f"注意事项: {entity_name}"]

            if hasattr(attention, 'properties'):
                props = attention.properties
                if props.get('risk_level'):
                    content_parts.append(f"风险等级: {props['risk_level']}")
                if props.get('source'):
                    content_parts.append(f"来源: {props['source']}")

            entity_kv = EntityKeyValue(
                entity_name=entity_name,
                index_keys=[entity_name],
                value_content='\n'.join(content_parts),
                entity_type="Attention",
                metadata={
                    "neo4j_node_id": entity_id,
                    "neo4j_label": "Attention",
                    "entity_type": "Attention",
                    "properties": getattr(attention, 'properties', {})
                }
            )

            self.entity_kv_store[entity_id] = entity_kv
            self.key_to_entities[entity_name].append(entity_id)

        # 处理安全注意事项实体
        for safety_notice in safety_notices:
            entity_id = safety_notice.node_id
            entity_name = safety_notice.name or f"安全注意事项_{entity_id}"

            content_parts = [f"安全注意事项：{entity_name}"]

            if hasattr(safety_notice, 'properties'):
                props = safety_notice.properties
                if props.get('description'):
                    content_parts.append(f"描述: {props['description']}")
                if props.get('risk_level'):
                    content_parts.append(f"风险等级: {props['risk_level']}")

            entity_kv = EntityKeyValue(
                entity_name=entity_name,
                index_keys=[entity_name],
                value_content='\n'.join(content_parts),
                entity_type="SafetyNotice",
                metadata={
                    "neo4j_node_id": entity_id,
                    "neo4j_label": "SafetyNotice",
                    "entity_type": "SafetyNotice",
                    "properties": getattr(safety_notice, 'properties', {})
                }
            )

            self.entity_kv_store[entity_id] = entity_kv
            self.key_to_entities[entity_name].append(entity_id)

        # 处理知识来源实体
        for knowledge_source in knowledge_sources:
            entity_id = knowledge_source.node_id
            entity_name = knowledge_source.name or f"知识来源_{entity_id}"

            content_parts = [f"知识来源：{entity_name}"]

            if hasattr(knowledge_source, 'properties'):
                props = knowledge_source.properties
                if props.get('source_type'):
                    content_parts.append(f"类型: {props['source_type']}")

            entity_kv = EntityKeyValue(
                entity_name=entity_name,
                index_keys=[entity_name],
                value_content='\n'.join(content_parts),
                entity_type="KnowledgeSource",
                metadata={
                    "neo4j_node_id": entity_id,
                    "neo4j_label": "KnowledgeSource",
                    "entity_type": "KnowledgeSource",
                    "properties": getattr(knowledge_source, 'properties', {})
                }
            )

            self.entity_kv_store[entity_id] = entity_kv
            self.key_to_entities[entity_name].append(entity_id)

        logger.info(f"实体键值对创建完成，共 {len(self.entity_kv_store)} 个实体")
        return self.entity_kv_store

    def _generate_relation_index_keys(self, source_entity: EntityKeyValue,
                                          target_entity: EntityKeyValue,
                                          relation_type: str) -> List[str]:
            """
            为关系生成多个索引键，包含全局主题
            """
            keys = [relation_type]  # 基础关系类型键

            # 根据关系类型和实体类型生成主题键
            if relation_type == "HAS_COMPONENT":
                # 设备-部件关系的主题键
                keys.extend([
                    "设备组成",
                    "部件配置",
                    f"{source_entity.entity_name}_部件",
                    target_entity.entity_name
                ])
            elif relation_type == "HAS_FAULT":
                # 设备-故障现象关系的主题键
                keys.extend([
                    "设备故障",
                    "故障现象",
                    f"{source_entity.entity_name}_故障",
                    "维修问题"
                ])
            elif relation_type == "CAUSED_BY":
                # 故障原因关系的主题键
                keys.extend([
                    "故障原因",
                    "故障分析",
                    f"{target_entity.entity_name}_原因",
                    "故障诊断"
                ])
            elif relation_type == "NEED_MAINTENANCE":
                # 故障原因-维修步骤关系的主题键
                keys.extend([
                    "维修步骤",
                    "故障维修",
                    f"{source_entity.entity_name}_维修",
                    "维修方案"
                ])
            elif relation_type == "INVOLVES_COMPONENT":
                # 维修步骤-部件关系的主题键
                keys.extend([
                    "维修部件",
                    "部件维修",
                    f"{target_entity.entity_name}_维修",
                    "维修操作"
                ])
            elif relation_type == "NEED_ATTENTION":
                # 维修步骤-注意事项关系的主题键
                keys.extend([
                    "注意事项",
                    "安全注意",
                    "维修安全",
                    "操作规范"
                ])
            elif relation_type == "LOCATED_AT":
                # 位置关系的主题键
                keys.extend([
                    "设备位置",
                    "安装位置",
                    target_entity.entity_name
                ])
            elif relation_type == "contains":
                # 包含关系的主题键
                keys.extend([
                    "包含关系",
                    "组成关系",
                    f"{source_entity.entity_name}_包含",
                    f"{target_entity.entity_name}_属于"
                ])
            elif relation_type == "consists_of":
                # 组成关系的主题键
                keys.extend([
                    "组成关系",
                    "结构关系",
                    f"{source_entity.entity_name}_组成",
                    f"{target_entity.entity_name}_构成"
                ])
            elif relation_type == "has_fault":
                # 有故障关系的主题键
                keys.extend([
                    "设备故障",
                    "故障关系",
                    f"{source_entity.entity_name}_故障",
                    "故障问题"
                ])
            elif relation_type == "caused_by":
                # 由...引起关系的主题键
                keys.extend([
                    "故障原因",
                    "因果关系",
                    f"{source_entity.entity_name}_原因",
                    "故障分析"
                ])
            elif relation_type == "fixed_by":
                # 由...修复关系的主题键
                keys.extend([
                    "维修方案",
                    "修复关系",
                    f"{source_entity.entity_name}_维修",
                    "维修步骤"
                ])
            elif relation_type == "has_notice":
                # 有注意事项关系的主题键
                keys.extend([
                    "注意事项",
                    "安全关系",
                    f"{source_entity.entity_name}_注意",
                    "安全提醒"
                ])
            elif relation_type == "relates_to":
                # 关联关系的主题键
                keys.extend([
                    "关联关系",
                    "相关关系",
                    f"{source_entity.entity_name}_关联",
                    f"{target_entity.entity_name}_相关"
                ])
            elif relation_type == "comes_from":
                # 来自关系的主题键
                keys.extend([
                    "来源关系",
                    "出处关系",
                    f"{source_entity.entity_name}_来源",
                    "知识来源"
                ])

            # 使用LLM增强关系索引键（可选）
            if getattr(self.config, 'enable_llm_relation_keys', False):
                enhanced_keys = self._llm_enhance_relation_keys(source_entity, target_entity, relation_type)
                keys.extend(enhanced_keys)

            # 去重并返回
            return list(set(keys))

    def create_relation_key_values(self, relationships: List[Tuple[str, str, str]]) -> Dict[str, RelationKeyValue]:
        """
        为关系创建键值对结构
        
        Args:
            relationships: 关系列表，每个元素为 (source_id, relation_type, target_id)
            
        Returns:
            关系键值对字典
        """
        logger.info("开始创建关系键值对...")
        
        for source_id, relation_type, target_id in relationships:
            # 获取源实体和目标实体
            source_entity = self.entity_kv_store.get(source_id)
            target_entity = self.entity_kv_store.get(target_id)
            
            if not source_entity or not target_entity:
                logger.warning(f"关系 {source_id}-[{relation_type}]->{target_id} 的实体不存在，跳过")
                continue
            
            # 生成关系ID
            relation_id = f"{source_id}_{relation_type}_{target_id}"
            
            # 生成索引键
            index_keys = self._generate_relation_index_keys(source_entity, target_entity, relation_type)
            
            # 构建关系描述内容
            content_parts = [
                f"关系类型: {relation_type}",
                f"源实体: {source_entity.entity_name}",
                f"目标实体: {target_entity.entity_name}"
            ]
            
            # 根据关系类型添加额外信息
            if relation_type == "HAS_COMPONENT":
                content_parts.append(f"设备组成: {source_entity.entity_name} 包含 {target_entity.entity_name}")
            elif relation_type == "HAS_FAULT":
                content_parts.append(f"设备故障: {source_entity.entity_name} 存在 {target_entity.entity_name} 故障")
            elif relation_type == "CAUSED_BY":
                content_parts.append(f"故障原因: {source_entity.entity_name} 由 {target_entity.entity_name} 导致")
            elif relation_type == "NEED_MAINTENANCE":
                content_parts.append(f"维修方案: {source_entity.entity_name} 需要 {target_entity.entity_name}")
            elif relation_type == "INVOLVES_COMPONENT":
                content_parts.append(f"维修部件: {source_entity.entity_name} 涉及 {target_entity.entity_name}")
            elif relation_type == "NEED_ATTENTION":
                content_parts.append(f"注意事项: {source_entity.entity_name} 需要注意 {target_entity.entity_name}")
            elif relation_type == "contains":
                content_parts.append(f"包含关系: {source_entity.entity_name} 包含 {target_entity.entity_name}")
            elif relation_type == "consists_of":
                content_parts.append(f"组成关系: {source_entity.entity_name} 由 {target_entity.entity_name} 组成")
            elif relation_type == "has_fault":
                content_parts.append(f"故障关系: {source_entity.entity_name} 存在 {target_entity.entity_name} 故障")
            elif relation_type == "caused_by":
                content_parts.append(f"因果关系: {source_entity.entity_name} 由 {target_entity.entity_name} 引起")
            elif relation_type == "fixed_by":
                content_parts.append(f"修复关系: {source_entity.entity_name} 由 {target_entity.entity_name} 修复")
            elif relation_type == "has_notice":
                content_parts.append(f"注意事项: {source_entity.entity_name} 有 {target_entity.entity_name} 注意事项")
            elif relation_type == "relates_to":
                content_parts.append(f"关联关系: {source_entity.entity_name} 与 {target_entity.entity_name} 相关")
            elif relation_type == "comes_from":
                content_parts.append(f"来源关系: {source_entity.entity_name} 来自 {target_entity.entity_name}")
            
            # 创建关系键值对
            relation_kv = RelationKeyValue(
                relation_id=relation_id,
                index_keys=index_keys,
                value_content='\n'.join(content_parts),
                relation_type=relation_type,
                source_entity=source_id,
                target_entity=target_id,
                metadata={
                    "source_entity_name": source_entity.entity_name,
                    "target_entity_name": target_entity.entity_name,
                    "source_entity_type": source_entity.entity_type,
                    "target_entity_type": target_entity.entity_type
                }
            )
            
            self.relation_kv_store[relation_id] = relation_kv
            
            # 更新索引映射
            for key in index_keys:
                self.key_to_relations[key].append(relation_id)
        
        logger.info(f"关系键值对创建完成，共 {len(self.relation_kv_store)} 个关系")
        return self.relation_kv_store

    def _llm_enhance_relation_keys(self, source_entity: EntityKeyValue,
                                   target_entity: EntityKeyValue,
                                   relation_type: str) -> List[str]:
        """
        使用LLM增强关系索引键，生成全局主题
        """
        prompt = f"""
        分析以下实体关系，生成相关的主题关键词：

        源实体: {source_entity.entity_name} ({source_entity.entity_type})
        目标实体: {target_entity.entity_name} ({target_entity.entity_type})
        关系类型: {relation_type}

        请生成3-5个相关的主题关键词，用于索引和检索。
        返回JSON格式：{{"keywords": ["关键词1", "关键词2", "关键词3"]}}
        """

        try:
            if self.llm_provider == "ollama":
                # 使用Ollama原生API
                url = f"{self.ollama_base_url}/api/chat"
                payload = {
                    "model": self.ollama_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 150
                    }
                }
                response = requests.post(url, json=payload, timeout=60)
                response.raise_for_status()
                result = response.json()
                content = result["message"]["content"].strip()
            else:
                # 使用vLLM OpenAI兼容API
                response = self.llm_client.chat.completions.create(
                    model=self.config.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=150
                )
                content = response.choices[0].message.content.strip()
            if "<|think|>" in content:
                # 移除<|think|>...</|think|>标签及其内容
                content = content.split("<|think|>")[-1].split("</|think>")[-1].strip()

            result = json.loads(content)
            return result.get("keywords", [])

        except Exception as e:
            logger.error(f"LLM增强关系索引键失败: {e}")
            return []

    def deduplicate_entities_and_relations(self):
        """
        去重相同的实体和关系，优化图操作
        """
        logger.info("开始去重实体和关系...")

        # 实体去重：基于名称
        name_to_entities = defaultdict(list)
        for entity_id, entity_kv in self.entity_kv_store.items():
            name_to_entities[entity_kv.entity_name].append(entity_id)

        # 合并重复实体
        entities_to_remove = []
        for name, entity_ids in name_to_entities.items():
            if len(entity_ids) > 1:
                # 保留第一个，合并其他的内容
                primary_id = entity_ids[0]
                primary_entity = self.entity_kv_store[primary_id]

                for entity_id in entity_ids[1:]:
                    duplicate_entity = self.entity_kv_store[entity_id]
                    # 合并内容
                    primary_entity.value_content += f"\n\n补充信息: {duplicate_entity.value_content}"
                    # 标记删除
                    entities_to_remove.append(entity_id)

        # 删除重复实体
        for entity_id in entities_to_remove:
            del self.entity_kv_store[entity_id]

        # 关系去重：基于源-目标-类型
        relation_signature_to_ids = defaultdict(list)
        for relation_id, relation_kv in self.relation_kv_store.items():
            signature = f"{relation_kv.source_entity}_{relation_kv.target_entity}_{relation_kv.relation_type}"
            relation_signature_to_ids[signature].append(relation_id)

        # 合并重复关系
        relations_to_remove = []
        for signature, relation_ids in relation_signature_to_ids.items():
            if len(relation_ids) > 1:
                # 保留第一个，删除其他
                for relation_id in relation_ids[1:]:
                    relations_to_remove.append(relation_id)

        # 删除重复关系
        for relation_id in relations_to_remove:
            del self.relation_kv_store[relation_id]

        # 重建索引映射
        self._rebuild_key_mappings()

        logger.info(f"去重完成 - 删除了 {len(entities_to_remove)} 个重复实体，{len(relations_to_remove)} 个重复关系")

    def _rebuild_key_mappings(self):
        """重建键到实体/关系的映射"""
        self.key_to_entities.clear()
        self.key_to_relations.clear()

        # 重建实体映射
        for entity_id, entity_kv in self.entity_kv_store.items():
            for key in entity_kv.index_keys:
                self.key_to_entities[key].append(entity_id)

        # 重建关系映射
        for relation_id, relation_kv in self.relation_kv_store.items():
            for key in relation_kv.index_keys:
                self.key_to_relations[key].append(relation_id)

    def get_entities_by_key(self, key: str) -> List[EntityKeyValue]:
        """根据索引键获取实体"""
        entity_ids = self.key_to_entities.get(key, [])
        return [self.entity_kv_store[eid] for eid in entity_ids if eid in self.entity_kv_store]

    def get_relations_by_key(self, key: str) -> List[RelationKeyValue]:
        """根据索引键获取关系"""
        relation_ids = self.key_to_relations.get(key, [])
        return [self.relation_kv_store[rid] for rid in relation_ids if rid in self.relation_kv_store]




    def get_statistics(self) -> Dict[str, Any]:
        """获取键值对存储统计信息"""
        return {
            "total_entities": len(self.entity_kv_store),
            "total_relations": len(self.relation_kv_store),
            "total_entity_keys": sum(len(kv.index_keys) for kv in self.entity_kv_store.values()),
            "total_relation_keys": sum(len(kv.index_keys) for kv in self.relation_kv_store.values()),
            "entity_types": {
                "Equipment": len([kv for kv in self.entity_kv_store.values() if kv.entity_type == "Equipment"]),
                "Component": len([kv for kv in self.entity_kv_store.values() if kv.entity_type == "Component"]),
                "FaultPhenomenon": len([kv for kv in self.entity_kv_store.values() if kv.entity_type == "FaultPhenomenon"]),
                "FaultReason": len([kv for kv in self.entity_kv_store.values() if kv.entity_type == "FaultReason"]),
                "MaintenanceStep": len([kv for kv in self.entity_kv_store.values() if kv.entity_type == "MaintenanceStep"]),
                "Attention": len([kv for kv in self.entity_kv_store.values() if kv.entity_type == "Attention"]),
                "EquipmentCategory": len([kv for kv in self.entity_kv_store.values() if kv.entity_type == "EquipmentCategory"]),
                "Fault": len([kv for kv in self.entity_kv_store.values() if kv.entity_type == "Fault"]),
                "SafetyNotice": len([kv for kv in self.entity_kv_store.values() if kv.entity_type == "SafetyNotice"]),
                "KnowledgeSource": len([kv for kv in self.entity_kv_store.values() if kv.entity_type == "KnowledgeSource"])
            }
        }