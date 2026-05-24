"""
CSV导入Neo4j脚本
将generate_csv目录下的CSV文件导入到Neo4j图数据库
支持去重功能
"""

import os
import logging
from typing import List, Dict, Any, Set, Tuple

import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()


class CSVToNeo4jImporter:
    """CSV导入Neo4j工具类"""

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        """
        初始化导入器

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
        self.csv_dir = "./generate_csv"
        
        self.dedup_stats = {
            "nodes_deduplicated": 0,
            "relations_deduplicated": 0,
            "nodes_skipped": 0,
            "relations_skipped": 0
        }

    def _session(self):
        """兼容 Neo4j 新旧 Bolt 协议会话创建"""
        try:
            return self.driver.session()
        except Exception as e:
            msg = str(e)
            if "Database name parameter" in msg or "Bolt Protocol Version(3, 0)" in msg:
                return self.driver.session()
            raise

    def connect(self):
        """连接Neo4j数据库"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
            )
            with self._session() as session:
                result = session.run("RETURN 1 as test")
                result.single()
            logger.info(f"成功连接到Neo4j: {self.uri}")
            return True
        except Exception as e:
            logger.error(f"连接Neo4j失败: {e}")
            return False

    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j连接已关闭")

    def clear_database(self):
        """清空数据库（可选）"""
        logger.warning("正在清空数据库...")
        with self._session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("数据库已清空")

    def create_indexes(self):
        """创建索引和约束以提高查询性能"""
        logger.info("创建索引与约束...")

        constraints = [
            "CREATE CONSTRAINT equipment_category_id_unique IF NOT EXISTS FOR (n:EquipmentCategory) REQUIRE n.category_id IS UNIQUE",
            "CREATE CONSTRAINT equipment_id_unique IF NOT EXISTS FOR (n:Equipment) REQUIRE n.equipment_id IS UNIQUE",
            "CREATE CONSTRAINT component_id_unique IF NOT EXISTS FOR (n:Component) REQUIRE n.component_id IS UNIQUE",
            "CREATE CONSTRAINT fault_id_unique IF NOT EXISTS FOR (n:Fault) REQUIRE n.fault_id IS UNIQUE",
            "CREATE CONSTRAINT fault_phenomenon_id_unique IF NOT EXISTS FOR (n:FaultPhenomenon) REQUIRE n.phenomenon_id IS UNIQUE",
            "CREATE CONSTRAINT fault_reason_id_unique IF NOT EXISTS FOR (n:FaultReason) REQUIRE n.cause_id IS UNIQUE",
            "CREATE CONSTRAINT maintenance_action_id_unique IF NOT EXISTS FOR (n:MaintenanceAction) REQUIRE n.action_id IS UNIQUE",
            "CREATE CONSTRAINT safety_notice_id_unique IF NOT EXISTS FOR (n:SafetyNotice) REQUIRE n.notice_id IS UNIQUE",
            "CREATE CONSTRAINT knowledge_source_id_unique IF NOT EXISTS FOR (n:KnowledgeSource) REQUIRE n.source_id IS UNIQUE",
        ]

        indexes = [
            "CREATE INDEX equipment_name_idx IF NOT EXISTS FOR (n:Equipment) ON (n.name)",
            "CREATE INDEX component_name_idx IF NOT EXISTS FOR (n:Component) ON (n.name)",
            "CREATE INDEX fault_name_idx IF NOT EXISTS FOR (n:Fault) ON (n.name)",
            "CREATE INDEX fault_reason_name_idx IF NOT EXISTS FOR (n:FaultReason) ON (n.cause_name)",
            "CREATE INDEX equipment_system_name_idx IF NOT EXISTS FOR (n:Equipment) ON (n.system_name)",
            "CREATE INDEX fault_phenomenon_desc_idx IF NOT EXISTS FOR (n:FaultPhenomenon) ON (n.description)",
        ]

        with self._session() as session:
            for query in constraints + indexes:
                try:
                    session.run(query)
                except Exception as e:
                    logger.warning(f"创建索引/约束时出错: {e}")

            # 全文索引：用于 CONTAINS 类检索的高频文本字段
            fulltext_queries = [
                (
                    "CALL db.index.fulltext.createNodeIndex('idx_fulltext_anchor', "
                    "['Equipment','Component','Fault','FaultPhenomenon','FaultReason','MaintenanceAction','SafetyNotice','KnowledgeSource'], "
                    "['name','description','cause_name','equipment_id','component_id','fault_id','phenomenon_id','cause_id','action_id','notice_id','source_id'])"
                ),
                (
                    "CALL db.index.fulltext.createNodeIndex('idx_fulltext_equipment_topic', "
                    "['Equipment'], ['name','type','model'])"
                ),
            ]
            for ft_query in fulltext_queries:
                try:
                    session.run(ft_query)
                except Exception as e:
                    # Neo4j 5+ 使用 CREATE FULLTEXT INDEX，旧版使用 procedure；这里保持兼容兜底
                    logger.warning(f"创建全文索引时出错（可能已存在或版本差异）: {e}")

        logger.info("索引与约束创建完成")

    def _escape_string(self, value: Any) -> str:
        """转义字符串中的特殊字符"""
        if value is None or pd.isna(value):
            return ""
        s = str(value)
        s = s.replace("\\", "\\\\")
        s = s.replace("'", "\\'")
        s = s.replace('"', '\\"')
        s = s.replace("\n", "\\n")
        s = s.replace("\r", "\\r")
        return s

    def deduplicate_csv(self, csv_path: str, id_column: str) -> Tuple[pd.DataFrame, int]:
        """
        对CSV文件进行去重

        Args:
            csv_path: CSV文件路径
            id_column: ID列名

        Returns:
            (去重后的DataFrame, 去重数量)
        """
        if not os.path.exists(csv_path):
            return pd.DataFrame(), 0

        df = pd.read_csv(csv_path)
        original_count = len(df)
        
        df_dedup = df.drop_duplicates(subset=[id_column], keep='first')
        dedup_count = original_count - len(df_dedup)
        
        if dedup_count > 0:
            logger.info(f"CSV去重: {csv_path} - 原始 {original_count} 条, 去重 {dedup_count} 条, 剩余 {len(df_dedup)} 条")
            self.dedup_stats["nodes_deduplicated"] += dedup_count
        
        return df_dedup, dedup_count

    def deduplicate_relations_csv(self, csv_path: str) -> Tuple[pd.DataFrame, int]:
        """
        对关系CSV文件进行去重

        Args:
            csv_path: CSV文件路径

        Returns:
            (去重后的DataFrame, 去重数量)
        """
        if not os.path.exists(csv_path):
            return pd.DataFrame(), 0

        df = pd.read_csv(csv_path)
        original_count = len(df)
        
        df_dedup = df.drop_duplicates(subset=['from_entity', 'from_id', 'relation_type', 'to_entity', 'to_id'], keep='first')
        dedup_count = original_count - len(df_dedup)
        
        if dedup_count > 0:
            logger.info(f"关系CSV去重: 原始 {original_count} 条, 去重 {dedup_count} 条, 剩余 {len(df_dedup)} 条")
            self.dedup_stats["relations_deduplicated"] += dedup_count
        
        return df_dedup, dedup_count

    def check_existing_nodes(self, label: str, id_field: str, ids: Set[str]) -> Set[str]:
        """
        检查数据库中已存在的节点

        Args:
            label: 节点标签
            id_field: ID字段名
            ids: 要检查的ID集合

        Returns:
            已存在的ID集合
        """
        if not ids:
            return set()
        
        existing_ids = set()
        
        with self.driver.session() as session:
            ids_list = list(ids)
            batch_size = 1000
            
            for i in range(0, len(ids_list), batch_size):
                batch = ids_list[i:i + batch_size]
                query = f"""
                MATCH (n:{label})
                WHERE n.{id_field} IN $ids
                RETURN n.{id_field} as id
                """
                result = session.run(query, ids=batch)
                for record in result:
                    existing_ids.add(record["id"])
        
        return existing_ids

    def check_existing_relations(self, relations: List[Dict]) -> Set[Tuple]:
        """
        检查数据库中已存在的关系

        Args:
            relations: 关系列表

        Returns:
            已存在的关系集合
        """
        if not relations:
            return set()
        
        existing_relations = set()
        
        relation_mapping = {
            ("EquipmentCategory", "Equipment"): {
                "CONTAINS": ("category_id", "equipment_id", "contains")
            },
            ("EquipmentCategory", "EquipmentCategory"): {
                "CONTAINS": ("category_id", "category_id", "contains")
            },
            ("EquipmentCategory", "Component"): {
                "CONTAINS": ("category_id", "component_id", "contains")
            },
            ("Equipment", "Component"): {
                "CONSISTS_OF": ("equipment_id", "component_id", "consists_of")
            },
            ("Equipment", "Equipment"): {
                "CONSISTS_OF": ("equipment_id", "equipment_id", "consists_of"),
                "CONTAINS": ("equipment_id", "equipment_id", "contains")
            },
            ("Equipment", "Fault"): {
                "HAS_FAULT": ("equipment_id", "fault_id", "has_fault")
            },
            ("Component", "Fault"): {
                "HAS_FAULT": ("component_id", "fault_id", "has_fault")
            },
            ("Fault", "FaultPhenomenon"): {
                "PRESENTS_AS": ("fault_id", "phenomenon_id", "presents_as")
            },
            ("Fault", "FaultReason"): {
                "CAUSED_BY": ("fault_id", "cause_id", "caused_by")
            },
            ("FaultReason", "FaultReason"): {
                "CAUSED_BY": ("cause_id", "cause_id", "caused_by")
            },
            ("FaultReason", "Component"): {
                "RELATES_TO": ("cause_id", "component_id", "relates_to")
            },
            ("FaultReason", "Equipment"): {
                "RELATES_TO": ("cause_id", "equipment_id", "relates_to")
            },
            ("Component", "Component"): {
                "CONSISTS_OF": ("component_id", "component_id", "consists_of")
            },
            ("FaultReason", "MaintenanceAction"): {
                "FIXED_BY": ("cause_id", "action_id", "fixed_by")
            },
            ("MaintenanceAction", "SafetyNotice"): {
                "HAS_NOTICE": ("action_id", "notice_id", "has_notice")
            },
            ("Equipment", "KnowledgeSource"): {
                "COMES_FROM": ("equipment_id", "source_id", "comes_from")
            },
            ("Component", "KnowledgeSource"): {
                "COMES_FROM": ("component_id", "source_id", "comes_from")
            },
            ("Fault", "KnowledgeSource"): {
                "COMES_FROM": ("fault_id", "source_id", "comes_from")
            },
            ("FaultPhenomenon", "KnowledgeSource"): {
                "COMES_FROM": ("phenomenon_id", "source_id", "comes_from")
            },
            ("FaultReason", "KnowledgeSource"): {
                "COMES_FROM": ("cause_id", "source_id", "comes_from")
            },
            ("MaintenanceAction", "KnowledgeSource"): {
                "COMES_FROM": ("action_id", "source_id", "comes_from")
            },
            ("SafetyNotice", "KnowledgeSource"): {
                "COMES_FROM": ("notice_id", "source_id", "comes_from")
            },
            ("EquipmentCategory", "KnowledgeSource"): {
                "COMES_FROM": ("category_id", "source_id", "comes_from")
            },
        }
        
        with self.driver.session() as session:
            for rel in relations:
                from_entity = rel['from_entity']
                to_entity = rel['to_entity']
                relation_type = rel['relation_type']
                
                key = (from_entity, to_entity)
                if key not in relation_mapping:
                    continue
                
                rel_info = relation_mapping[key].get(relation_type)
                if not rel_info:
                    continue
                
                from_id_field, to_id_field, rel_name = rel_info
                
                query = (
                    f"MATCH (from:{from_entity} {{{from_id_field}: $from_id}})-[r:{rel_name}]->"
                    f"(to:{to_entity} {{{to_id_field}: $to_id}}) "
                    f"RETURN count(r) as count"
                )
                
                result = session.run(query, from_id=rel['from_id'], to_id=rel['to_id'])
                record = result.single()
                
                if record and record["count"] > 0:
                    existing_relations.add((rel['from_id'], relation_type, rel['to_id']))
        
        return existing_relations

    def import_equipment_categories(self) -> int:
        """导入装备大类节点"""
        csv_path = os.path.join(self.csv_dir, "equipmentcategorys.csv")
        if not os.path.exists(csv_path):
            logger.warning(f"文件不存在: {csv_path}")
            return 0

        df, _ = self.deduplicate_csv(csv_path, "category_id")
        if df.empty:
            return 0
        logger.info(f"导入 {len(df)} 个EquipmentCategory节点...")

        count = 0
        try:
            with self.driver.session() as session:
                for _, row in df.iterrows():
                    try:
                        query = f"""
                        MERGE (n:EquipmentCategory {{category_id: '{self._escape_string(row['category_id'])}'}})
                        SET n.name = '{self._escape_string(row['name'])}',
                            n.description = '{self._escape_string(row.get('description', ''))}',
                            n.source_id = '{self._escape_string(row.get('source_id', ''))}'
                        """
                        session.run(query)
                        count += 1
                    except Exception as e:
                        logger.error(f"导入EquipmentCategory失败: {row['category_id']}, 错误: {e}")
        except KeyboardInterrupt:
            logger.info(f"导入中断，已导入 {count} 个EquipmentCategory节点")
            return count
        except Exception as e:
            logger.error(f"导入EquipmentCategory节点时发生错误: {e}")
            return count

        logger.info(f"成功导入 {count} 个EquipmentCategory节点")
        return count

    def import_equipments(self) -> int:
        """导入设备节点"""
        csv_path = os.path.join(self.csv_dir, "equipments.csv")
        if not os.path.exists(csv_path):
            logger.warning(f"文件不存在: {csv_path}")
            return 0

        df, _ = self.deduplicate_csv(csv_path, "equipment_id")
        if df.empty:
            return 0
        logger.info(f"导入 {len(df)} 个Equipment节点...")

        count = 0
        try:
            with self.driver.session() as session:
                for _, row in df.iterrows():
                    try:
                        query = f"""
                        MERGE (n:Equipment {{equipment_id: '{self._escape_string(row['equipment_id'])}'}})
                        SET n.name = '{self._escape_string(row['name'])}',
                            n.type = '{self._escape_string(row.get('type', ''))}',
                            n.model = '{self._escape_string(row.get('model', ''))}',
                            n.source_id = '{self._escape_string(row.get('source_id', ''))}'
                        """
                        session.run(query)
                        count += 1
                    except Exception as e:
                        logger.error(f"导入Equipment失败: {row['equipment_id']}, 错误: {e}")
        except KeyboardInterrupt:
            logger.info(f"导入中断，已导入 {count} 个Equipment节点")
            return count
        except Exception as e:
            logger.error(f"导入Equipment节点时发生错误: {e}")
            return count

        logger.info(f"成功导入 {count} 个Equipment节点")
        return count

    def import_components(self) -> int:
        """导入部件节点"""
        csv_path = os.path.join(self.csv_dir, "components.csv")
        if not os.path.exists(csv_path):
            logger.warning(f"文件不存在: {csv_path}")
            return 0

        df, _ = self.deduplicate_csv(csv_path, "component_id")
        if df.empty:
            return 0
        logger.info(f"导入 {len(df)} 个Component节点...")

        count = 0
        try:
            with self.driver.session() as session:
                for _, row in df.iterrows():
                    try:
                        query = f"""
                        MERGE (n:Component {{component_id: '{self._escape_string(row['component_id'])}'}})
                        SET n.name = '{self._escape_string(row['name'])}',
                            n.spec = '{self._escape_string(row.get('spec', ''))}',
                            n.type = '{self._escape_string(row.get('type', ''))}',
                            n.source_id = '{self._escape_string(row.get('source_id', ''))}'
                        """
                        session.run(query)
                        count += 1
                    except Exception as e:
                        logger.error(f"导入Component失败: {row['component_id']}, 错误: {e}")
        except KeyboardInterrupt:
            logger.info(f"导入中断，已导入 {count} 个Component节点")
            return count
        except Exception as e:
            logger.error(f"导入Component节点时发生错误: {e}")
            return count

        logger.info(f"成功导入 {count} 个Component节点")
        return count

    def import_faults(self) -> int:
        """导入故障节点"""
        csv_path = os.path.join(self.csv_dir, "faults.csv")
        if not os.path.exists(csv_path):
            logger.warning(f"文件不存在: {csv_path}")
            return 0

        df, _ = self.deduplicate_csv(csv_path, "fault_id")
        if df.empty:
            return 0
        logger.info(f"导入 {len(df)} 个Fault节点...")

        count = 0
        try:
            with self.driver.session() as session:
                for _, row in df.iterrows():
                    try:
                        query = f"""
                        MERGE (n:Fault {{fault_id: '{self._escape_string(row['fault_id'])}'}})
                        SET n.name = '{self._escape_string(row['name'])}',
                            n.fault_type = '{self._escape_string(row.get('fault_type', ''))}',
                            n.severity = '{self._escape_string(row.get('severity', ''))}',
                            n.occurrence_frequency = '{self._escape_string(row.get('occurrence_frequency', ''))}',
                            n.description = '{self._escape_string(row.get('description', ''))}',
                            n.source_id = '{self._escape_string(row.get('source_id', ''))}'
                        """
                        session.run(query)
                        count += 1
                    except Exception as e:
                        logger.error(f"导入Fault失败: {row['fault_id']}, 错误: {e}")
        except KeyboardInterrupt:
            logger.info(f"导入中断，已导入 {count} 个Fault节点")
            return count
        except Exception as e:
            logger.error(f"导入Fault节点时发生错误: {e}")
            return count

        logger.info(f"成功导入 {count} 个Fault节点")
        return count

    def import_fault_phenomenons(self) -> int:
        """导入故障现象节点"""
        csv_path = os.path.join(self.csv_dir, "faultphenomenons.csv")
        if not os.path.exists(csv_path):
            logger.warning(f"文件不存在: {csv_path}")
            return 0

        df, _ = self.deduplicate_csv(csv_path, "phenomenon_id")
        if df.empty:
            return 0
        logger.info(f"导入 {len(df)} 个FaultPhenomenon节点...")

        count = 0
        try:
            with self.driver.session() as session:
                for _, row in df.iterrows():
                    try:
                        query = f"""
                        MERGE (n:FaultPhenomenon {{phenomenon_id: '{self._escape_string(row['phenomenon_id'])}'}})
                        SET n.description = '{self._escape_string(row.get('description', ''))}',
                            n.source_id = '{self._escape_string(row.get('source_id', ''))}'
                        """
                        session.run(query)
                        count += 1
                    except Exception as e:
                        logger.error(f"导入FaultPhenomenon失败: {row['phenomenon_id']}, 错误: {e}")
        except KeyboardInterrupt:
            logger.info(f"导入中断，已导入 {count} 个FaultPhenomenon节点")
            return count
        except Exception as e:
            logger.error(f"导入FaultPhenomenon节点时发生错误: {e}")
            return count

        logger.info(f"成功导入 {count} 个FaultPhenomenon节点")
        return count

    def import_fault_reasons(self) -> int:
        """导入故障原因节点"""
        csv_path = os.path.join(self.csv_dir, "faultreasons.csv")
        if not os.path.exists(csv_path):
            logger.warning(f"文件不存在: {csv_path}")
            return 0

        df, _ = self.deduplicate_csv(csv_path, "cause_id")
        if df.empty:
            return 0
        logger.info(f"导入 {len(df)} 个FaultReason节点...")

        count = 0
        try:
            with self.driver.session() as session:
                for _, row in df.iterrows():
                    try:
                        level = row.get('level', '')
                        level_str = str(int(level)) if pd.notna(level) and level != '' else ''
                        query = f"""
                        MERGE (n:FaultReason {{cause_id: '{self._escape_string(row['cause_id'])}'}})
                        SET n.cause_name = '{self._escape_string(row['cause_name'])}',
                            n.description = '{self._escape_string(row.get('description', ''))}',
                            n.category = '{self._escape_string(row.get('category', ''))}',
                            n.level = '{level_str}',
                            n.source_id = '{self._escape_string(row.get('source_id', ''))}'
                        """
                        session.run(query)
                        count += 1
                    except Exception as e:
                        logger.error(f"导入FaultReason失败: {row['cause_id']}, 错误: {e}")
        except KeyboardInterrupt:
            logger.info(f"导入中断，已导入 {count} 个FaultReason节点")
            return count
        except Exception as e:
            logger.error(f"导入FaultReason节点时发生错误: {e}")
            return count

        logger.info(f"成功导入 {count} 个FaultReason节点")
        return count

    def import_maintenance_actions(self) -> int:
        """导入维修步骤节点"""
        csv_path = os.path.join(self.csv_dir, "maintenanceactions.csv")
        if not os.path.exists(csv_path):
            logger.warning(f"文件不存在: {csv_path}")
            return 0

        df, _ = self.deduplicate_csv(csv_path, "action_id")
        if df.empty:
            return 0
        logger.info(f"导入 {len(df)} 个MaintenanceAction节点...")

        count = 0
        try:
            with self.driver.session() as session:
                for _, row in df.iterrows():
                    try:
                        step_order = row.get('step_order', '')
                        step_order_str = str(int(step_order)) if pd.notna(step_order) and step_order != '' else ''
                        query = f"""
                        MERGE (n:MaintenanceAction {{action_id: '{self._escape_string(row['action_id'])}'}})
                        SET n.step_order = '{step_order_str}',
                            n.description = '{self._escape_string(row.get('description', ''))}',
                            n.estimated_time = '{self._escape_string(row.get('estimated_time', ''))}',
                            n.tools = '{self._escape_string(row.get('tools', ''))}',
                            n.source_id = '{self._escape_string(row.get('source_id', ''))}'
                        """
                        session.run(query)
                        count += 1
                    except Exception as e:
                        logger.error(f"导入MaintenanceAction失败: {row['action_id']}, 错误: {e}")
        except KeyboardInterrupt:
            logger.info(f"导入中断，已导入 {count} 个MaintenanceAction节点")
            return count
        except Exception as e:
            logger.error(f"导入MaintenanceAction节点时发生错误: {e}")
            return count

        logger.info(f"成功导入 {count} 个MaintenanceAction节点")
        return count

    def import_safety_notices(self) -> int:
        """导入注意事项节点"""
        csv_path = os.path.join(self.csv_dir, "safetynotices.csv")
        if not os.path.exists(csv_path):
            logger.warning(f"文件不存在: {csv_path}")
            return 0

        df, _ = self.deduplicate_csv(csv_path, "notice_id")
        if df.empty:
            return 0
        logger.info(f"导入 {len(df)} 个SafetyNotice节点...")

        count = 0
        try:
            with self.driver.session() as session:
                for _, row in df.iterrows():
                    try:
                        query = f"""
                        MERGE (n:SafetyNotice {{notice_id: '{self._escape_string(row['notice_id'])}'}})
                        SET n.level = '{self._escape_string(row.get('level', ''))}',
                            n.description = '{self._escape_string(row.get('description', ''))}',
                            n.consequence = '{self._escape_string(row.get('consequence', ''))}',
                            n.source_id = '{self._escape_string(row.get('source_id', ''))}'
                        """
                        session.run(query)
                        count += 1
                    except Exception as e:
                        logger.error(f"导入SafetyNotice失败: {row['notice_id']}, 错误: {e}")
        except KeyboardInterrupt:
            logger.info(f"导入中断，已导入 {count} 个SafetyNotice节点")
            return count
        except Exception as e:
            logger.error(f"导入SafetyNotice节点时发生错误: {e}")
            return count

        logger.info(f"成功导入 {count} 个SafetyNotice节点")
        return count

    def import_knowledge_sources(self) -> int:
        """导入知识来源节点"""
        csv_path = os.path.join(self.csv_dir, "knowledgesources.csv")
        if not os.path.exists(csv_path):
            logger.warning(f"文件不存在: {csv_path}")
            return 0

        df, _ = self.deduplicate_csv(csv_path, "source_id")
        if df.empty:
            return 0
        logger.info(f"导入 {len(df)} 个KnowledgeSource节点...")

        count = 0
        try:
            with self.driver.session() as session:
                for _, row in df.iterrows():
                    try:
                        query = f"""
                        MERGE (n:KnowledgeSource {{source_id: '{self._escape_string(row['source_id'])}'}})
                        SET n.name = '{self._escape_string(row.get('name', ''))}',
                            n.type = '{self._escape_string(row.get('type', ''))}',
                            n.chapter = '{self._escape_string(row.get('chapter', ''))}',
                            n.section = '{self._escape_string(row.get('section', ''))}',
                            n.reliability = '{self._escape_string(row.get('reliability', ''))}'
                        """
                        session.run(query)
                        count += 1
                    except Exception as e:
                        logger.error(f"导入KnowledgeSource失败: {row['source_id']}, 错误: {e}")
        except KeyboardInterrupt:
            logger.info(f"导入中断，已导入 {count} 个KnowledgeSource节点")
            return count
        except Exception as e:
            logger.error(f"导入KnowledgeSource节点时发生错误: {e}")
            return count

        logger.info(f"成功导入 {count} 个KnowledgeSource节点")
        return count

    def import_relations(self) -> int:
        """导入关系"""
        csv_path = os.path.join(self.csv_dir, "relationships.csv")
        if not os.path.exists(csv_path):
            logger.warning(f"文件不存在: {csv_path}")
            return 0

        df, _ = self.deduplicate_relations_csv(csv_path)
        if df.empty:
            return 0
        logger.info(f"导入 {len(df)} 个关系...")

        relation_mapping = {
            ("EquipmentCategory", "Equipment"): {
                "CONTAINS": ("category_id", "equipment_id", "contains")
            },
            ("EquipmentCategory", "EquipmentCategory"): {
                "CONTAINS": ("category_id", "category_id", "contains")
            },
            ("EquipmentCategory", "Component"): {
                "CONTAINS": ("category_id", "component_id", "contains")
            },
            ("Equipment", "Component"): {
                "CONSISTS_OF": ("equipment_id", "component_id", "consists_of")
            },
            ("Equipment", "Equipment"): {
                "CONSISTS_OF": ("equipment_id", "equipment_id", "consists_of")
            },
            ("Equipment", "Fault"): {
                "HAS_FAULT": ("equipment_id", "fault_id", "has_fault")
            },
            ("Component", "Fault"): {
                "HAS_FAULT": ("component_id", "fault_id", "has_fault")
            },
            ("Fault", "FaultPhenomenon"): {
                "PRESENTS_AS": ("fault_id", "phenomenon_id", "presents_as")
            },
            ("Fault", "FaultReason"): {
                "CAUSED_BY": ("fault_id", "cause_id", "caused_by")
            },
            ("FaultReason", "FaultReason"): {
                "CAUSED_BY": ("cause_id", "cause_id", "caused_by")
            },
            ("FaultReason", "Component"): {
                "RELATES_TO": ("cause_id", "component_id", "relates_to")
            },
            ("FaultReason", "Equipment"): {
                "RELATES_TO": ("cause_id", "equipment_id", "relates_to")
            },
            ("Component", "Component"): {
                "CONSISTS_OF": ("component_id", "component_id", "consists_of")
            },
            ("FaultReason", "MaintenanceAction"): {
                "FIXED_BY": ("cause_id", "action_id", "fixed_by")
            },
            ("MaintenanceAction", "SafetyNotice"): {
                "HAS_NOTICE": ("action_id", "notice_id", "has_notice")
            },
            ("Equipment", "KnowledgeSource"): {
                "COMES_FROM": ("equipment_id", "source_id", "comes_from")
            },
            ("Component", "KnowledgeSource"): {
                "COMES_FROM": ("component_id", "source_id", "comes_from")
            },
            ("Fault", "KnowledgeSource"): {
                "COMES_FROM": ("fault_id", "source_id", "comes_from")
            },
            ("FaultPhenomenon", "KnowledgeSource"): {
                "COMES_FROM": ("phenomenon_id", "source_id", "comes_from")
            },
            ("FaultReason", "KnowledgeSource"): {
                "COMES_FROM": ("cause_id", "source_id", "comes_from")
            },
            ("MaintenanceAction", "KnowledgeSource"): {
                "COMES_FROM": ("action_id", "source_id", "comes_from")
            },
            ("SafetyNotice", "KnowledgeSource"): {
                "COMES_FROM": ("notice_id", "source_id", "comes_from")
            },
            ("EquipmentCategory", "KnowledgeSource"): {
                "COMES_FROM": ("category_id", "source_id", "comes_from")
            },
        }

        count = 0
        try:
            with self.driver.session() as session:
                for _, row in df.iterrows():
                    try:
                        from_entity = row['from_entity']
                        from_id = row['from_id']
                        relation_type = row['relation_type']
                        to_entity = row['to_entity']
                        to_id = row['to_id']

                        key = (from_entity, to_entity)
                        if key not in relation_mapping:
                            logger.warning(f"未知的关系类型: {from_entity} -> {to_entity}")
                            continue

                        rel_info = relation_mapping[key].get(relation_type)
                        if not rel_info:
                            logger.warning(f"未知的关系类型: {relation_type}")
                            continue

                        from_id_field, to_id_field, rel_name = rel_info

                        query = f"""
                        MATCH (from:{from_entity} {{{from_id_field}: '{self._escape_string(from_id)}'}})
                        MATCH (to:{to_entity} {{{to_id_field}: '{self._escape_string(to_id)}'}})
                        MERGE (from)-[:{rel_name}]->(to)
                        """

                        session.run(query)
                        count += 1
                    except Exception as e:
                        logger.error(f"导入关系失败: {row.get('from_id', '')} -> {row.get('to_id', '')}, 错误: {e}")
        except KeyboardInterrupt:
            logger.info(f"导入中断，已导入 {count} 个关系")
            return count
        except Exception as e:
            logger.error(f"导入关系时发生错误: {e}")
            return count

        logger.info(f"成功导入 {count} 个关系")
        return count

    def backfill_comes_from_relations(self) -> int:
        """按节点 source_id 回填实体到 KnowledgeSource 的 COMES_FROM 关系"""
        logger.info("开始按 source_id 回填 COMES_FROM 关系...")

        entity_mappings = [
            ("EquipmentCategory", "category_id"),
            ("Equipment", "equipment_id"),
            ("Component", "component_id"),
            ("Fault", "fault_id"),
            ("FaultPhenomenon", "phenomenon_id"),
            ("FaultReason", "cause_id"),
            ("MaintenanceAction", "action_id"),
            ("SafetyNotice", "notice_id"),
        ]

        total_created = 0
        with self.driver.session() as session:
            for label, id_field in entity_mappings:
                try:
                    query = f"""
                    MATCH (n:{label})
                    WHERE n.source_id IS NOT NULL AND trim(toString(n.source_id)) <> ''
                    MATCH (ks:KnowledgeSource {{source_id: n.source_id}})
                    MERGE (n)-[r:comes_from]->(ks)
                    RETURN count(r) as rel_count
                    """
                    record = session.run(query).single()
                    created = int(record["rel_count"]) if record and record.get("rel_count") is not None else 0
                    total_created += created
                    logger.info(f"{label} 回填 COMES_FROM 数量: {created}")
                except Exception as e:
                    logger.warning(f"{label} 回填 COMES_FROM 失败: {e}")

        logger.info(f"COMES_FROM 回填完成，总计: {total_created}")
        return total_created

    def import_all(self, clear_db: bool = False):
        """
        导入所有数据

        Args:
            clear_db: 是否在导入前清空数据库
        """
        logger.info("=" * 50)
        logger.info("开始导入CSV数据到Neo4j")
        logger.info("=" * 50)

        if not self.connect():
            logger.error("无法连接到Neo4j，导入终止")
            return

        try:
            if clear_db:
                self.clear_database()

            self.create_indexes()

            total_nodes = 0
            total_nodes += self.import_knowledge_sources()
            total_nodes += self.import_equipment_categories()
            total_nodes += self.import_equipments()
            total_nodes += self.import_components()
            total_nodes += self.import_faults()
            total_nodes += self.import_fault_phenomenons()
            total_nodes += self.import_fault_reasons()
            total_nodes += self.import_maintenance_actions()
            total_nodes += self.import_safety_notices()

            total_relations = self.import_relations()
            total_relations += self.backfill_comes_from_relations()

            logger.info("=" * 50)
            logger.info(f"导入完成！")
            logger.info(f"总节点数: {total_nodes}")
            logger.info(f"总关系数: {total_relations}")
            if self.dedup_stats["nodes_deduplicated"] > 0 or self.dedup_stats["relations_deduplicated"] > 0:
                logger.info(f"去重统计:")
                logger.info(f"  节点去重: {self.dedup_stats['nodes_deduplicated']} 条")
                logger.info(f"  关系去重: {self.dedup_stats['relations_deduplicated']} 条")
            logger.info("=" * 50)

        finally:
            self.close()


def main():
    """主函数"""
    from config import DEFAULT_CONFIG

    importer = CSVToNeo4jImporter(
        uri=DEFAULT_CONFIG.neo4j_uri,
        user=DEFAULT_CONFIG.neo4j_user,
        password=DEFAULT_CONFIG.neo4j_password,
        database=DEFAULT_CONFIG.neo4j_database
    )

    print("\n" + "=" * 50)
    print("CSV导入Neo4j工具")
    print("=" * 50)
    print(f"CSV目录: ./generate_csv")
    print(f"Neo4j URI: {DEFAULT_CONFIG.neo4j_uri}")
    print(f"数据库: {DEFAULT_CONFIG.neo4j_database}")
    print("=" * 50)
    #
    # confirm = input("\n是否清空现有数据库后导入？(y/N): ").strip().lower()
    confirm = 'y'
    clear_db = confirm == 'y'

    importer.import_all(clear_db=clear_db)


if __name__ == "__main__":
    main()
