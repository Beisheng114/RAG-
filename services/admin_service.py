import os
import csv
import zipfile
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, List, Tuple

from fastapi import HTTPException, Header, UploadFile
from neo4j import GraphDatabase

from config import DEFAULT_CONFIG
from core.system_context import get_rag_system
from core.security import (
    get_admin_token,
    is_admin_enabled,
    require_admin,
    mask_secret,
    is_sensitive_config_key,
)
from csv_to_neo4j import CSVToNeo4jImporter


def get_safe_config_preview() -> Dict[str, Any]:
    cfg = DEFAULT_CONFIG.to_dict()
    safe_cfg: Dict[str, Any] = {}
    for k, v in cfg.items():
        if is_sensitive_config_key(k):
            safe_cfg[k] = mask_secret(str(v))
        else:
            safe_cfg[k] = v

    safe_cfg["deepseek_api_key"] = mask_secret(os.getenv("DEEPSEEK_API_KEY", ""))
    admin_token = get_admin_token()
    safe_cfg["rag_admin_token_hint"] = mask_secret(admin_token or "")
    safe_cfg["admin_enabled"] = is_admin_enabled()
    return safe_cfg


def _safe_mtime(path: Optional[Path]) -> str:
    if not path or not path.exists():
        return ""
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")


def _neo4j_driver():
    return GraphDatabase.driver(
        DEFAULT_CONFIG.neo4j_uri,
        auth=(DEFAULT_CONFIG.neo4j_user, DEFAULT_CONFIG.neo4j_password),
    )


def _is_neo4j_database_param_error(err: Exception) -> bool:
    msg = str(err)
    return ("Database name parameter" in msg) or ("Bolt Protocol Version(3, 0)" in msg)


def _run_query_with_compat_session(driver, query: str):
    """
    兼容执行查询：优先使用 database 参数；若不支持则回退默认库。
    """
    try:
        with driver.session(database=DEFAULT_CONFIG.neo4j_database) as session:
            return list(session.run(query))
    except Exception as e:
        if not _is_neo4j_database_param_error(e):
            raise
        with driver.session() as session:
            return list(session.run(query))


def _neo4j_export_rules() -> List[Tuple[str, List[str], str]]:
    return [
        (
            "equipmentcategorys.csv",
            ["category_id", "name", "description", "source_id"],
            """
            MATCH (n:EquipmentCategory)
            RETURN
                coalesce(n.category_id, '') AS category_id,
                coalesce(n.name, '') AS name,
                coalesce(n.description, '') AS description,
                coalesce(n.source_id, '') AS source_id
            """,
        ),
        (
            "equipments.csv",
            ["equipment_id", "name", "type", "model", "source_id"],
            """
            MATCH (n:Equipment)
            RETURN
                coalesce(n.equipment_id, '') AS equipment_id,
                coalesce(n.name, '') AS name,
                coalesce(n.type, '') AS type,
                coalesce(n.model, '') AS model,
                coalesce(n.source_id, '') AS source_id
            """,
        ),
        (
            "components.csv",
            ["component_id", "name", "spec", "type", "source_id"],
            """
            MATCH (n:Component)
            RETURN
                coalesce(n.component_id, '') AS component_id,
                coalesce(n.name, '') AS name,
                coalesce(n.spec, '') AS spec,
                coalesce(n.type, '') AS type,
                coalesce(n.source_id, '') AS source_id
            """,
        ),
        (
            "faults.csv",
            ["fault_id", "name", "fault_type", "severity", "occurrence_frequency", "description", "source_id"],
            """
            MATCH (n:Fault)
            RETURN
                coalesce(n.fault_id, '') AS fault_id,
                coalesce(n.name, '') AS name,
                coalesce(n.fault_type, '') AS fault_type,
                coalesce(n.severity, '') AS severity,
                coalesce(n.occurrence_frequency, '') AS occurrence_frequency,
                coalesce(n.description, '') AS description,
                coalesce(n.source_id, '') AS source_id
            """,
        ),
        (
            "faultphenomenons.csv",
            ["phenomenon_id", "description", "source_id"],
            """
            MATCH (n:FaultPhenomenon)
            RETURN
                coalesce(n.phenomenon_id, '') AS phenomenon_id,
                coalesce(n.description, '') AS description,
                coalesce(n.source_id, '') AS source_id
            """,
        ),
        (
            "faultreasons.csv",
            ["cause_id", "cause_name", "description", "category", "level", "source_id"],
            """
            MATCH (n:FaultReason)
            RETURN
                coalesce(n.cause_id, '') AS cause_id,
                coalesce(n.cause_name, '') AS cause_name,
                coalesce(n.description, '') AS description,
                coalesce(n.category, '') AS category,
                coalesce(n.level, '') AS level,
                coalesce(n.source_id, '') AS source_id
            """,
        ),
        (
            "maintenanceactions.csv",
            ["action_id", "step_order", "description", "estimated_time", "tools", "source_id"],
            """
            MATCH (n:MaintenanceAction)
            RETURN
                coalesce(n.action_id, '') AS action_id,
                coalesce(n.step_order, '') AS step_order,
                coalesce(n.description, '') AS description,
                coalesce(n.estimated_time, '') AS estimated_time,
                coalesce(n.tools, '') AS tools,
                coalesce(n.source_id, '') AS source_id
            """,
        ),
        (
            "safetynotices.csv",
            ["notice_id", "level", "description", "consequence", "source_id"],
            """
            MATCH (n:SafetyNotice)
            RETURN
                coalesce(n.notice_id, '') AS notice_id,
                coalesce(n.level, '') AS level,
                coalesce(n.description, '') AS description,
                coalesce(n.consequence, '') AS consequence,
                coalesce(n.source_id, '') AS source_id
            """,
        ),
        (
            "knowledgesources.csv",
            ["source_id", "name", "type", "chapter", "section", "reliability"],
            """
            MATCH (n:KnowledgeSource)
            RETURN
                coalesce(n.source_id, '') AS source_id,
                coalesce(n.name, '') AS name,
                coalesce(n.type, '') AS type,
                coalesce(n.chapter, '') AS chapter,
                coalesce(n.section, '') AS section,
                coalesce(n.reliability, '') AS reliability
            """,
        ),
    ]


def _neo4j_relationship_export_queries() -> List[str]:
    return [
        "MATCH (a:EquipmentCategory)-[:contains]->(b:Equipment) RETURN 'EquipmentCategory' AS from_entity, coalesce(a.category_id,'') AS from_id, 'CONTAINS' AS relation_type, 'Equipment' AS to_entity, coalesce(b.equipment_id,'') AS to_id",
        "MATCH (a:EquipmentCategory)-[:contains]->(b:EquipmentCategory) RETURN 'EquipmentCategory' AS from_entity, coalesce(a.category_id,'') AS from_id, 'CONTAINS' AS relation_type, 'EquipmentCategory' AS to_entity, coalesce(b.category_id,'') AS to_id",
        "MATCH (a:EquipmentCategory)-[:contains]->(b:Component) RETURN 'EquipmentCategory' AS from_entity, coalesce(a.category_id,'') AS from_id, 'CONTAINS' AS relation_type, 'Component' AS to_entity, coalesce(b.component_id,'') AS to_id",
        "MATCH (a:Equipment)-[:consists_of]->(b:Component) RETURN 'Equipment' AS from_entity, coalesce(a.equipment_id,'') AS from_id, 'CONSISTS_OF' AS relation_type, 'Component' AS to_entity, coalesce(b.component_id,'') AS to_id",
        "MATCH (a:Equipment)-[:consists_of]->(b:Equipment) RETURN 'Equipment' AS from_entity, coalesce(a.equipment_id,'') AS from_id, 'CONSISTS_OF' AS relation_type, 'Equipment' AS to_entity, coalesce(b.equipment_id,'') AS to_id",
        "MATCH (a:Equipment)-[:contains]->(b:Equipment) RETURN 'Equipment' AS from_entity, coalesce(a.equipment_id,'') AS from_id, 'CONSISTS_OF' AS relation_type, 'Equipment' AS to_entity, coalesce(b.equipment_id,'') AS to_id",
        "MATCH (a:Equipment)-[:has_fault]->(b:Fault) RETURN 'Equipment' AS from_entity, coalesce(a.equipment_id,'') AS from_id, 'HAS_FAULT' AS relation_type, 'Fault' AS to_entity, coalesce(b.fault_id,'') AS to_id",
        "MATCH (a:Component)-[:has_fault]->(b:Fault) RETURN 'Component' AS from_entity, coalesce(a.component_id,'') AS from_id, 'HAS_FAULT' AS relation_type, 'Fault' AS to_entity, coalesce(b.fault_id,'') AS to_id",
        "MATCH (a:Fault)-[:presents_as]->(b:FaultPhenomenon) RETURN 'Fault' AS from_entity, coalesce(a.fault_id,'') AS from_id, 'PRESENTS_AS' AS relation_type, 'FaultPhenomenon' AS to_entity, coalesce(b.phenomenon_id,'') AS to_id",
        "MATCH (a:Fault)-[:caused_by]->(b:FaultReason) RETURN 'Fault' AS from_entity, coalesce(a.fault_id,'') AS from_id, 'CAUSED_BY' AS relation_type, 'FaultReason' AS to_entity, coalesce(b.cause_id,'') AS to_id",
        "MATCH (a:FaultReason)-[:caused_by]->(b:FaultReason) RETURN 'FaultReason' AS from_entity, coalesce(a.cause_id,'') AS from_id, 'CAUSED_BY' AS relation_type, 'FaultReason' AS to_entity, coalesce(b.cause_id,'') AS to_id",
        "MATCH (a:FaultReason)-[:relates_to]->(b:Component) RETURN 'FaultReason' AS from_entity, coalesce(a.cause_id,'') AS from_id, 'RELATES_TO' AS relation_type, 'Component' AS to_entity, coalesce(b.component_id,'') AS to_id",
        "MATCH (a:FaultReason)-[:relates_to]->(b:Equipment) RETURN 'FaultReason' AS from_entity, coalesce(a.cause_id,'') AS from_id, 'RELATES_TO' AS relation_type, 'Equipment' AS to_entity, coalesce(b.equipment_id,'') AS to_id",
        "MATCH (a:Component)-[:consists_of]->(b:Component) RETURN 'Component' AS from_entity, coalesce(a.component_id,'') AS from_id, 'CONSISTS_OF' AS relation_type, 'Component' AS to_entity, coalesce(b.component_id,'') AS to_id",
        "MATCH (a:FaultReason)-[:fixed_by]->(b:MaintenanceAction) RETURN 'FaultReason' AS from_entity, coalesce(a.cause_id,'') AS from_id, 'FIXED_BY' AS relation_type, 'MaintenanceAction' AS to_entity, coalesce(b.action_id,'') AS to_id",
        "MATCH (a:MaintenanceAction)-[:has_notice]->(b:SafetyNotice) RETURN 'MaintenanceAction' AS from_entity, coalesce(a.action_id,'') AS from_id, 'HAS_NOTICE' AS relation_type, 'SafetyNotice' AS to_entity, coalesce(b.notice_id,'') AS to_id",
        "MATCH (a:Equipment)-[:comes_from]->(b:KnowledgeSource) RETURN 'Equipment' AS from_entity, coalesce(a.equipment_id,'') AS from_id, 'COMES_FROM' AS relation_type, 'KnowledgeSource' AS to_entity, coalesce(b.source_id,'') AS to_id",
        "MATCH (a:Component)-[:comes_from]->(b:KnowledgeSource) RETURN 'Component' AS from_entity, coalesce(a.component_id,'') AS from_id, 'COMES_FROM' AS relation_type, 'KnowledgeSource' AS to_entity, coalesce(b.source_id,'') AS to_id",
        "MATCH (a:Fault)-[:comes_from]->(b:KnowledgeSource) RETURN 'Fault' AS from_entity, coalesce(a.fault_id,'') AS from_id, 'COMES_FROM' AS relation_type, 'KnowledgeSource' AS to_entity, coalesce(b.source_id,'') AS to_id",
        "MATCH (a:FaultPhenomenon)-[:comes_from]->(b:KnowledgeSource) RETURN 'FaultPhenomenon' AS from_entity, coalesce(a.phenomenon_id,'') AS from_id, 'COMES_FROM' AS relation_type, 'KnowledgeSource' AS to_entity, coalesce(b.source_id,'') AS to_id",
        "MATCH (a:FaultReason)-[:comes_from]->(b:KnowledgeSource) RETURN 'FaultReason' AS from_entity, coalesce(a.cause_id,'') AS from_id, 'COMES_FROM' AS relation_type, 'KnowledgeSource' AS to_entity, coalesce(b.source_id,'') AS to_id",
        "MATCH (a:MaintenanceAction)-[:comes_from]->(b:KnowledgeSource) RETURN 'MaintenanceAction' AS from_entity, coalesce(a.action_id,'') AS from_id, 'COMES_FROM' AS relation_type, 'KnowledgeSource' AS to_entity, coalesce(b.source_id,'') AS to_id",
        "MATCH (a:SafetyNotice)-[:comes_from]->(b:KnowledgeSource) RETURN 'SafetyNotice' AS from_entity, coalesce(a.notice_id,'') AS from_id, 'COMES_FROM' AS relation_type, 'KnowledgeSource' AS to_entity, coalesce(b.source_id,'') AS to_id",
        "MATCH (a:EquipmentCategory)-[:comes_from]->(b:KnowledgeSource) RETURN 'EquipmentCategory' AS from_entity, coalesce(a.category_id,'') AS from_id, 'COMES_FROM' AS relation_type, 'KnowledgeSource' AS to_entity, coalesce(b.source_id,'') AS to_id",
    ]


def _write_csv(path: Path, headers: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            normalized = {k: ("" if row.get(k) is None else str(row.get(k))) for k in headers}
            writer.writerow(normalized)


def _export_neo4j_to_csv(export_dir: Path) -> Dict[str, int]:
    export_dir.mkdir(parents=True, exist_ok=True)
    stats: Dict[str, int] = {}

    driver = _neo4j_driver()
    try:
        for filename, headers, query in _neo4j_export_rules():
            records = _run_query_with_compat_session(driver, query)
            rows = [dict(r) for r in records]
            _write_csv(export_dir / filename, headers, rows)
            stats[filename] = len(rows)

        rel_rows: List[Dict[str, Any]] = []
        for rel_query in _neo4j_relationship_export_queries():
            rel_rows.extend([dict(r) for r in _run_query_with_compat_session(driver, rel_query)])

        _write_csv(
            export_dir / "relationships.csv",
            ["from_entity", "from_id", "relation_type", "to_entity", "to_id"],
            rel_rows,
        )
        stats["relationships.csv"] = len(rel_rows)
    finally:
        driver.close()

    return stats


def _find_csv_root(extracted_root: Path) -> Optional[Path]:
    required = {
        "equipmentcategorys.csv",
        "equipments.csv",
        "components.csv",
        "faults.csv",
        "faultphenomenons.csv",
        "faultreasons.csv",
        "maintenanceactions.csv",
        "safetynotices.csv",
        "knowledgesources.csv",
        "relationships.csv",
    }

    candidates = [extracted_root]
    candidates.extend([p for p in extracted_root.rglob("*") if p.is_dir()])

    for d in candidates:
        existing = {p.name.lower() for p in d.glob("*.csv")}
        if required.issubset(existing):
            return d
    return None


def get_knowledge_base_version() -> Dict[str, Any]:
    backup_root = Path("backups")
    latest_backup = None
    if backup_root.exists():
        backups = sorted(backup_root.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
        if backups:
            latest_backup = backups[0]

    conversations_dir = Path("conversations")
    config_file = Path("config.py")

    rag_system = get_rag_system()
    node_counts: Dict[str, int] = {}
    total_entities = 0
    try:
        if rag_system:
            node_counts = rag_system.get_node_type_counts() or {}
            total_entities = int(sum(node_counts.values()))
    except Exception:
        node_counts = {}
        total_entities = 0

    return {
        "version": datetime.now().strftime("v%Y.%m.%d"),
        "vector_index_type": getattr(DEFAULT_CONFIG, "vector_index_type", "unknown"),
        "qdrant_collection": getattr(DEFAULT_CONFIG, "qdrant_collection_name", ""),
        "last_backup_file": str(latest_backup).replace("\\", "/") if latest_backup else "",
        "last_backup_time": _safe_mtime(latest_backup) if latest_backup else "",
        "conversations_last_update": _safe_mtime(conversations_dir),
        "config_last_update": _safe_mtime(config_file),
        "total_entities": total_entities,
        "node_type_count": len(node_counts),
    }


def get_knowledge_graph_overview() -> Dict[str, Any]:
    rag_system = get_rag_system()
    if not rag_system:
        raise RuntimeError("RAG系统未初始化")

    counts = rag_system.get_node_type_counts() or {}
    total_nodes = int(sum(counts.values()))

    total_edges = 0
    driver = _neo4j_driver()
    try:
        records = _run_query_with_compat_session(driver, "MATCH ()-[r]->() RETURN count(r) AS c")
        rec = records[0] if records else None
        total_edges = int(rec["c"] if rec and rec["c"] is not None else 0)
    finally:
        driver.close()

    return {
        "node_limit": "all",
        "nodes": [],
        "edges": [],
        "stats": {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "entity_types": counts,
        },
    }


def get_admin_dashboard() -> Dict[str, Any]:
    rag_system = get_rag_system()
    if not rag_system:
        raise RuntimeError("RAG系统未初始化")

    counts = rag_system.get_node_type_counts() or {}
    overview = get_knowledge_graph_overview()

    return {
        "version": get_knowledge_base_version(),
        "node_counts": counts,
        "graph": overview,
    }


def make_backup_zip() -> Dict[str, Any]:
    backup_root = Path("backups")
    backup_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = backup_root / f"kg_qdrant_backup_{ts}.zip"

    with tempfile.TemporaryDirectory(prefix="kg_backup_") as tmp:
        tmp_dir = Path(tmp)
        neo4j_export_dir = tmp_dir / "neo4j_csv_export"
        neo4j_csv_stats = _export_neo4j_to_csv(neo4j_export_dir)

        include_paths = [
            Path("conversations"),
            Path("config.py"),
            neo4j_export_dir,
        ]

        qdrant_local_dir = os.getenv("QDRANT_LOCAL_DATA_DIR", "")
        if qdrant_local_dir:
            include_paths.append(Path(qdrant_local_dir))

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in include_paths:
                if not p.exists():
                    continue
                if p.is_file():
                    zf.write(p, arcname=str(p.name if p.parent == tmp_dir else p))
                else:
                    for file in p.rglob("*"):
                        if file.is_file():
                            if p == neo4j_export_dir:
                                arcname = str(Path("neo4j_csv_export") / file.relative_to(neo4j_export_dir))
                            else:
                                arcname = str(file)
                            zf.write(file, arcname=arcname)

    return {
        "backup_file": str(zip_path).replace("\\", "/"),
        "backup_time": datetime.now().isoformat(timespec="seconds"),
        "neo4j_csv_export": neo4j_csv_stats,
    }


def restore_backup_zip(zip_file: UploadFile) -> Dict[str, Any]:
    restore_root = Path("backup_restores")
    restore_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"restore_{ts}_{zip_file.filename or 'backup.zip'}"
    zip_path = restore_root / zip_name

    content = zip_file.file.read()
    with open(zip_path, "wb") as f:
        f.write(content)

    target = restore_root / f"extracted_{ts}"
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target)

    csv_root = _find_csv_root(target)
    if not csv_root:
        return {
            "uploaded_backup": str(zip_path).replace("\\", "/"),
            "extracted_to": str(target).replace("\\", "/"),
            "imported": False,
            "note": "未在压缩包中找到可导入的CSV集合（需要包含equipmentcategorys/equipments/.../relationships.csv）",
        }

    importer = CSVToNeo4jImporter(
        uri=DEFAULT_CONFIG.neo4j_uri,
        user=DEFAULT_CONFIG.neo4j_user,
        password=DEFAULT_CONFIG.neo4j_password,
        database=DEFAULT_CONFIG.neo4j_database,
    )
    importer.csv_dir = str(csv_root)
    importer.import_all(clear_db=True)

    return {
        "uploaded_backup": str(zip_path).replace("\\", "/"),
        "extracted_to": str(target).replace("\\", "/"),
        "csv_root": str(csv_root).replace("\\", "/"),
        "imported": True,
        "note": "已通过 csv_to_neo4j 完成知识库导入（clear_db=True）。",
    }
