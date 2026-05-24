from typing import Dict, Any, List, Optional
import os
import tempfile
from pathlib import Path

from core.system_context import get_rag_system


MAX_CASE_TEXT_LEN = 5000
MAX_UPLOAD_FILES = 1
MAX_UPLOAD_FILE_SIZE = 8 * 1024 * 1024  # 8MB
ALLOWED_UPLOAD_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".pdf"}


def query_graph_data(query: str, entity_type: str, node_limit: int, system_name: str) -> Dict[str, Any]:
    rag_system = get_rag_system()
    if not rag_system:
        return {"success": False, "message": "RAG系统未初始化", "nodes": [], "edges": [], "stats": None}

    nodes, edges, stats = rag_system.query_graph(query, entity_type, node_limit, system_name)
    return {
        "success": True,
        "nodes": nodes,
        "edges": edges,
        "stats": stats,
        "message": None,
    }


def get_node_counts_data() -> Dict[str, Any]:
    rag_system = get_rag_system()
    if not rag_system:
        return {"success": False, "message": "RAG系统未初始化", "counts": None}

    counts = rag_system.get_node_type_counts()
    return {"success": True, "counts": counts, "message": None}


def _validate_material_input(case_text: str, uploaded_files: Optional[List[Any]]) -> Optional[str]:
    text = (case_text or "")
    if len(text) > MAX_CASE_TEXT_LEN:
        return f"文本过长，最多允许 {MAX_CASE_TEXT_LEN} 字"

    files = [f for f in (uploaded_files or []) if f]
    if len(files) > MAX_UPLOAD_FILES:
        return "最多只允许上传 1 个文件（图片或 PDF）"

    if files:
        filename = (getattr(files[0], "filename", "") or "").strip()
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ALLOWED_UPLOAD_EXTS:
            return "仅支持上传 1 张图片（png/jpg/jpeg/webp/bmp）或 1 个 PDF"

        file_obj = getattr(files[0], "file", None)
        if file_obj is not None:
            try:
                cur = file_obj.tell()
            except Exception:
                cur = 0
            file_obj.seek(0, os.SEEK_END)
            size = file_obj.tell()
            file_obj.seek(cur, os.SEEK_SET)
            if size > MAX_UPLOAD_FILE_SIZE:
                return f"文件过大，单文件最大 {MAX_UPLOAD_FILE_SIZE // (1024 * 1024)}MB"

    return None


def _save_uploaded_files_to_temp(uploaded_files) -> List[str]:
    saved_paths: List[str] = []
    if not uploaded_files:
        return saved_paths

    temp_root = Path(tempfile.mkdtemp(prefix="material_upload_"))
    for f in uploaded_files:
        if not f:
            continue
        filename = (getattr(f, "filename", "") or "").strip()
        if not filename:
            continue
        suffix = os.path.splitext(filename)[1].lower()
        if suffix not in ALLOWED_UPLOAD_EXTS:
            continue
        dst = temp_root / f"{len(saved_paths)+1:03d}{suffix}"
        content = f.file.read()
        with open(dst, "wb") as wf:
            wf.write(content)
        saved_paths.append(str(dst))
    return saved_paths


def preview_material_data(case_text: str, uploaded_files: Optional[List[Any]] = None) -> Dict[str, Any]:
    rag_system = get_rag_system()
    if not rag_system or not rag_system.graph_data_insert:
        return {"success": False, "message": "RAG系统未初始化"}

    err = _validate_material_input(case_text, uploaded_files)
    if err:
        return {"success": False, "message": err}

    text = (case_text or "").strip()
    file_paths = _save_uploaded_files_to_temp(uploaded_files)

    image_paths = [p for p in file_paths if p.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp"))]

    if image_paths:
        parsed_data = rag_system.graph_data_insert.analyze_case(case=text or "图片资料", use_deepseek=False, image_paths=image_paths)
    else:
        if not text:
            return {"success": False, "message": "请提供案例文本或上传图片/PDF/Word"}
        parsed_data = rag_system.graph_data_insert.analyze_case(case=text, use_deepseek=False)

    import uuid
    case_key = uuid.uuid4().hex
    cyphers = rag_system.graph_data_insert.build_cypher(parsed_data, case_key=case_key)
    relation_key_values = rag_system.graph_data_insert.create_relation_key_values(parsed_data, case_key=case_key)

    return {
        "success": True,
        "message": "预览成功",
        "result": {
            "parsed_data": parsed_data,
            "cyphers": cyphers,
            "statistics": {
                "maintenance_steps": len(parsed_data.get("维修步骤", [])),
                "cypher_statements": len(cyphers),
                "relation_key_values": len(relation_key_values),
            },
        },
    }


def import_material_with_parsed_data(parsed_data_obj: Dict[str, Any], case_text: str, uploaded_files: Optional[List[Any]] = None) -> Dict[str, Any]:
    rag_system = get_rag_system()
    if not rag_system or not rag_system.graph_data_insert:
        return {"success": False, "message": "RAG系统未初始化"}

    err = _validate_material_input(case_text, uploaded_files)
    if err:
        return {"success": False, "message": err}

    _ = _save_uploaded_files_to_temp(uploaded_files)

    result = rag_system.graph_data_insert.insert_case(
        case=case_text,
        dry_run=False,
        use_deepseek=False,
        parsed_data=parsed_data_obj,
    )

    if result.get("success"):
        neo4j_node_ids = result.get("neo4j_node_ids", []) or []
        rag_system.update_knowledge_base_incremental(neo4j_node_ids)

    return {
        "success": bool(result.get("success")),
        "message": result.get("message"),
        "result": result.get("result"),
    }


def import_material_data(case_text: str, uploaded_files: Optional[List[Any]] = None) -> Dict[str, Any]:
    rag_system = get_rag_system()
    if not rag_system or not rag_system.graph_data_insert:
        return {"success": False, "message": "RAG系统未初始化", "result": None}

    err = _validate_material_input(case_text, uploaded_files)
    if err:
        return {"success": False, "message": err, "result": None}

    text = (case_text or "").strip()
    file_paths = _save_uploaded_files_to_temp(uploaded_files)
    image_paths = [p for p in file_paths if p.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp"))]

    if not text and not image_paths:
        return {"success": False, "message": "请提供案例文本或上传图片/PDF/Word", "result": None}

    result = rag_system.graph_data_insert.insert_case(
        case=text or "图片资料",
        dry_run=False,
        use_deepseek=False,
        image_paths=image_paths if image_paths else None,
    )

    if result.get("success"):
        neo4j_node_ids = result.get("neo4j_node_ids", []) or []
        rag_system.update_knowledge_base_incremental(neo4j_node_ids)
        return {"success": True, "message": "资料导入成功", "result": result}

    return {
        "success": False,
        "message": result.get("message", "导入失败"),
        "result": result,
    }
