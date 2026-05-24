from __future__ import annotations

import os
import sys
import uuid
import time
import threading
import importlib
import contextlib
from pathlib import Path
from typing import Dict, Any, List

import yaml

from core.system_context import get_rag_system


# ==================== 现有CSV导入能力 ====================

def import_csv_directory_service(directory: str = "csv_generate", merge_mode: str = "merge") -> Dict[str, Any]:
    rag_system = get_rag_system()
    if not rag_system or not rag_system.graph_data_insert:
        return {"success": False, "message": "RAG系统未初始化"}

    target_dir = Path(directory)
    if not target_dir.exists() or not target_dir.is_dir():
        return {"success": False, "message": f"目录不存在: {directory}"}

    csv_files = sorted(target_dir.glob("*.csv"))
    if not csv_files:
        return {"success": False, "message": f"目录中未找到CSV文件: {directory}"}

    imported_files: List[str] = []
    all_node_ids: List[str] = []

    for csv_file in csv_files:
        try:
            import pandas as pd
            df = pd.read_csv(csv_file)
        except Exception as e:
            return {"success": False, "message": f"读取CSV失败 {csv_file.name}: {e}"}

        if df.empty:
            continue

        for _, row in df.iterrows():
            pieces = []
            for col in df.columns:
                val = row.get(col)
                if val is None:
                    continue
                sval = str(val).strip()
                if sval:
                    pieces.append(f"{col}: {sval}")
            case_text = "\n".join(pieces)
            if not case_text:
                continue

            result = rag_system.graph_data_insert.insert_case(
                case=case_text,
                dry_run=False,
                use_deepseek=False,
            )
            if result.get("success"):
                all_node_ids.extend(result.get("neo4j_node_ids", []) or [])

        imported_files.append(str(csv_file).replace("\\", "/"))

    unique_node_ids = list(dict.fromkeys([str(x) for x in all_node_ids if x]))
    if unique_node_ids:
        rag_system.update_knowledge_base_incremental(unique_node_ids)

    return {
        "success": True,
        "message": f"CSV导入完成，模式={merge_mode}",
        "imported_files": imported_files,
        "imported_node_ids": unique_node_ids,
    }


# ==================== kg_generator_v2 抽取任务 ====================

TASKS_LOCK = threading.Lock()
EXTRACTION_TASKS: Dict[str, Dict[str, Any]] = {}
MAX_LOG_LINES = 2000


_OCR_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


class _TaskLogWriter:
    def __init__(self, task: Dict[str, Any]):
        self.task = task

    def write(self, message: str):
        if not message:
            return
        for line in message.splitlines():
            if line.strip():
                _append_task_log(self.task, line)

    def flush(self):
        return


def _append_task_log(task: Dict[str, Any], line: str):
    logs = task.setdefault("logs", [])
    logs.append(line.rstrip("\n"))
    if len(logs) > MAX_LOG_LINES:
        task["logs"] = logs[-MAX_LOG_LINES:]


def _needs_ocr(input_path: Path) -> bool:
    return input_path.is_file() and input_path.suffix.lower() in _OCR_EXTENSIONS


def _perform_external_ocr(
    source_path: Path,
    output_md_path: Path,
    ocr_provider: str = "",
    ocr_endpoint: str = "",
    ocr_api_key: str = "",
) -> Dict[str, Any]:
    if not ocr_endpoint.strip():
        return {
            "success": False,
            "message": "当前文件需要OCR识别，但未配置ocr_endpoint。请对接外部OCR服务后再重试。",
        }

    try:
        import requests

        headers = {}
        if ocr_api_key.strip():
            headers["Authorization"] = f"Bearer {ocr_api_key.strip()}"
        if ocr_provider.strip():
            headers["X-OCR-Provider"] = ocr_provider.strip()

        with source_path.open("rb") as f:
            files = {"file": (source_path.name, f)}
            response = requests.post(ocr_endpoint, headers=headers, files=files, timeout=180)
            response.raise_for_status()
            data = response.json() if response.content else {}

        text = str(data.get("text", "")).strip()
        if not text:
            return {"success": False, "message": "OCR服务返回为空，缺少可用text内容"}

        output_md_path.parent.mkdir(parents=True, exist_ok=True)
        output_md_path.write_text(text, encoding="utf-8")
        return {
            "success": True,
            "message": f"OCR识别完成，输出: {str(output_md_path).replace('\\', '/')}",
            "ocr_text_length": len(text),
        }
    except Exception as e:
        return {"success": False, "message": f"OCR调用失败: {e}"}


def _import_generator_module(project_root: Path):
    csv_dir = project_root / "csv_generate"
    if not csv_dir.exists():
        raise FileNotFoundError(f"未找到目录: {str(csv_dir).replace('\\', '/')}")

    csv_dir_str = str(csv_dir)
    if csv_dir_str not in sys.path:
        sys.path.insert(0, csv_dir_str)

    return importlib.import_module("kg_generator_v2")


def _run_kg_generator_directly(
    task: Dict[str, Any],
    input_path: Path,
    output_dir: Path,
    config_path: Path,
    book_name: str,
):
    project_root = Path(__file__).resolve().parent.parent
    module = _import_generator_module(project_root)

    if not hasattr(module, "KnowledgeGraphGenerator"):
        raise AttributeError("kg_generator_v2.py 中未找到 KnowledgeGraphGenerator")

    generator_cls = getattr(module, "KnowledgeGraphGenerator")
    generator = generator_cls(str(config_path))

    if input_path.is_file():
        generator.process_md_file(str(input_path), book_name or "")
        generator.relationship_manager.auto_complete_relations()
    elif input_path.is_dir():
        generator.process_directory(str(input_path), book_name or "")
    else:
        raise FileNotFoundError(f"输入路径不存在: {str(input_path)}")

    generator.save_to_csv(str(output_dir))
    total_entities = len(generator.knowledge_pool.get_all_entities())
    total_relations = len(generator.relationship_manager.get_relations())
    _append_task_log(task, f"[DONE] 处理完成，总实体数={total_entities}，总关系数={total_relations}")


def _run_extraction_task(task_id: str):
    with TASKS_LOCK:
        task = EXTRACTION_TASKS.get(task_id)
        if not task:
            return
        task["status"] = "running"
        task["started_at"] = int(time.time())

    input_path = Path(task["input_path"])
    output_dir = Path(task["output_dir"])
    config_path = Path(task["config_path"])
    book_name = task.get("book_name", "").strip()

    actual_input = input_path
    if _needs_ocr(input_path):
        _append_task_log(task, f"[INFO] 检测到OCR文件: {input_path.name}，开始调用外部OCR...")
        temp_ocr_dir = output_dir / "_ocr_cache"
        temp_md = temp_ocr_dir / f"{input_path.stem}_ocr.md"
        ocr_result = _perform_external_ocr(
            source_path=input_path,
            output_md_path=temp_md,
            ocr_provider=task.get("ocr_provider", ""),
            ocr_endpoint=task.get("ocr_endpoint", ""),
            ocr_api_key=task.get("ocr_api_key", ""),
        )
        _append_task_log(task, f"[OCR] {ocr_result.get('message', '')}")
        if not ocr_result.get("success"):
            with TASKS_LOCK:
                task["status"] = "failed"
                task["finished_at"] = int(time.time())
                task["error"] = ocr_result.get("message")
            return
        actual_input = temp_md

    if not config_path.exists():
        with TASKS_LOCK:
            task["status"] = "failed"
            task["finished_at"] = int(time.time())
            task["error"] = f"配置文件不存在: {str(config_path).replace('\\', '/')}"
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    _append_task_log(
        task,
        f"[RUN] 直接调用 kg_generator_v2 模块，input={str(actual_input).replace('\\', '/')}",
    )

    log_writer = _TaskLogWriter(task)
    try:
        with contextlib.redirect_stdout(log_writer), contextlib.redirect_stderr(log_writer):
            _run_kg_generator_directly(
                task=task,
                input_path=actual_input,
                output_dir=output_dir,
                config_path=config_path,
                book_name=book_name,
            )

        with TASKS_LOCK:
            task["finished_at"] = int(time.time())
            task["return_code"] = 0
            task["status"] = "completed"
    except Exception as e:
        with TASKS_LOCK:
            task["status"] = "failed"
            task["finished_at"] = int(time.time())
            task["return_code"] = 1
            task["error"] = str(e)
            _append_task_log(task, f"[ERROR] 任务异常: {e}")


def _resolve_config_file(config_path: str) -> Path:
    candidate = Path(config_path)
    if candidate.exists():
        return candidate.resolve()

    project_root = Path(__file__).resolve().parent.parent
    fallback = project_root / "csv_generate" / config_path
    if fallback.exists():
        return fallback.resolve()

    return candidate.resolve()


def _apply_runtime_llm_config(config_file: Path, llm_provider: str = "", llm_api_key: str = ""):
    if not llm_provider and not llm_api_key:
        return

    with config_file.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    llm_section = data.setdefault("llm", {})
    providers = llm_section.setdefault("providers", {})

    if llm_provider:
        llm_section["current_provider"] = llm_provider

    provider_name = llm_section.get("current_provider", "")
    if provider_name and llm_api_key:
        provider_conf = providers.setdefault(provider_name, {})
        provider_conf["api_key"] = llm_api_key

    with config_file.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def _create_runtime_config(config_path: str, llm_provider: str = "", llm_api_key: str = "") -> Path:
    source = _resolve_config_file(config_path)
    if not source.exists():
        return source

    runtime_dir = Path(__file__).resolve().parent.parent / "uploads" / "runtime_configs"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    runtime_path = runtime_dir / f"kg_runtime_{uuid.uuid4().hex}.yaml"
    runtime_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    _apply_runtime_llm_config(runtime_path, llm_provider=llm_provider, llm_api_key=llm_api_key)
    return runtime_path


def start_kg_extraction_task(
    input_path: str,
    output_dir: str,
    config_path: str = "config.yaml",
    book_name: str = "",
    ocr_provider: str = "",
    ocr_endpoint: str = "",
    ocr_api_key: str = "",
    llm_provider: str = "",
    llm_api_key: str = "",
) -> Dict[str, Any]:
    source = Path(input_path)
    if not source.exists():
        return {"success": False, "message": f"输入路径不存在: {input_path}"}

    runtime_config = _create_runtime_config(
        config_path=config_path,
        llm_provider=llm_provider,
        llm_api_key=llm_api_key,
    )

    task_id = uuid.uuid4().hex
    task = {
        "task_id": task_id,
        "status": "pending",
        "input_path": str(source.resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "config_path": str(runtime_config.resolve()),
        "book_name": book_name or "",
        "ocr_provider": ocr_provider or "",
        "ocr_endpoint": ocr_endpoint or "",
        "ocr_api_key": ocr_api_key or "",
        "created_at": int(time.time()),
        "started_at": None,
        "finished_at": None,
        "pid": None,
        "return_code": None,
        "error": None,
        "logs": [],
    }

    with TASKS_LOCK:
        EXTRACTION_TASKS[task_id] = task

    worker = threading.Thread(target=_run_extraction_task, args=(task_id,), daemon=True)
    worker.start()

    return {"success": True, "message": "抽取任务已启动", "task_id": task_id}


def get_kg_extraction_task_status(task_id: str, last_index: int = 0) -> Dict[str, Any]:
    with TASKS_LOCK:
        task = EXTRACTION_TASKS.get(task_id)
        if not task:
            return {"success": False, "message": "任务不存在"}

        logs = task.get("logs", [])
        safe_last_index = max(0, int(last_index or 0))
        new_logs = logs[safe_last_index:]

        return {
            "success": True,
            "task": {
                "task_id": task["task_id"],
                "status": task["status"],
                "input_path": task["input_path"],
                "output_dir": task["output_dir"],
                "config_path": task["config_path"],
                "book_name": task.get("book_name", ""),
                "created_at": task["created_at"],
                "started_at": task["started_at"],
                "finished_at": task["finished_at"],
                "pid": task["pid"],
                "return_code": task["return_code"],
                "error": task["error"],
                "log_size": len(logs),
                "new_logs": new_logs,
                "next_index": len(logs),
            },
        }


def cancel_kg_extraction_task(task_id: str) -> Dict[str, Any]:
    with TASKS_LOCK:
        task = EXTRACTION_TASKS.get(task_id)
        if not task:
            return {"success": False, "message": "任务不存在"}

        if task["status"] not in {"pending", "running"}:
            return {"success": False, "message": f"当前状态不支持取消: {task['status']}"}

        task["status"] = "cancelled"
        task["finished_at"] = int(time.time())
        _append_task_log(task, "[CANCEL] 任务已取消（当前为模块内调用，若已执行中需等待当前阶段结束）")

    return {"success": True, "message": "已标记取消任务"}
