from __future__ import annotations

import uuid
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Form, Query, UploadFile, File
from pydantic import BaseModel

import yaml

from services.kg_import_service import (
    import_csv_directory_service,
    start_kg_extraction_task,
    get_kg_extraction_task_status,
    cancel_kg_extraction_task,
)


router = APIRouter(prefix="/api", tags=["kg-import"])

ALLOWED_UPLOAD_EXTENSIONS = {".md", ".txt", ".pdf", ".png", ".jpg", ".jpeg"}


class CsvImportResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    imported_files: List[str] = []
    imported_node_ids: List[str] = []


class ExtractionTaskStartResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    task_id: Optional[str] = None


class ExtractionTaskStatusResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    task: Optional[Dict[str, Any]] = None


class ExtractionOptionsResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    llm_providers: List[str] = []
    default_provider: Optional[str] = None


@router.post("/kg/csv-import", response_model=CsvImportResponse)
async def import_csv_directory(directory: str = Form("csv_generate"), merge_mode: str = Form("merge")):
    result = import_csv_directory_service(directory=directory, merge_mode=merge_mode)
    return CsvImportResponse(**result)


@router.post("/kg/extract/start", response_model=ExtractionTaskStartResponse)
async def start_extraction(
    file: UploadFile = File(...),
    output_dir: str = Form("csv_generate"),
    config_path: str = Form("config.yaml"),
    book_name: str = Form(""),
    llm_provider: str = Form(""),
    llm_api_key: str = Form(""),
    ocr_provider: str = Form(""),
    ocr_endpoint: str = Form(""),
    ocr_api_key: str = Form(""),
):
    if not file.filename:
        return ExtractionTaskStartResponse(success=False, message="上传文件名为空")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        allowed_text = ", ".join(sorted(ALLOWED_UPLOAD_EXTENSIONS))
        return ExtractionTaskStartResponse(
            success=False,
            message=f"不支持的文件类型: {ext or '无扩展名'}。仅支持: {allowed_text}",
        )

    upload_root = Path("uploads") / "extract_tasks"
    upload_root.mkdir(parents=True, exist_ok=True)

    safe_name = Path(file.filename).name
    save_name = f"{uuid.uuid4().hex}_{safe_name}"
    saved_path = upload_root / save_name

    try:
        with saved_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        return ExtractionTaskStartResponse(success=False, message=f"保存上传文件失败: {e}")
    finally:
        await file.close()

    result = start_kg_extraction_task(
        input_path=str(saved_path),
        output_dir=output_dir,
        config_path=config_path,
        book_name=book_name,
        llm_provider=llm_provider,
        llm_api_key=llm_api_key,
        ocr_provider=ocr_provider,
        ocr_endpoint=ocr_endpoint,
        ocr_api_key=ocr_api_key,
    )
    return ExtractionTaskStartResponse(**result)


@router.get("/kg/extract/status", response_model=ExtractionTaskStatusResponse)
async def extraction_status(task_id: str = Query(...), last_index: int = Query(0, ge=0)):
    result = get_kg_extraction_task_status(task_id=task_id, last_index=last_index)
    return ExtractionTaskStatusResponse(**result)


@router.post("/kg/extract/cancel", response_model=ExtractionTaskStatusResponse)
async def extraction_cancel(task_id: str = Form(...)):
    result = cancel_kg_extraction_task(task_id=task_id)
    return ExtractionTaskStatusResponse(**result)


@router.get("/kg/extract/options", response_model=ExtractionOptionsResponse)
async def extraction_options(config_path: str = Query("config.yaml")):
    project_root = Path(__file__).resolve().parent.parent
    candidates = [Path(config_path), project_root / config_path, project_root / "csv_generate" / config_path]

    config_file = None
    for p in candidates:
        if p.exists() and p.is_file():
            config_file = p
            break

    if not config_file:
        return ExtractionOptionsResponse(success=False, message=f"配置文件不存在: {config_path}")

    try:
        with config_file.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        llm_section = data.get("llm", {})
        providers = llm_section.get("providers", {})
        provider_names = sorted([str(k) for k in providers.keys()])
        default_provider = llm_section.get("current_provider")
        return ExtractionOptionsResponse(
            success=True,
            llm_providers=provider_names,
            default_provider=default_provider,
        )
    except Exception as e:
        return ExtractionOptionsResponse(success=False, message=f"读取配置失败: {e}")
