from typing import Optional, Dict, Any

from fastapi import APIRouter, Depends, UploadFile, File
from pydantic import BaseModel

from services.admin_service import (
    require_admin,
    get_safe_config_preview,
    make_backup_zip,
    restore_backup_zip,
    get_admin_dashboard,
    get_knowledge_base_version,
    get_knowledge_graph_overview,
)


router = APIRouter(prefix="/api", tags=["admin"])


class AdminOpResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


@router.get("/config/preview", response_model=AdminOpResponse)
def config_preview(_admin: bool = Depends(require_admin)):
    return AdminOpResponse(success=True, data={"config": get_safe_config_preview()})


@router.get("/admin/dashboard", response_model=AdminOpResponse)
def admin_dashboard(_admin: bool = Depends(require_admin)):
    try:
        data = get_admin_dashboard()
        return AdminOpResponse(success=True, data=data)
    except Exception as e:
        return AdminOpResponse(success=False, message=str(e))


@router.get("/admin/version", response_model=AdminOpResponse)
def admin_version(_admin: bool = Depends(require_admin)):
    try:
        data = get_knowledge_base_version()
        return AdminOpResponse(success=True, data=data)
    except Exception as e:
        return AdminOpResponse(success=False, message=str(e))


@router.get("/admin/graph-overview", response_model=AdminOpResponse)
def admin_graph_overview(_admin: bool = Depends(require_admin)):
    try:
        data = get_knowledge_graph_overview()
        return AdminOpResponse(success=True, data=data)
    except Exception as e:
        return AdminOpResponse(success=False, message=str(e))


@router.post("/admin/backup", response_model=AdminOpResponse)
def admin_backup(_admin: bool = Depends(require_admin)):
    try:
        data = make_backup_zip()
        return AdminOpResponse(success=True, message="备份完成", data=data)
    except Exception as e:
        return AdminOpResponse(success=False, message=str(e))


@router.post("/admin/restore", response_model=AdminOpResponse)
async def admin_restore(backup_file: UploadFile = File(...), _admin: bool = Depends(require_admin)):
    try:
        data = restore_backup_zip(backup_file)
        return AdminOpResponse(success=True, message="备份包已导入并解压", data=data)
    except Exception as e:
        return AdminOpResponse(success=False, message=str(e))
