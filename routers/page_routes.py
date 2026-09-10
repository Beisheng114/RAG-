import os
from fastapi import APIRouter
from fastapi.responses import RedirectResponse, FileResponse

router = APIRouter(tags=["pages"])

INDEX_HTML_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static", "index.html")


@router.get("/")
def root_index():
    """根路由直接返回 SPA 单页面载体（index.html，默认启动页）。"""
    if os.path.exists(INDEX_HTML_PATH):
        return FileResponse(INDEX_HTML_PATH, media_type="text/html")
    return RedirectResponse(url="/static/index.html")


@router.get("/index.html")
def index_html():
    if os.path.exists(INDEX_HTML_PATH):
        return FileResponse(INDEX_HTML_PATH, media_type="text/html")
    return RedirectResponse(url="/static/index.html")


# 合并删除的独立页面：旧路径重定向到 index.html 对应 hash 视图（SPA 单载体方案）
_LEGACY_PAGE_REDIRECTS = {
    "home.html": "#launch-page",
    "extract.html": "#extract-page",
    "kg_admin.html": "#kb-manager-page",
    "config.html": "#config-page",
}


@router.get("/static/home.html")
def legacy_home_html():
    return RedirectResponse(url="/static/index.html#launch-page")


@router.get("/static/extract.html")
def legacy_extract_html():
    return RedirectResponse(url="/static/index.html#extract-page")


@router.get("/static/kg_admin.html")
def legacy_kg_admin_html():
    return RedirectResponse(url="/static/index.html#kb-manager-page")


@router.get("/static/config.html")
def legacy_config_html():
    return RedirectResponse(url="/static/index.html#config-page")
