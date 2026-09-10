import os
from fastapi import APIRouter
from fastapi.responses import RedirectResponse

router = APIRouter(tags=["pages"])

INDEX_HTML_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static", "index.html")


@router.get("/")
def root_index():
    """根路由重定向到 /static/index.html（SPA 单页面载体）。

    注意：不能直接 FileResponse 返回 index.html——页面内全部静态资源
    是相对路径引用（./css/...、libs/...、js/...），必须让页面 URL 位于
    /static/ 之下，相对路径才能正确解析到 /static/css/... 等资源；
    直接挂在根路径会解析为 /css/...（404）导致整站样式与脚本失效。
    """
    if os.path.exists(INDEX_HTML_PATH):
        return RedirectResponse(url="/static/index.html")
    return RedirectResponse(url="/static/index.html")


@router.get("/index.html")
def index_html():
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
