import os
from fastapi import APIRouter
from fastapi.responses import RedirectResponse, FileResponse

router = APIRouter(tags=["pages"])

INDEX_HTML_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static", "index.html")


@router.get("/")
def root_redirect():
    return RedirectResponse(url="/static/home.html")


@router.get("/index.html")
def index_html():
    if os.path.exists(INDEX_HTML_PATH):
        return FileResponse(INDEX_HTML_PATH, media_type="text/html")
    return RedirectResponse(url="/static/index.html")
