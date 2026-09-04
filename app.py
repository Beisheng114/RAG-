"""
FastAPI 应用入口（装配层）

业务逻辑已下沉：
- services/conversation_service.py  对话 CRUD（SQLite 持久化 + 内存缓存）
- services/case_state_service.py    维修过程记录（case state）全部逻辑
- services/export_service.py        对话导出
- routers/                          admin / graph / kg_import / pages 路由
- core/security.py                  鉴权与 CORS 白名单
- core/system_context.py            RAG 系统全局实例
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uuid
import json
import os
from urllib.parse import quote
from pathlib import Path
from datetime import datetime
from ragmain import AdvancedGraphRAGSystem
from core.system_context import set_rag_system
from core.security import get_api_key, get_cors_origins, check_api_access
from routers.admin_routes import router as admin_router
from routers.kg_import_routes import router as kg_import_router
from routers.graph_routes import router as graph_router
from routers.page_routes import router as page_router
from services.conversation_service import conversation_service
from services.case_state_service import (
    default_case_state,
    ensure_conversation_case_state,
    generate_conversation_title,
    _build_fault_context_system_note,
    _is_model_refusal,
    _clear_case_draft,
    generate_case_draft_from_conversation,
    extract_keywords_from_conversation,
    build_maintenance_record,
    build_postmortem,
    apply_draft,
)
from services import export_service

# 全局RAG系统实例
rag_system = None

app = FastAPI(
    title="船舶故障维修RAG系统 API",
    description="提供对话管理、问答和知识库管理功能",
    version="1.1.0"
)

# 启动事件
@app.on_event("startup")
async def startup_event():
    global rag_system
    print("正在初始化RAG系统...")
    rag_system = AdvancedGraphRAGSystem()
    rag_system.initialize_system()
    rag_system.build_knowledge_base()
    set_rag_system(rag_system)
    print("RAG系统初始化完成！")

# 配置CORS（白名单可通过环境变量 RAG_CORS_ORIGINS 覆盖，逗号分隔；
# 禁止与 allow_credentials=True 组合使用 "*"，详见 core/security.py）
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def api_key_guard(request, call_next):
    """可选的 API 鉴权中间件

    设置环境变量 RAG_API_KEY 后，所有 /api/* 请求需携带有效的
    X-API-Key（或 X-Admin-Token）头；未设置时维持本地部署的默认行为。
    CORS 预检请求（OPTIONS）直接放行。
    """
    if (
        get_api_key() is not None
        and request.method != "OPTIONS"
        and request.url.path.startswith("/api/")
        and not check_api_access(request.headers)
    ):
        return JSONResponse(
            status_code=401,
            content={"detail": "缺少或无效的 X-API-Key 请求头"},
        )
    return await call_next(request)


# 数据模型
class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: str

class Conversation(BaseModel):
    id: str
    title: str
    messages: List[Message]
    created_at: str
    updated_at: str
    case_state: Optional[Dict[str, Any]] = None

class QueryRequest(BaseModel):
    conversation_id: Optional[str] = None
    message: str
    search_mode: Optional[str] = "intelligent"

class CreateConversationRequest(BaseModel):
    title: str

class ConversationList(BaseModel):
    conversations: List[Conversation]

class QueryResponse(BaseModel):
    conversation_id: str
    message: Message

class ExportResponse(BaseModel):
    success: bool
    url: Optional[str] = None

class GraphQueryRequest(BaseModel):
    query: str
    entity_type: str = "all"
    node_limit: int = 200
    system_name: Optional[str] = "all"

class GraphNode(BaseModel):
    id: str
    label: str
    type: str
    title: Optional[str] = None
    system_name: Optional[str] = None

class GraphEdge(BaseModel):
    from_id: str
    to_id: str
    label: str

class GraphQueryResponse(BaseModel):
    success: bool
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    stats: Optional[Dict[str, Any]] = None
    message: Optional[str] = None

class ImportResponse(BaseModel):
    success: bool
    conversation_id: Optional[str] = None
    message: Optional[str] = None

class MaterialImportResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


class MaterialLibraryItem(BaseModel):
    path: str
    name: str
    size: int
    updated_at: str


class MaterialLibraryListResponse(BaseModel):
    success: bool
    files: List[MaterialLibraryItem] = []


class CsvImportResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    imported_files: List[str] = []
    imported_node_ids: List[str] = []


class AdminOpResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


# ---------- 对话管理 ----------

@app.post("/api/conversations", response_model=Conversation)
def create_conversation(request: CreateConversationRequest):
    """创建新对话"""
    conv = conversation_service.create(request.title, case_state=default_case_state())
    return Conversation(**conv)


@app.get("/api/conversations", response_model=ConversationList)
def list_conversations():
    """获取所有对话列表"""
    return ConversationList(
        conversations=[Conversation(**conv) for conv in conversation_service.list_all()]
    )


@app.get("/api/conversations/{conversation_id}", response_model=Conversation)
def get_conversation(conversation_id: str):
    """获取单个对话详情"""
    if not conversation_service.exists(conversation_id):
        raise HTTPException(status_code=404, detail="对话不存在")
    ensure_conversation_case_state(conversation_id)
    return Conversation(**conversation_service.get(conversation_id))


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(conversation_id: str):
    """删除对话"""
    if not conversation_service.delete(conversation_id):
        raise HTTPException(status_code=404, detail="对话不存在")
    return {"success": True}


# ---------- 问答 ----------

@app.post("/api/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """发送查询并获取回答"""
    if not request.conversation_id:
        title = generate_conversation_title(request.message)
        conv = conversation_service.create(title, case_state=default_case_state())
        conversation_id = conv["id"]
    else:
        conversation_id = request.conversation_id
        if not conversation_service.exists(conversation_id):
            raise HTTPException(status_code=404, detail="对话不存在")

    ensure_conversation_case_state(conversation_id)
    conv = conversation_service.get(conversation_id)

    # 添加用户消息
    user_message = Message(
        role="user",
        content=request.message,
        timestamp=datetime.now().isoformat()
    )
    conv["messages"].append(user_message.model_dump())

    # 获取对话历史（排除刚刚添加的用户消息）
    conversation_history = conv["messages"][:-1]
    # 注入“已确认故障上下文”，用于约束问答范围
    sys_note = _build_fault_context_system_note(conv.get("case_state") or {})
    if sys_note:
        conversation_history = [sys_note] + (conversation_history or [])

    # 调用RAG系统获取回答
    result, analysis = rag_system.ask_question_with_routing(
        request.message,
        stream=False,
        explain_routing=False,
        conversation_history=conversation_history,
        search_mode=request.search_mode
    )

    # 添加助手消息
    assistant_message = Message(
        role="assistant",
        content=result,
        timestamp=datetime.now().isoformat()
    )
    conv["messages"].append(assistant_message.model_dump())
    conv["updated_at"] = datetime.now().isoformat()
    conversation_service.save(conversation_id)

    # 自动生成“待确认草案”：仅在非拒答场景执行
    try:
        if _is_model_refusal(result):
            _clear_case_draft(conversation_id)
        else:
            generate_case_draft_from_conversation(conversation_id)
    except Exception:
        pass

    return QueryResponse(
        conversation_id=conversation_id,
        message=assistant_message
    )


@app.post("/api/query/stream")
async def query_stream(request: QueryRequest):
    """发送查询并获取流式回答"""
    if not request.conversation_id:
        title = generate_conversation_title(request.message)
        conv = conversation_service.create(title, case_state=default_case_state())
        conversation_id = conv["id"]
    else:
        conversation_id = request.conversation_id
        if not conversation_service.exists(conversation_id):
            raise HTTPException(status_code=404, detail="对话不存在")

    ensure_conversation_case_state(conversation_id)
    conv = conversation_service.get(conversation_id)

    # 添加用户消息
    user_message = Message(
        role="user",
        content=request.message,
        timestamp=datetime.now().isoformat()
    )
    conv["messages"].append(user_message.model_dump())

    # 获取对话历史（排除刚刚添加的用户消息）
    conversation_history = conv["messages"][:-1]

    # 定义生成器函数
    async def generate():
        full_response = ""

        # 调用RAG系统的流式输出
        result, analysis = rag_system.ask_question_with_routing(
            request.message,
            stream=True,
            explain_routing=False,
            conversation_history=conversation_history,
            search_mode=request.search_mode
        )

        # 如果返回的是生成器（流式输出）
        if hasattr(result, '__iter__') and not isinstance(result, str):
            # 使用async for处理生成器
            import asyncio
            loop = asyncio.get_event_loop()

            # 将同步生成器转换为异步
            def iter_generator():
                for chunk in result:
                    if chunk:
                        yield chunk

            # 在后台线程中运行生成器
            gen = iter_generator()
            while True:
                try:
                    chunk = await loop.run_in_executor(None, next, gen, None)
                    if chunk is None:
                        break
                    full_response += chunk
                    yield f"data: {json.dumps({'content': chunk, 'conversation_id': conversation_id}, ensure_ascii=False)}\n\n"
                except StopIteration:
                    break
        else:
            # 如果不是流式输出，直接返回结果
            full_response = result
            yield f"data: {json.dumps({'content': result, 'conversation_id': conversation_id}, ensure_ascii=False)}\n\n"

        # 添加助手消息到对话
        assistant_message = Message(
            role="assistant",
            content=full_response,
            timestamp=datetime.now().isoformat()
        )
        conv["messages"].append(assistant_message.model_dump())
        conv["updated_at"] = datetime.now().isoformat()
        conversation_service.save(conversation_id)

        # 自动生成“待确认草案”：仅在非拒答场景执行
        try:
            if _is_model_refusal(full_response):
                _clear_case_draft(conversation_id)
            else:
                generate_case_draft_from_conversation(conversation_id)
        except Exception:
            pass

        # 发送结束标记
        yield f"data: {json.dumps({'done': True, 'conversation_id': conversation_id}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ---------- 维修过程记录（case state） ----------

class CaseStateResponse(BaseModel):
    success: bool
    conversation_id: str
    case_state: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


class CaseStateUpdateRequest(BaseModel):
    case_state: Dict[str, Any]

class ApplyDraftRequest(BaseModel):
    fault_context: Optional[Dict[str, Any]] = None
    measures: Optional[List[str]] = None
    todos: Optional[List[str]] = None
    components: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    # 默认不自动确认故障范围，需用户在前端显式确认
    confirm_fault_context: Optional[bool] = False


@app.get("/api/conversations/{conversation_id}/case-state", response_model=CaseStateResponse)
def get_case_state(conversation_id: str):
    ensure_conversation_case_state(conversation_id)
    return CaseStateResponse(
        success=True,
        conversation_id=conversation_id,
        case_state=conversation_service.get(conversation_id)["case_state"],
    )


@app.put("/api/conversations/{conversation_id}/case-state", response_model=CaseStateResponse)
def update_case_state(conversation_id: str, request: CaseStateUpdateRequest):
    ensure_conversation_case_state(conversation_id)
    if not isinstance(request.case_state, dict):
        raise HTTPException(status_code=400, detail="case_state 必须是对象")

    conv = conversation_service.get(conversation_id)
    base = default_case_state()
    # 只允许更新白名单字段，避免前端误写导致结构污染
    allowed_keys = set(base.keys())
    incoming = {k: v for k, v in request.case_state.items() if k in allowed_keys}

    merged = {**conv["case_state"], **incoming}
    conv["case_state"] = merged
    conv["updated_at"] = datetime.now().isoformat()
    conversation_service.save(conversation_id)
    return CaseStateResponse(success=True, conversation_id=conversation_id, case_state=merged)


@app.post("/api/conversations/{conversation_id}/case-state/generate-maintenance-record", response_model=CaseStateResponse)
def generate_maintenance_record(conversation_id: str):
    state = build_maintenance_record(conversation_id)
    return CaseStateResponse(success=True, conversation_id=conversation_id, case_state=state)


@app.post("/api/conversations/{conversation_id}/case-state/generate-postmortem", response_model=CaseStateResponse)
def generate_postmortem(conversation_id: str):
    state = build_postmortem(conversation_id)
    return CaseStateResponse(success=True, conversation_id=conversation_id, case_state=state)


@app.post("/api/conversations/{conversation_id}/case-state/generate-keywords", response_model=CaseStateResponse)
def generate_case_keywords(conversation_id: str):
    """为当前对话提取“维修/预防”关键词，并写入 case_state.keywords。"""
    ensure_conversation_case_state(conversation_id)
    extract_keywords_from_conversation(conversation_id)
    return CaseStateResponse(
        success=True,
        conversation_id=conversation_id,
        case_state=conversation_service.get(conversation_id)["case_state"],
    )


@app.get("/api/conversations/{conversation_id}/case-state/keywords", response_model=CaseStateResponse)
def get_case_keywords(conversation_id: str):
    """获取当前对话的关键词；若尚未生成，则自动生成并返回。"""
    ensure_conversation_case_state(conversation_id)
    try:
        # 若已存在关键词，extract_keywords_from_conversation 会直接复用
        extract_keywords_from_conversation(conversation_id)
    except Exception:
        pass
    return CaseStateResponse(
        success=True,
        conversation_id=conversation_id,
        case_state=conversation_service.get(conversation_id)["case_state"],
    )


@app.post("/api/conversations/{conversation_id}/case-state/generate-draft", response_model=CaseStateResponse)
def generate_case_draft(conversation_id: str):
    """生成“待用户确认”的故障草案（故障范围/建议措施/待处理/关键词）。"""
    ensure_conversation_case_state(conversation_id)
    generate_case_draft_from_conversation(conversation_id)
    return CaseStateResponse(
        success=True,
        conversation_id=conversation_id,
        case_state=conversation_service.get(conversation_id)["case_state"],
    )


@app.post("/api/conversations/{conversation_id}/case-state/apply-draft", response_model=CaseStateResponse)
def apply_case_draft(conversation_id: str, request: ApplyDraftRequest):
    """接受（可部分拒绝后的）草案内容，合并到正式 case_state。"""
    state = apply_draft(
        conversation_id,
        fault_context=request.fault_context,
        measures=request.measures,
        todos=request.todos,
        components=request.components,
        keywords=request.keywords,
        confirm_fault_context=bool(request.confirm_fault_context),
    )
    return CaseStateResponse(success=True, conversation_id=conversation_id, case_state=state)


# ---------- 导出 / 导入 ----------

@app.get("/api/export/{conversation_id}", response_model=ExportResponse)
def export_conversation(conversation_id: str):
    """导出对话"""
    if not conversation_service.exists(conversation_id):
        raise HTTPException(status_code=404, detail="对话不存在")
    url = export_service.export_json(conversation_service.get(conversation_id))
    return ExportResponse(success=True, url=url)


@app.get("/api/export-md/{conversation_id}", response_model=ExportResponse)
def export_conversation_to_markdown(conversation_id: str):
    """导出对话为 Markdown 文档 (.md)"""
    if not conversation_service.exists(conversation_id):
        raise HTTPException(status_code=404, detail="对话不存在")
    url = export_service.export_markdown(conversation_service.get(conversation_id))
    return ExportResponse(success=True, url=url)


@app.get("/api/system/stats")
def get_system_stats():
    """获取系统统计信息"""
    return conversation_service.stats()


@app.post("/api/import", response_model=ImportResponse)
async def import_conversation(file: UploadFile = File(...), title: str = Form(...)):
    """导入对话"""
    try:
        # 读取上传的文件
        contents = await file.read()
        conversation_data = json.loads(contents)

        # 验证文件格式
        if "messages" not in conversation_data:
            return ImportResponse(success=False, message="无效的对话文件格式")

        # 处理消息内容，确保格式正确
        for message in conversation_data["messages"]:
            # 确保消息包含必要的字段
            if "role" not in message:
                message["role"] = "assistant"
            if "content" not in message:
                message["content"] = ""
            if "timestamp" not in message:
                message["timestamp"] = datetime.now().isoformat()

        # 创建新对话
        conversation_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        conversation = {
            "id": conversation_id,
            "title": title,
            "messages": conversation_data["messages"],
            "created_at": now,
            "updated_at": now
        }
        conversation_service.save_conv(conversation)

        return ImportResponse(success=True, conversation_id=conversation_id)
    except Exception as e:
        return ImportResponse(success=False, message=str(e))

# 资料查询（本地 PDF）
SOURCE_DIR = Path("./source")
SOURCE_DIR.mkdir(parents=True, exist_ok=True)


def _safe_source_path(filename: str) -> Path:
    safe_name = Path(filename or "").name
    if not safe_name:
        raise HTTPException(status_code=400, detail="文件名无效")
    target = (SOURCE_DIR / safe_name).resolve()
    source_root = SOURCE_DIR.resolve()
    if source_root != target and source_root not in target.parents:
        raise HTTPException(status_code=400, detail="非法路径")
    return target


@app.get("/api/material-library", response_model=MaterialLibraryListResponse)
def list_material_library(keyword: Optional[str] = None):
    files: List[MaterialLibraryItem] = []
    kw = (keyword or "").strip().lower()

    if not SOURCE_DIR.exists():
        SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    for p in SOURCE_DIR.iterdir():
        if not p.is_file() or p.suffix.lower() != ".pdf":
            continue
        if kw and kw not in p.name.lower():
            continue
        stat = p.stat()
        encoded_name = quote(p.name)
        files.append(MaterialLibraryItem(
            path=f"/api/material-library/open/{encoded_name}",
            name=p.name,
            size=stat.st_size,
            updated_at=datetime.fromtimestamp(stat.st_mtime).isoformat()
        ))

    files.sort(key=lambda x: x.updated_at, reverse=True)
    return MaterialLibraryListResponse(success=True, files=files)


@app.post("/api/material-library/import")
async def import_material_library(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="请上传 PDF 文件")

    imported: List[str] = []
    skipped: List[str] = []

    for f in files:
        filename = Path(f.filename or "").name
        if not filename.lower().endswith(".pdf"):
            skipped.append(filename or "(unknown)")
            continue

        target = _safe_source_path(filename)
        content = await f.read()
        with open(target, "wb") as out:
            out.write(content)
        imported.append(filename)

    return {
        "success": True,
        "message": f"导入完成：成功 {len(imported)} 个，跳过 {len(skipped)} 个",
        "imported": imported,
        "skipped": skipped,
    }


@app.get("/api/material-library/open/{filename:path}")
def open_material_pdf(filename: str):
    target = _safe_source_path(filename)
    if not target.exists() or not target.is_file() or target.suffix.lower() != ".pdf":
        raise HTTPException(status_code=404, detail="PDF 文件不存在")
    return FileResponse(
        path=str(target),
        media_type="application/pdf",
        headers={"Content-Disposition": f"inline; filename*=UTF-8''{quote(target.name)}"}
    )


# 路由拆分注册
app.include_router(kg_import_router)
app.include_router(admin_router)
app.include_router(graph_router)
app.include_router(page_router)

# 静态文件服务
from fastapi.staticfiles import StaticFiles

if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/static", StaticFiles(directory="static"), name="static")

if not os.path.exists("exports"):
    os.makedirs("exports")

app.mount("/exports", StaticFiles(directory="exports"), name="exports")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
