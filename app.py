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

# 全局RAG系统实例
rag_system = None

app = FastAPI(
    title="船舶故障维修RAG系统 API",
    description="提供对话管理、问答和知识库管理功能",
    version="1.0.0"
)

# ========= 维修过程记录（对话侧边卡片） =========
def _now_iso() -> str:
    return datetime.now().isoformat()


def default_case_state() -> Dict[str, Any]:
    return {
        "status": "in_progress",  # in_progress | ended
        "maintenance_measures": [],  # 新建待确认 [{id, text, created_at, status='new'}]
        "ready_measures": [],  # 已确认待执行 [{id, text, created_at, confirmed_at, status='confirmed'}]
        "todo": [],  # [{id, text, created_at, done: bool}]
        "confirmed_measures": [],  # 已执行 [{id, text, created_at, confirmed_at, executed_at, status='executed'}]
        "keywords": [],  # [str]
        "failed_components": [],  # [str]
        "fault_context": {
            # 由用户在右侧卡片确认的“本次故障范围”，用于约束问答与复盘
            "equipment": "",          # 设备/机器名称
            "fault_summary": "",      # 故障概述（一句话）
            "phenomenon": "",         # 故障现象（可选）
            "confirmed": False,       # 用户是否确认
            "confirmed_at": "",       # 确认时间
        },
        "draft": {
            # 模型从对话中提取的“待用户确认”草案，不直接当事实写入
            "fault_context": {
                "equipment": "",
                "fault_summary": "",
                "phenomenon": "",
            },
            "suggested_measures": [],   # 建议维修措施（待确认）
            "suggested_todos": [],      # 建议待处理事项（待确认）
            "suggested_components": [], # 建议故障部件（待确认）
            "keywords": [],             # 关键词（XXX故障）
            "generated_at": "",
        },
        "maintenance_record": "",  # markdown/text
        "postmortem": "",  # markdown/text
        "updated_at": _now_iso(),
    }


def ensure_conversation_case_state(conversation_id: str) -> None:
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="对话不存在")
    conv = conversations[conversation_id]
    if not isinstance(conv.get("case_state"), dict):
        conv["case_state"] = default_case_state()
    else:
        # 兼容旧数据：缺字段则补齐
        base = default_case_state()
        base.update(conv["case_state"])
        conv["case_state"] = base
        # 深层兼容：fault_context 需要按字段合并
        if not isinstance(conv["case_state"].get("fault_context"), dict):
            conv["case_state"]["fault_context"] = default_case_state()["fault_context"]
        else:
            fc_base = default_case_state()["fault_context"].copy()
            fc_base.update(conv["case_state"]["fault_context"])
            conv["case_state"]["fault_context"] = fc_base
        if not isinstance(conv["case_state"].get("draft"), dict):
            conv["case_state"]["draft"] = default_case_state()["draft"]
        else:
            d_base = default_case_state()["draft"].copy()
            d_base.update(conv["case_state"]["draft"])
            if not isinstance(d_base.get("fault_context"), dict):
                d_base["fault_context"] = default_case_state()["draft"]["fault_context"]
            else:
                dfc = default_case_state()["draft"]["fault_context"].copy()
                dfc.update(d_base["fault_context"])
                d_base["fault_context"] = dfc
            conv["case_state"]["draft"] = d_base

        # 兼容前后端不同阶段状态结构：统一补齐三段措施列表
        conv["case_state"]["maintenance_measures"] = conv["case_state"].get("maintenance_measures") or []
        conv["case_state"]["ready_measures"] = conv["case_state"].get("ready_measures") or []
        conv["case_state"]["confirmed_measures"] = conv["case_state"].get("confirmed_measures") or []

        # 对旧数据做轻量归一化，保证读取与渲染稳定
        normalized_new = []
        for it in conv["case_state"]["maintenance_measures"]:
            if not isinstance(it, dict):
                continue
            text = str(it.get("text") or "").strip()
            if not text:
                continue
            normalized_new.append({
                "id": str(it.get("id") or uuid.uuid4()),
                "text": text,
                "created_at": str(it.get("created_at") or _now_iso()),
                "status": "new",
            })
        conv["case_state"]["maintenance_measures"] = normalized_new

        normalized_ready = []
        for it in conv["case_state"]["ready_measures"]:
            if not isinstance(it, dict):
                continue
            text = str(it.get("text") or "").strip()
            if not text:
                continue
            normalized_ready.append({
                "id": str(it.get("id") or uuid.uuid4()),
                "text": text,
                "created_at": str(it.get("created_at") or _now_iso()),
                "confirmed_at": str(it.get("confirmed_at") or _now_iso()),
                "status": "confirmed",
            })
        conv["case_state"]["ready_measures"] = normalized_ready

        normalized_executed = []
        for it in conv["case_state"]["confirmed_measures"]:
            if not isinstance(it, dict):
                continue
            text = str(it.get("text") or "").strip()
            if not text:
                continue
            normalized_executed.append({
                "id": str(it.get("id") or uuid.uuid4()),
                "text": text,
                "created_at": str(it.get("created_at") or _now_iso()),
                "confirmed_at": str(it.get("confirmed_at") or it.get("created_at") or _now_iso()),
                "executed_at": str(it.get("executed_at") or it.get("created_at") or _now_iso()),
                "status": "executed",
            })
        conv["case_state"]["confirmed_measures"] = normalized_executed


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

# 对话管理
conversations: Dict[str, Dict] = {}

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

# 工具函数
def generate_conversation_title(first_message: str) -> str:
    """根据第一条消息生成对话标题"""
    return first_message[:30] + "..." if len(first_message) > 30 else first_message

def save_conversation(conversation_id: str):
    """保存对话到文件"""
    if not os.path.exists("conversations"):
        os.makedirs("conversations")
    
    file_path = f"conversations/{conversation_id}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(conversations[conversation_id], f, ensure_ascii=False, indent=2)

def load_conversations():
    """从文件加载对话"""
    global conversations
    if not os.path.exists("conversations"):
        return
    
    for filename in os.listdir("conversations"):
        if filename.endswith(".json"):
            conversation_id = filename.replace(".json", "")
            file_path = f"conversations/{filename}"
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    conversations[conversation_id] = json.load(f)
                # 补齐 case_state，保持对旧数据兼容
                try:
                    ensure_conversation_case_state(conversation_id)
                except Exception:
                    pass
            except Exception as e:
                print(f"加载对话失败 {filename}: {e}")

# 加载现有对话
load_conversations()

# API端点
@app.post("/api/conversations", response_model=Conversation)
def create_conversation(request: CreateConversationRequest):
    """创建新对话"""
    conversation_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    
    conversation = {
        "id": conversation_id,
        "title": request.title,
        "messages": [],
        "created_at": now,
        "updated_at": now,
        "case_state": default_case_state(),
    }
    
    conversations[conversation_id] = conversation
    save_conversation(conversation_id)
    
    return Conversation(**conversation)

@app.get("/api/conversations", response_model=ConversationList)
def list_conversations():
    """获取所有对话列表"""
    return ConversationList(
        conversations=[Conversation(**conv) for conv in conversations.values()]
    )

@app.get("/api/conversations/{conversation_id}", response_model=Conversation)
def get_conversation(conversation_id: str):
    """获取单个对话详情"""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="对话不存在")

    ensure_conversation_case_state(conversation_id)
    return Conversation(**conversations[conversation_id])

@app.post("/api/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """发送查询并获取回答"""
    # 如果没有指定对话ID，创建新对话
    if not request.conversation_id:
        conversation_id = str(uuid.uuid4())
        title = generate_conversation_title(request.message)
        now = datetime.now().isoformat()
        
        conversations[conversation_id] = {
            "id": conversation_id,
            "title": title,
            "messages": [],
            "created_at": now,
            "updated_at": now,
            "case_state": default_case_state(),
        }
    else:
        conversation_id = request.conversation_id
        if conversation_id not in conversations:
            raise HTTPException(status_code=404, detail="对话不存在")

    ensure_conversation_case_state(conversation_id)
    
    # 添加用户消息
    user_message = Message(
        role="user",
        content=request.message,
        timestamp=datetime.now().isoformat()
    )
    conversations[conversation_id]["messages"].append(user_message.model_dump())
    
    # 获取对话历史（排除刚刚添加的用户消息）
    conversation_history = conversations[conversation_id]["messages"][:-1]
    # 注入“已确认故障上下文”，用于约束问答范围
    sys_note = _build_fault_context_system_note(conversations[conversation_id].get("case_state") or {})
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
    conversations[conversation_id]["messages"].append(assistant_message.model_dump())
    
    # 更新对话时间
    conversations[conversation_id]["updated_at"] = datetime.now().isoformat()
    
    # 保存对话
    save_conversation(conversation_id)

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
    # 如果没有指定对话ID，创建新对话
    if not request.conversation_id:
        conversation_id = str(uuid.uuid4())
        title = generate_conversation_title(request.message)
        now = datetime.now().isoformat()
        
        conversations[conversation_id] = {
            "id": conversation_id,
            "title": title,
            "messages": [],
            "created_at": now,
            "updated_at": now,
            "case_state": default_case_state(),
        }
    else:
        conversation_id = request.conversation_id
        if conversation_id not in conversations:
            raise HTTPException(status_code=404, detail="对话不存在")

    ensure_conversation_case_state(conversation_id)
    
    # 添加用户消息
    user_message = Message(
        role="user",
        content=request.message,
        timestamp=datetime.now().isoformat()
    )
    conversations[conversation_id]["messages"].append(user_message.model_dump())
    
    # 获取对话历史（排除刚刚添加的用户消息）
    conversation_history = conversations[conversation_id]["messages"][:-1]
    
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
        conversations[conversation_id]["messages"].append(assistant_message.model_dump())
        
        # 更新对话时间
        conversations[conversation_id]["updated_at"] = datetime.now().isoformat()
        
        # 保存对话
        save_conversation(conversation_id)

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


# ========= 维修过程记录 API =========
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
    return CaseStateResponse(success=True, conversation_id=conversation_id, case_state=conversations[conversation_id]["case_state"])


@app.put("/api/conversations/{conversation_id}/case-state", response_model=CaseStateResponse)
def update_case_state(conversation_id: str, request: CaseStateUpdateRequest):
    ensure_conversation_case_state(conversation_id)
    if not isinstance(request.case_state, dict):
        raise HTTPException(status_code=400, detail="case_state 必须是对象")

    base = default_case_state()
    # 只允许更新白名单字段，避免前端误写导致结构污染
    allowed_keys = set(base.keys())
    incoming = {k: v for k, v in request.case_state.items() if k in allowed_keys}

    merged = {**conversations[conversation_id]["case_state"], **incoming}
    merged["updated_at"] = _now_iso()
    conversations[conversation_id]["case_state"] = merged
    conversations[conversation_id]["updated_at"] = _now_iso()
    save_conversation(conversation_id)
    return CaseStateResponse(success=True, conversation_id=conversation_id, case_state=merged)


def _format_conversation_messages_for_record(messages: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for m in messages or []:
        role = m.get("role", "")
        role_cn = "用户" if role == "user" else "助手"
        ts = m.get("timestamp", "")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if ts:
            lines.append(f"- **{role_cn}**（{ts}）: {content}")
        else:
            lines.append(f"- **{role_cn}**: {content}")
    return "\n".join(lines)

def _build_fault_context_system_note(case_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    将用户确认的故障上下文注入到问答上下文中，避免模型把整段对话当成“已确认事实”。
    """
    fc = (case_state or {}).get("fault_context") or {}
    if not isinstance(fc, dict) or not fc.get("confirmed"):
        return None

    equipment = (fc.get("equipment") or "").strip()
    fault_summary = (fc.get("fault_summary") or "").strip()
    phenomenon = (fc.get("phenomenon") or "").strip()
    confirmed_at = (fc.get("confirmed_at") or "").strip()

    lines = [
        "【已确认的本次故障上下文（用户确认）】",
        f"- 设备/机器：{equipment or '（未填写）'}",
        f"- 故障概述：{fault_summary or '（未填写）'}",
        f"- 故障现象：{phenomenon or '（未填写）'}",
    ]
    if confirmed_at:
        lines.append(f"- 确认时间：{confirmed_at}")
    lines += [
        "",
        "约束：回答与建议必须围绕以上“已确认故障上下文”。",
        "若用户对话中出现与该故障无关的信息，需明确标注“与本次故障关联性不足/待确认”，不要当作既定事实。",
    ]

    return {
        "role": "system",
        "content": "\n".join(lines).strip(),
        "timestamp": _now_iso(),
    }


def _is_model_refusal(text: str) -> bool:
    """
    判断模型是否拒答/无法回答。用于拦截草案自动生成。
    """
    t = (text or "").strip()
    if not t:
        return True
    markers = [
        "无法回答", "我不能", "不能回答", "抱歉", "无法提供", "无法处理",
        "没有足够的信息", "根据检索到的信息，我无法回答这个问题",
        "I can't", "cannot answer", "insufficient information",
    ]
    hit = sum(1 for m in markers if m.lower() in t.lower())
    # 命中多个拒答特征，或文本过短时视为拒答
    return hit >= 2 or len(t) < 20


def _clear_case_draft(conversation_id: str) -> None:
    """
    清空草案，避免拒答后显示过期/误导性草案。
    """
    ensure_conversation_case_state(conversation_id)
    conv = conversations[conversation_id]
    conv["case_state"]["draft"] = default_case_state()["draft"]
    conv["case_state"]["updated_at"] = _now_iso()
    conv["updated_at"] = _now_iso()
    save_conversation(conversation_id)

def _format_conversation_messages_for_llm(messages: List[Dict[str, Any]], max_chars: int = 4000) -> str:
    """
    将对话消息格式化为给 LLM 的文本（做字符数截断避免提示词过长）。
    """
    raw = _format_conversation_messages_for_record(messages)
    raw = raw.strip()
    if len(raw) <= max_chars:
        return raw
    return raw[-max_chars:]

def _safe_parse_keywords(text: str, max_items: int = 10) -> List[str]:
    """
    尝试从模型输出中解析关键词（优先 JSON 数组；失败则从文本提取逗号/换行分隔词）。
    """
    import re
    if not text:
        return []
    text = text.strip()
    try:
        import json
        obj = json.loads(text)
        if isinstance(obj, list):
            items = [str(x).strip() for x in obj if str(x).strip()]
            # 去重（保持顺序）
            out = []
            seen = set()
            for it in items:
                if it not in seen:
                    seen.add(it)
                    out.append(it)
            return out[:max_items]
    except Exception:
        pass

    # 兜底：提取类似 "a, b, c" 或 "a\nb\nc"
    parts = re.split(r"[,\n;；，]\s*", text)
    items = [p.strip().strip('"').strip("'") for p in parts if p.strip()]
    # 去重
    out = []
    seen = set()
    for it in items:
        if it and it not in seen:
            seen.add(it)
            out.append(it)
    return out[:max_items]

def _safe_parse_case_draft_json(text: str) -> Dict[str, Any]:
    """
    解析模型返回的草案 JSON，失败时返回空草案结构。
    """
    empty = {
        "fault_context": {
            "equipment": "",
            "fault_summary": "",
            "phenomenon": "",
        },
        "suggested_measures": [],
        "suggested_todos": [],
        "suggested_components": [],
        "keywords": [],
    }
    if not text:
        return empty
    raw = text.strip()
    try:
        obj = json.loads(raw)
    except Exception:
        # 兜底：截取最外层 JSON 对象
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return empty
        try:
            obj = json.loads(raw[start:end + 1])
        except Exception:
            return empty

    if not isinstance(obj, dict):
        return empty

    fc = obj.get("fault_context") or {}
    if not isinstance(fc, dict):
        fc = {}

    def _clean_str(v: Any) -> str:
        return str(v).strip() if v is not None else ""

    def _clean_list(v: Any, max_items: int = 12) -> List[str]:
        if not isinstance(v, list):
            return []
        out: List[str] = []
        for item in v:
            s = _clean_str(item)
            if s and s not in out:
                out.append(s)
        return out[:max_items]

    draft = {
        "fault_context": {
            "equipment": _clean_str(fc.get("equipment")),
            "fault_summary": _clean_str(fc.get("fault_summary")),
            "phenomenon": _clean_str(fc.get("phenomenon")),
        },
        "suggested_measures": _clean_list(obj.get("suggested_measures")),
        "suggested_todos": _clean_list(obj.get("suggested_todos")),
        "suggested_components": _clean_list(obj.get("suggested_components")),
        "keywords": _clean_list(obj.get("keywords"), max_items=10),
    }
    # 关键词规范成“XXX故障”
    norm_kw: List[str] = []
    for k in draft["keywords"]:
        kk = k if k.endswith("故障") else f"{k}故障"
        if kk not in norm_kw:
            norm_kw.append(kk)
    draft["keywords"] = norm_kw[:10]
    return draft


def extract_keywords_from_conversation(conversation_id: str) -> List[str]:
    """
    从对话中提取船舶维修相关关键词，并写回 case_state.keywords。
    """
    ensure_conversation_case_state(conversation_id)
    conv = conversations[conversation_id]
    state = conv["case_state"]

    # 若已有关键词，直接复用
    existing = state.get("keywords")
    if isinstance(existing, list) and existing:
        return existing

    messages = conv.get("messages", []) or []
    conv_text = _format_conversation_messages_for_llm(messages)

    prompt = "\n".join([
        "你是一名船舶维修领域的资深工程师/知识管理专家。",
        "请从下面的对话内容中提取与故障定位、维修措施、预防改进相关的关键词。",
        "关键词格式要求：每个关键词必须严格为“XXX故障”（主体+故障），不要输出其它格式。",
        "例如：发电机无法建压故障、旋转方向错误故障、蓄电池老化故障、励磁回路开路故障、外部负载短路故障。",
        "输出要求：",
        "1) 仅输出一个 JSON 数组（如：['关键词1','关键词2']），不要输出多余解释。",
        "2) 关键词数量 5-10 个，每个元素都必须以“故障”结尾。",
        "3) 尽量用对话里真实出现/明确指向的主体名称或故障描述（不要胡编新的设备/部件）。",
        "",
        "对话内容：",
        conv_text,
    ])

    keywords = []
    try:
        raw = _generate_freeform_text(prompt)
        keywords = _safe_parse_keywords(raw, max_items=10)
    except Exception:
        keywords = []

    # 本地归一化：保证都以“故障”结尾，避免模型漏写
    normalized: List[str] = []
    for k in keywords:
        kk = (k or "").strip().strip('"').strip("'")
        if not kk:
            continue
        if not kk.endswith("故障"):
            kk = kk + "故障"
        # 去重（保持顺序）
        if kk not in normalized:
            normalized.append(kk)
    keywords = normalized[:10]

    if not keywords:
        keywords = ["故障定位故障", "维修措施故障", "风险控制故障"]

    state["keywords"] = keywords
    state["updated_at"] = _now_iso()
    conv["case_state"] = state
    conv["updated_at"] = _now_iso()
    save_conversation(conversation_id)
    return keywords


def generate_case_draft_from_conversation(conversation_id: str) -> Dict[str, Any]:
    """
    使用 LLM 从当前对话生成“待用户确认”的故障草案。
    注意：草案不会自动作为事实生效，需用户在前端确认/采纳。
    """
    ensure_conversation_case_state(conversation_id)
    conv = conversations[conversation_id]
    state = conv["case_state"]
    messages = conv.get("messages", []) or []
    conv_text = _format_conversation_messages_for_llm(messages, max_chars=5000)

    prompt = "\n".join([
        "你是船舶维修领域助手。请基于对话内容抽取“待用户确认”的故障草案。",
        "严格要求：",
        "1) 仅输出一个 JSON 对象，不要输出其它解释。",
        "2) 只提取对话中明确提及/高置信的信息，不要臆造。",
        "3) keywords 每项必须是“XXX故障”格式。",
        "",
        "JSON Schema：",
        "{",
        '  "fault_context": {',
        '    "equipment": "故障设备/机器",',
        '    "fault_summary": "故障概述(一句话)",',
        '    "phenomenon": "故障现象(可空)"',
        "  },",
        '  "suggested_measures": ["建议维修措施1","建议维修措施2"],',
        '  "suggested_todos": ["建议待处理事项1","建议待处理事项2"],',
        '  "suggested_components": ["建议故障部件1","建议故障部件2"],',
        '  "keywords": ["XXX故障","YYY故障"]',
        "}",
        "",
        "对话内容：",
        conv_text,
    ])

    raw = _generate_freeform_text(prompt)
    draft = _safe_parse_case_draft_json(raw)
    draft["generated_at"] = _now_iso()

    state["draft"] = draft
    state["updated_at"] = _now_iso()
    conv["case_state"] = state
    conv["updated_at"] = _now_iso()
    save_conversation(conversation_id)
    return draft


def extract_failed_components_from_conversation(conversation_id: str) -> List[str]:
    """
    从对话中提取“故障发生的具体部件/零部件/子系统”。
    """
    ensure_conversation_case_state(conversation_id)
    conv = conversations[conversation_id]
    state = conv["case_state"]

    existing = state.get("failed_components")
    if isinstance(existing, list) and existing:
        return existing

    messages = conv.get("messages", []) or []
    conv_text = _format_conversation_messages_for_llm(messages)

    prompt = "\n".join([
        "你是一名船舶维修工程师与故障管理专家。",
        "请从以下对话内容中提取故障涉及的具体部件/零部件/子系统（尽量具体到可维护的对象，例如：泵/阀/轴承/控制器/电机绕组/传感器等）。",
        "输出要求：",
        "1) 仅输出一个 JSON 数组，例如：['主机燃油喷嘴','增压器转子']",
        "2) 数量 3-8 个，去重，短语即可。",
        "",
        "对话内容：",
        conv_text,
    ])

    components: List[str] = []
    try:
        raw = _generate_freeform_text(prompt)
        components = _safe_parse_keywords(raw, max_items=8)
    except Exception:
        components = []

    if not components:
        components = ["（待从对话中识别的部件）"]

    state["failed_components"] = components
    state["updated_at"] = _now_iso()
    conv["case_state"] = state
    conv["updated_at"] = _now_iso()
    save_conversation(conversation_id)
    return components


def _get_executed_measures_structured(case_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    “已执行的具体措施”列表：来自 confirmed_measures。
    """
    confirmed = case_state.get("confirmed_measures") or []
    # 按创建时间排序，保证 step_no 与实际录入/执行的先后一致
    confirmed_sorted = sorted(confirmed, key=lambda x: x.get("created_at") or "")
    out: List[Dict[str, Any]] = []
    for idx, it in enumerate(confirmed_sorted):
        text = (it.get("text") or "").strip()
        if not text:
            continue
        out.append({
            "step_no": idx + 1,
            "id": it.get("id") or "",
            "text": text,
            "created_at": it.get("created_at") or "",
        })
    return out


def _get_pending_measures_structured(case_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    “待处理（维修措施）”列表：来自 maintenance_measures（未确认）。
    """
    pending = case_state.get("maintenance_measures") or []
    # 按创建时间排序，保证 step_no 与实际录入/计划的先后一致
    pending_sorted = sorted(pending, key=lambda x: x.get("created_at") or "")
    out: List[Dict[str, Any]] = []
    for idx, it in enumerate(pending_sorted):
        text = (it.get("text") or "").strip()
        if not text:
            continue
        out.append({
            "step_no": idx + 1,
            "id": it.get("id") or "",
            "text": text,
            "created_at": it.get("created_at") or "",
        })
    return out


def _generate_freeform_text(prompt: str) -> str:
    """
    生成不依赖检索文档的自由文本（用于复盘等）。
    复用 GenerationIntegrationModule 当前的 provider 配置。
    """
    if not rag_system or not getattr(rag_system, "generation_module", None):
        return ""
    gm = rag_system.generation_module
    try:
        calculated_max_tokens = getattr(gm, "_calculate_max_tokens", lambda _: 1024)(prompt)
        temperature = getattr(gm, "temperature", 0.3)

        if getattr(gm, "llm_provider", "") == "ollama":
            import requests
            url = f"{gm.ollama_base_url.rstrip('/')}/api/chat"
            payload = {
                "model": gm.ollama_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": calculated_max_tokens,
                },
            }
            resp = requests.post(url, json=payload, timeout=90)
            resp.raise_for_status()
            data = resp.json()
            return ((data.get("message") or {}).get("content") or "").strip()

        client = getattr(gm, "client", None)
        if not client:
            return ""
        response = client.chat.completions.create(
            model=getattr(gm, "model_name", ""),
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=calculated_max_tokens,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception:
        return ""


@app.post("/api/conversations/{conversation_id}/case-state/generate-maintenance-record", response_model=CaseStateResponse)
def generate_maintenance_record(conversation_id: str):
    ensure_conversation_case_state(conversation_id)
    conv = conversations[conversation_id]
    state = conv["case_state"]

    measures = state.get("maintenance_measures") or []
    todos = state.get("todo") or []
    confirmed = state.get("confirmed_measures") or []

    executed_steps = _get_executed_measures_structured(state)
    pending_steps = _get_pending_measures_structured(state)

    # 维护记录日期：使用生成当日
    record_date = datetime.now().date().isoformat()

    def _list_items(items, key="text"):
        out = []
        for it in items:
            text = (it.get(key) or "").strip()
            if text:
                out.append(f"- {text}")
        return "\n".join(out) if out else "- （无）"

    def _fmt_steps(steps: List[Dict[str, Any]]) -> str:
        out = []
        for s in steps:
            text = (s.get("text") or "").strip()
            step_no = s.get("step_no")
            if not text:
                continue
            if step_no:
                out.append(f"{step_no}. {text}")
            else:
                out.append(f"- {text}")
        return "\n".join(out) if out else "- （无）"

    record_md = "\n".join([
        "## 维修记录",
        f"- **对话ID**：{conversation_id}",
        f"- **记录日期**：{record_date}",
        f"- **生成时间**：{_now_iso()}",
        "",
        "### 已执行的具体措施",
        _fmt_steps(executed_steps),
        "",
        "### 待处理（未确认的维修措施）",
        _fmt_steps(pending_steps),
        "",
        "### 待处理事项（补充）",
        _list_items([t for t in todos if not t.get('done')]),
        "",
        "### 对话要点（原始记录）",
        _format_conversation_messages_for_record(conv.get("messages", [])),
        "",
    ]).strip() + "\n"

    state["maintenance_record"] = record_md
    state["status"] = "ended"
    state["updated_at"] = _now_iso()
    conv["case_state"] = state
    conv["updated_at"] = _now_iso()
    save_conversation(conversation_id)
    return CaseStateResponse(success=True, conversation_id=conversation_id, case_state=state)


@app.post("/api/conversations/{conversation_id}/case-state/generate-postmortem", response_model=CaseStateResponse)
def generate_postmortem(conversation_id: str):
    ensure_conversation_case_state(conversation_id)
    conv = conversations[conversation_id]
    state = conv["case_state"]

    record = (state.get("maintenance_record") or "").strip()
    if not record:
        # 先生成维修记录，保证复盘有输入
        generate_maintenance_record(conversation_id)
        state = conv["case_state"]
        record = (state.get("maintenance_record") or "").strip()

    # 先提取关键词，帮助后续“预防与改进”更聚焦
    try:
        keywords = extract_keywords_from_conversation(conversation_id)
    except Exception:
        keywords = state.get("keywords") or []

    # 提取故障具体部件，便于复盘时做“部件-措施-预防”的关联
    try:
        failed_components = extract_failed_components_from_conversation(conversation_id)
    except Exception:
        failed_components = state.get("failed_components") or []

    executed_measures = _get_executed_measures_structured(state)
    pending_measures = _get_pending_measures_structured(state)

    # 优先用 LLM 生成复盘；失败则用模板兜底
    postmortem_md = ""
    try:
        full_conv_text = _format_conversation_messages_for_llm(conv.get("messages", []) or [])
        kw_text = ", ".join(keywords) if isinstance(keywords, list) else str(keywords)

        commutation_keywords = [
            "电刷", "换向器", "火花", "电弧", "电刷压力", "接触", "积碳", "油污", "跳动",
            "冒火花", "磨损", "烧蚀", "烧伤", "凹凸", "表面", "触点",
            "电刷座", "电刷架", "弹簧",
        ]

        def _is_commutation_related(text: str) -> bool:
            t = (text or "").strip()
            if not t:
                return False
            for k in commutation_keywords:
                if k in t:
                    return True
            return False

        # 如果对话本身没有出现换向相关的关键词，则不允许模型把换向器/电刷/火花等当作故障点提出
        commutation_present_in_conv = any((k and k in full_conv_text) for k in commutation_keywords)
        commutation_prompt_rule = ""
        if not commutation_present_in_conv:
            commutation_prompt_rule = "若对话中未出现与“换向器/电刷/火花/电弧”等相关的关键词，则不要在复盘中把这些内容作为可能故障点或预防重点；表格“对应故障部件/证据”应写“待补充/不直接关联”。"

        # 将结构化数据转成“纯文本要点”，避免模型把 JSON 当作输出内容
        executed_steps_text = ""
        if executed_measures:
            executed_steps_text = "\n".join([
                f"- step_no {s.get('step_no')}: {(s.get('text') or '').strip()}"
                + ("（疑似不相关：未命中关键诊断关键词）" if not _is_commutation_related(s.get("text") or "") else "（疑似相关）")
                for s in executed_measures
            ])
        else:
            executed_steps_text = "- （无已执行措施）"

        pending_steps_text = ""
        if pending_measures:
            pending_steps_text = "\n".join([
                f"- step_no {s.get('step_no')}: {(s.get('text') or '').strip()}"
                + ("（疑似不相关：未命中关键诊断关键词）" if not _is_commutation_related(s.get("text") or "") else "（疑似相关）")
                for s in pending_measures
            ])
        else:
            pending_steps_text = "- （无待处理维修措施）"

        components_text = ""
        if failed_components:
            components_text = "\n".join([f"- {c}" for c in failed_components])
        else:
            components_text = "- （待从对话识别）"

        prompt = "\n".join([
            "你是一名船舶维修领域的资深工程师，同时具备故障管理与质量改进经验。",
            "请对下面的“对话整体内容”进行事后复盘，并输出面向未来的预防改进方案。",
            "要求：",
            "1) 只输出指定的 Markdown 模板内容，不要额外添加“适用场景/核心问题/总结优化版”等其它段落。",
            "2) 每个小标题必须完整出现：### 1..7（按序号），不要跳号。",
            "3) “故障部件-措施关联”必须基于 step_no 的已执行措施逐条生成表格行，无法判断时填“待补充”。",
            "4) 严禁输出任何 JSON（包括不要输出类似 [ {...} ] 的结构，也不要把输入 JSON 原样复述）。",
            "5) 严禁输出代码块（不要用 ```）。",
            "",
            commutation_prompt_rule,
            "复盘目标：",
            "1) 总结故障现象与业务/安全影响（根据对话内容推断，不要胡编）",
            "2) 做根因分析（建议使用5Why/鱼骨逻辑）",
            "3) 回顾维修措施的执行过程与效果（关联维修记录）",
            "4) 给出遗留风险与不确定性",
            "5) 提出未来预防与改进：流程/备件/培训/检查监测点/责任人/触发条件",
            "",
            "输出格式（Markdown）：",
            "## 复盘结果",
            "### 1. 现象与影响",
            "- （按对话内容写，尽量量化/可观测指标；没有则写“待补充”）",
            "",
            "### 2. 根因分析（5Why/鱼骨）",
            "- 5Why/鱼骨：按层级列出（不要虚构；证据不足则写“待补充”）",
            "",
            "### 3. 维修过程回顾与效果",
            "- 简述：做了哪些关键动作",
            "- 效果：对故障/指标的改善情况（待补充/无法确认则说明）",
            "",
            "### 4. 遗留问题与风险",
            "- 遗留：尚未完全解决的点",
            "- 风险：可能复发原因/不确定性",
            "",
            "### 5. 未来预防与改进（务必可执行）",
            "- 流程改进：谁/何时/触发条件/检查点",
            "- 备件与工艺：需更换/需纳入标准的项",
            "- 培训与宣贯：需要覆盖的知识点",
            "- 监测与复检：建议监测频率/指标/判据",
            "",
            "### 6. 故障部件-措施关联（必须逐条引用“已执行的具体措施”，按 step_no 对应）",
            "请输出 Markdown 表格，至少包含列：",
            "| step_no | 已执行措施 | 对应故障部件 | 预计效果/验证点 | 证据 |",
            "表格行规则：",
            "- 必须为每一个已执行措施 step_no 生成一行（同一步可能关联多个部件也可在单元格写“部件A/部件B”）",
            "- 如果“已执行措施”文本与“故障部件（部件列表）”明显不匹配，允许把“对应故障部件”写为“待补充/不直接关联”，并在“证据”写：该措施与本次故障主题关联性不足（待补充）。",
            "- “对应故障部件”优先从“故障部件（从对话提取）”列表里选择；若无对应或文本关联性不足，写“待补充”。",
            "- 证据不足写“待补充”",
            "- 若某条措施在“已执行的具体措施”输入里带有“疑似不相关”标记，则该行必须写：对应故障部件=“不直接关联/待补充”，证据=“措施与本次故障主题关联性不足（待补充）”。",
            "",
            "### 7. 关键关键词（用于后续检索/知识复用）",
            "- 关键词列表（来自自动提取的关键词 kw）",
            "",
            "示范格式（仅示例结构，不要照搬内容）：",
            "## 复盘结果",
            "### 1. 现象与影响",
            "- 例：XXX导致XXX（待补充）",
            "",
            "### 2. 根因分析（5Why/鱼骨）",
            "- 例：5Why：……（待补充）",
            "",
            "### 6. 故障部件-措施关联",
            "| step_no | 已执行措施 | 对应故障部件 | 预计效果/验证点 | 证据 |",
            "| 1 | 例：更换…… | 例：轴承 | 例：振动降低 | 待补充 |",
            "",
            "关键词（用于聚焦输出）：",
            kw_text,
            "",
            "=== 故障部件（从对话提取） ===",
            components_text,
            "",
            "=== 对话整体内容（用于推断现象/根因/效果） ===",
            full_conv_text,
            "",
            "=== 当前维修记录（卡片生成，供关联执行过程） ===",
            record,
            "",
            "=== 已执行的具体措施（用于 step_no 表格；不要新增措施；不要输出 JSON） ===",
            executed_steps_text,
            "",
            "=== 待处理（未确认）的维修措施（可用于“遗留问题与风险”） ===",
            pending_steps_text,
        ])
        postmortem_md = _generate_freeform_text(prompt)
    except Exception:
        postmortem_md = ""

    if not postmortem_md:
        postmortem_md = "\n".join([
            "## 复盘结果",
            "",
            "### 1. 现象与影响",
            "- （待补充）",
            "",
            "### 2. 根因分析",
            "- （待补充）",
            "",
            "### 3. 已采取措施与效果",
            "- （待补充）",
            "",
            "### 4. 遗留问题与风险",
            "- （待补充）",
            "",
            "### 5. 预防与改进",
            "- （待补充）",
            "",
            "### 6. 故障部件-措施关联",
            "- （待补充）",
            "",
            "### 7. 关键关键词（用于后续检索/知识复用）",
            "- （待补充）",
        ]).strip() + "\n"

    # 在复盘末尾强制补充日期（确保你要求的“最后加日期”）
    post_date = datetime.now().date().isoformat()
    postmortem_md = postmortem_md.strip() + f"\n\n---\n日期：{post_date}\n"

    state["postmortem"] = postmortem_md
    state["updated_at"] = _now_iso()
    conv["case_state"] = state
    conv["updated_at"] = _now_iso()
    save_conversation(conversation_id)
    return CaseStateResponse(success=True, conversation_id=conversation_id, case_state=state)


@app.post("/api/conversations/{conversation_id}/case-state/generate-keywords", response_model=CaseStateResponse)
def generate_case_keywords(conversation_id: str):
    """
    为当前对话提取“维修/预防”关键词，并写入 case_state.keywords。
    """
    ensure_conversation_case_state(conversation_id)
    keywords = extract_keywords_from_conversation(conversation_id)
    return CaseStateResponse(
        success=True,
        conversation_id=conversation_id,
        case_state=conversations[conversation_id]["case_state"],
    )


@app.get("/api/conversations/{conversation_id}/case-state/keywords", response_model=CaseStateResponse)
def get_case_keywords(conversation_id: str):
    """
    获取当前对话的关键词；若尚未生成，则自动生成并返回。
    """
    ensure_conversation_case_state(conversation_id)
    try:
        # 若已存在关键词，extract_keywords_from_conversation 会直接复用
        extract_keywords_from_conversation(conversation_id)
    except Exception:
        pass
    return CaseStateResponse(
        success=True,
        conversation_id=conversation_id,
        case_state=conversations[conversation_id]["case_state"],
    )


@app.post("/api/conversations/{conversation_id}/case-state/generate-draft", response_model=CaseStateResponse)
def generate_case_draft(conversation_id: str):
    """
    生成“待用户确认”的故障草案（故障范围/建议措施/待处理/关键词）。
    """
    ensure_conversation_case_state(conversation_id)
    generate_case_draft_from_conversation(conversation_id)
    return CaseStateResponse(
        success=True,
        conversation_id=conversation_id,
        case_state=conversations[conversation_id]["case_state"],
    )


@app.post("/api/conversations/{conversation_id}/case-state/apply-draft", response_model=CaseStateResponse)
def apply_case_draft(conversation_id: str, request: ApplyDraftRequest):
    """
    接受（可部分拒绝后的）草案内容，合并到正式 case_state。
    """
    ensure_conversation_case_state(conversation_id)
    conv = conversations[conversation_id]
    state = conv["case_state"]
    draft = state.get("draft") or {}

    def _norm_list(items: Optional[List[str]], to_fault=False) -> List[str]:
        out: List[str] = []
        for it in (items or []):
            s = str(it).strip()
            if not s:
                continue
            if to_fault and not s.endswith("故障"):
                s = f"{s}故障"
            if s not in out:
                out.append(s)
        return out

    # 1) 确认故障范围（以请求优先，其次使用 draft）
    fc_req = request.fault_context if isinstance(request.fault_context, dict) else {}
    fc_src = (draft.get("fault_context") if isinstance(draft.get("fault_context"), dict) else {}) or {}
    equipment = str((fc_req.get("equipment") if fc_req else fc_src.get("equipment")) or "").strip()
    fault_summary = str((fc_req.get("fault_summary") if fc_req else fc_src.get("fault_summary")) or "").strip()
    phenomenon = str((fc_req.get("phenomenon") if fc_req else fc_src.get("phenomenon")) or "").strip()

    state["fault_context"] = state.get("fault_context") or {}
    state["fault_context"]["equipment"] = equipment
    state["fault_context"]["fault_summary"] = fault_summary
    state["fault_context"]["phenomenon"] = phenomenon
    if request.confirm_fault_context:
        state["fault_context"]["confirmed"] = bool(equipment and fault_summary)
        state["fault_context"]["confirmed_at"] = _now_iso() if state["fault_context"]["confirmed"] else ""
    else:
        # 仅填充范围信息，不自动确认
        state["fault_context"]["confirmed"] = bool(state["fault_context"].get("confirmed", False) and equipment and fault_summary)
        if not state["fault_context"]["confirmed"]:
            state["fault_context"]["confirmed_at"] = ""

    # 2) 接受措施（进入待处理维修措施）
    accepted_measures = _norm_list(
        request.measures if request.measures is not None else draft.get("suggested_measures") or []
    )
    state["maintenance_measures"] = state.get("maintenance_measures") or []
    existing_m = {str((x.get("text") or "")).strip() for x in state["maintenance_measures"]}
    for m in accepted_measures:
        if m in existing_m:
            continue
        state["maintenance_measures"].append({
            "id": str(uuid.uuid4()),
            "text": m,
            "created_at": _now_iso(),
            "status": "new",
            "confirmed": False,
        })
        existing_m.add(m)

    # 3) 接受待处理事项
    accepted_todos = _norm_list(
        request.todos if request.todos is not None else draft.get("suggested_todos") or []
    )
    state["todo"] = state.get("todo") or []
    existing_t = {str((x.get("text") or "")).strip() for x in state["todo"]}
    for t in accepted_todos:
        if t in existing_t:
            continue
        state["todo"].append({
            "id": str(uuid.uuid4()),
            "text": t,
            "created_at": _now_iso(),
            "done": False,
        })
        existing_t.add(t)

    # 4) 接受关键词和故障部件
    accepted_keywords = _norm_list(
        request.keywords if request.keywords is not None else draft.get("keywords") or [],
        to_fault=True
    )
    if accepted_keywords:
        state["keywords"] = accepted_keywords[:10]

    accepted_components = _norm_list(
        request.components if request.components is not None else (draft.get("suggested_components") or [])
    )
    if accepted_components:
        state["failed_components"] = accepted_components[:20]

    state["updated_at"] = _now_iso()
    conv["case_state"] = state
    conv["updated_at"] = _now_iso()
    save_conversation(conversation_id)
    return CaseStateResponse(success=True, conversation_id=conversation_id, case_state=state)

@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(conversation_id: str):
    """删除对话"""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="对话不存在")
    
    # 删除文件
    file_path = f"conversations/{conversation_id}.json"
    if os.path.exists(file_path):
        os.remove(file_path)
    
    # 从内存中删除
    del conversations[conversation_id]
    
    return {"success": True}

@app.get("/api/export/{conversation_id}", response_model=ExportResponse)
def export_conversation(conversation_id: str):
    """导出对话"""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="对话不存在")
    
    conversation = conversations[conversation_id]
    
    # 生成导出文件
    if not os.path.exists("exports"):
        os.makedirs("exports")
    
    export_file = f"exports/conversation_{conversation_id}.json"
    with open(export_file, "w", encoding="utf-8") as f:
        json.dump(conversation, f, ensure_ascii=False, indent=2)
    
    return ExportResponse(
        success=True,
        url=f"/exports/{os.path.basename(export_file)}"
    )


@app.get("/api/export-md/{conversation_id}", response_model=ExportResponse)
def export_conversation_to_markdown(conversation_id: str):
    """导出对话为 Markdown 文档 (.md)"""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="对话不存在")

    conversation = conversations[conversation_id]

    if not os.path.exists("exports"):
        os.makedirs("exports")

    export_file = f"exports/conversation_{conversation_id}.md"
    conversation_title = conversation.get("title") or f"conversation_{conversation_id}"

    # Build a human-friendly markdown export.
    lines = []
    lines.append(f"# {conversation_title}")
    lines.append("")
    lines.append(f"> 导出时间：{datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"> 对话 ID：{conversation_id}")
    lines.append("")
    lines.append("---")
    lines.append("")

    messages = conversation.get("messages", []) or []
    for i, message in enumerate(messages, start=1):
        role = message.get("role", "assistant")
        content = message.get("content", "") or ""
        timestamp = message.get("timestamp", "") or ""

        role_cn = "用户" if role == "user" else "助手"
        lines.append(f"## {i}. {role_cn}")
        if timestamp:
            lines.append(f"**时间**：{timestamp}")
        lines.append("")
        # content is already markdown-ish (the UI uses it as markdown), so we keep it as-is.
        lines.append(content)
        lines.append("")
        lines.append("---")
        lines.append("")

    # Use UTF-8-SIG for better Windows Word/Markdown app compatibility.
    with open(export_file, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(lines).rstrip() + "\n")

    return ExportResponse(
        success=True,
        url=f"/exports/{os.path.basename(export_file)}"
    )

@app.get("/api/system/stats")
def get_system_stats():
    """获取系统统计信息"""
    # 调用RAG系统的统计方法
    stats = {
        "conversation_count": len(conversations),
        "total_messages": sum(len(conv["messages"]) for conv in conversations.values())
    }
    
    return stats

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
        
        conversations[conversation_id] = conversation
        save_conversation(conversation_id)
        
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
