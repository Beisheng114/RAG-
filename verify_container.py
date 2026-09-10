"""
容器内全链路验证脚本（在 rag-app 镜像内执行）

阶段1：逐模块 import（编译+导入链）
阶段2：app 装配 + 路由表完整性
阶段3：TestClient 接口冒烟（不启动 lifespan，模拟"外部服务未就绪"的降级态）
        —— 对话 CRUD / case-state 全链路 / 鉴权 / CORS / 导出 / 静态页
阶段4：降级行为确认（query 需启动、admin 未配置 503 等）
"""
import json
import os
import sys
import traceback

sys.path.insert(0, "/app")
os.chdir("/app")
os.environ.pop("RAG_ADMIN_TOKEN", None)
os.environ.pop("RAG_API_KEY", None)
os.environ.pop("RAG_CORS_ORIGINS", None)

PASS, FAIL = [], []


def check(name, fn):
    try:
        result = fn()
        PASS.append(name)
        print(f"[PASS] {name}" + (f"  ({result})" if result else ""))
        return True
    except Exception as e:
        FAIL.append((name, f"{type(e).__name__}: {e}"))
        print(f"[FAIL] {name} -> {type(e).__name__}: {e}")
        if os.getenv("VERIFY_TRACEBACK"):
            traceback.print_exc()
        return False


print("=" * 60)
print("阶段1：逐模块导入（导入链验证）")
print("=" * 60)

modules = [
    "config", "csv_to_neo4j",
    "core.system_context", "core.security", "core.rrf",
    "rag_modules",
    "rag_modules.qdrant_index_construction",
    "rag_modules.reranker",
    "rag_modules.hybrid_retrieval",
    "rag_modules.graph_rag_retrieval",
    "rag_modules.intelligent_query_router",
    "rag_modules.generation_integration",
    "rag_modules.graph_data_insert",
    "ragmain",
    "services.conversation_store",
    "services.conversation_service",
    "services.case_state_service",
    "services.export_service",
    "services.admin_service",
    "services.graph_service",
    "services.kg_import_service",
    "routers.admin_routes",
    "routers.kg_import_routes",
    "routers.graph_routes",
    "routers.page_routes",
    "app",
    "evaluation.run_eval",
]

for m in modules:
    def _imp(m=m):
        __import__(m)
        return "ok"
    check(f"import {m}", _imp)

print()
print("=" * 60)
print("阶段2：app 装配与路由表")
print("=" * 60)

from app import app  # noqa: E402

def _routes():
    """路由完整性用【请求探测】验证：新版 FastAPI(0.115+) 的 include_router
    子路由以 _IncludedRouter 聚合对象存在，不再平铺进 app.routes，
    内省 .path/.methods 会漏检——以实际请求区分路由是否存在。
    注意 graph 路由真实路径为 /api/graph/*（router prefix=/api + 子路径）"""
    from fastapi.testclient import TestClient
    probe = TestClient(app, raise_server_exceptions=False)
    # 先建真实对话，供 apply-draft 探测（404 存在歧义：路由缺失 vs 对话不存在）
    conv_id = probe.post("/api/conversations", json={"title": "route-probe"}).json()["id"]
    probes = [
        ("GET", "/api/conversations", {200}),
        ("POST", "/api/query", {500}),  # 未启动RAG时500，但路由存在
        ("POST", "/api/query/stream", {500}),
        ("GET", "/api/system/stats", {200}),
        ("POST", "/api/import", {422}),  # 缺表单参数→422，路由存在
        ("GET", "/api/material-library", {200}),
        ("GET", "/api/config/preview", {503}),  # admin未配置→503
        ("GET", "/api/admin/dashboard", {503}),
        ("POST", "/api/graph/query", {422}),
        ("GET", "/api/graph/node-counts", {200}),
        (f"POST /api/conversations/{conv_id}/case-state/apply-draft", None, None),  # 占位见下
    ]
    missing = []
    for method, path, ok in probes:
        if path is None:
            continue
        r = probe.request(method, path, json={"message": "t"} if method == "POST" else None)
        if r.status_code == 404:
            missing.append(f"{method} {path}")
    # apply-draft：真实对话 + 空草案 → 200
    r = probe.post(f"/api/conversations/{conv_id}/case-state/apply-draft", json={})
    if r.status_code == 404:
        missing.append("POST /api/conversations/{id}/case-state/apply-draft")
    probe.delete(f"/api/conversations/{conv_id}")
    assert not missing, f"路由缺失: {missing}"
    total = len(app.routes)
    return f"{total} 个注册路由对象（含_IncludedRouter聚合），关键路由请求探测全部命中"

check("路由表完整性（请求探测）", _routes)
check("中间件链", lambda: f"{len(app.user_middleware)} 个（CORS+APIKey）" if len(app.user_middleware) >= 1 else (_ for _ in ()).throw(AssertionError("无中间件")))

print()
print("=" * 60)
print("阶段3：TestClient 接口冒烟（lifespan 未启动的降级态）")
print("=" * 60)

from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(app, raise_server_exceptions=False)
client.raise_server_exceptions = False

# --- 静态页与根路由 ---
check("GET / 返回 SPA 主页面", lambda: (lambda r: (_ for _ in ()).throw(AssertionError(r.status_code)) if r.status_code not in (200, 307) else f"{r.status_code}")(client.get("/", follow_redirects=False)))
check("GET /static/index.html", lambda: (_ for _ in ()).throw(AssertionError(client.get("/static/index.html").status_code)) if client.get("/static/index.html").status_code != 200 else "200")
check("GET /static/home.html 旧链接重定向", lambda: (lambda r: (_ for _ in ()).throw(AssertionError(r.status_code)) if r.status_code not in (200, 307) else f"{r.status_code}")(client.get("/static/home.html", follow_redirects=False)))

# --- 对话 CRUD 全链路 ---
conv_id = None

def _create():
    global conv_id
    r = client.post("/api/conversations", json={"title": "验证对话"})
    assert r.status_code == 200, r.text
    conv_id = r.json()["id"]
    assert r.json()["case_state"]["status"] == "in_progress"
    return conv_id[:8]

check("POST /api/conversations 创建", _create)
check("GET /api/conversations 列表", lambda: (_ for _ in ()).throw(AssertionError("空列表")) if len(client.get("/api/conversations").json()["conversations"]) == 0 else "非空")
check("GET /api/conversations/{id} 详情", lambda: (_ for _ in ()).throw(AssertionError()) if client.get(f"/api/conversations/{conv_id}").json().get("id") != conv_id else "id 匹配")

# --- case-state 链路（LLM 降级下的模板兜底） ---
check("GET case-state", lambda: (_ for _ in ()).throw(AssertionError(r.text)) if (r := client.get(f"/api/conversations/{conv_id}/case-state")).status_code != 200 or not r.json()["success"] else "ok")

def _put_case():
    r = client.put(f"/api/conversations/{conv_id}/case-state",
                   json={"case_state": {"keywords": ["发电机故障"], "todo": [{"id": "t1", "text": "检查电刷", "created_at": "2026-01-01", "done": False}]}})
    assert r.status_code == 200, r.text
    ks = r.json()["case_state"]["keywords"]
    assert "发电机故障" in ks, ks
    return "白名单字段合并 ok"

check("PUT case-state 更新", _put_case)
check("POST generate-keywords（LLM降级→兜底词）", lambda: (_ for _ in ()).throw(AssertionError(r.text)) if (r := client.post(f"/api/conversations/{conv_id}/case-state/generate-keywords")).status_code != 200 or not r.json()["case_state"]["keywords"] else f"keywords={r.json()['case_state']['keywords'][:2]}")
check("POST generate-maintenance-record（模板）", lambda: (_ for _ in ()).throw(AssertionError(r.text)) if (r := client.post(f"/api/conversations/{conv_id}/case-state/generate-maintenance-record")).status_code != 200 or "维修记录" not in r.json()["case_state"]["maintenance_record"] else "markdown 生成 ok")
check("POST generate-postmortem（LLM降级→模板兜底）", lambda: (_ for _ in ()).throw(AssertionError(r.text)) if (r := client.post(f"/api/conversations/{conv_id}/case-state/generate-postmortem")).status_code != 200 or "复盘结果" not in r.json()["case_state"]["postmortem"] else "模板兜底 ok")
check("POST apply-draft", lambda: (_ for _ in ()).throw(AssertionError(r.text)) if (r := client.post(f"/api/conversations/{conv_id}/case-state/apply-draft", json={"fault_context": {"equipment": "主机", "fault_summary": "排气高温"}, "confirm_fault_context": True})).status_code != 200 or not r.json()["case_state"]["fault_context"]["confirmed"] else "确认生效")

# --- 统计/导出/导入 ---
check("GET /api/system/stats", lambda: (_ for _ in ()).throw(AssertionError()) if "conversation_count" not in client.get("/api/system/stats").json() else "ok")
check("GET /api/export/{id}", lambda: (_ for _ in ()).throw(AssertionError()) if not client.get(f"/api/export/{conv_id}").json().get("url") else "url 返回")
check("GET /api/export-md/{id}", lambda: (_ for _ in ()).throw(AssertionError()) if not client.get(f"/api/export-md/{conv_id}").json().get("url") else "url 返回")
check("GET /api/material-library", lambda: (_ for _ in ()).throw(AssertionError()) if client.get("/api/material-library").status_code != 200 else "空列表 ok")

def _import_conv():
    payload = {"messages": [{"role": "user", "content": "主机过热", "timestamp": "2026-01-01"}]}
    r = client.post("/api/import",
                    files={"file": ("conv.json", json.dumps(payload).encode(), "application/json")},
                    data={"title": "导入的对话"})
    assert r.status_code == 200 and r.json()["success"], r.text
    return "multipart 表单 ok（python-multipart 正常）"

check("POST /api/import multipart", _import_conv)

# --- 管理接口鉴权 ---
check("admin 未配置 token → 503 禁用", lambda: (_ for _ in ()).throw(AssertionError(r.status_code)) if (r := client.get("/api/config/preview")).status_code != 503 else "503")

os.environ["RAG_ADMIN_TOKEN"] = "verify-secret"
check("admin 有效 token → 200", lambda: (_ for _ in ()).throw(AssertionError(r.text)) if (r := client.get("/api/config/preview", headers={"X-Admin-Token": "verify-secret"})).status_code != 200 else "200")
check("admin 错 token → 401", lambda: (_ for _ in ()).throw(AssertionError()) if client.get("/api/config/preview", headers={"X-Admin-Token": "wrong"}).status_code != 401 else "401")

# --- API Key 中间件 ---
os.environ["RAG_API_KEY"] = "verify-key"
check("APIKey 未携带 → 401", lambda: (_ for _ in ()).throw(AssertionError()) if client.get("/api/conversations").status_code != 401 else "401")
check("APIKey 正确 → 200", lambda: (_ for _ in ()).throw(AssertionError()) if client.get("/api/conversations", headers={"X-API-Key": "verify-key"}).status_code != 200 else "200")
check("APIKey 模式下 OPTIONS 预检放行", lambda: (_ for _ in ()).throw(AssertionError()) if client.options("/api/conversations", headers={"Origin": "http://localhost:8002", "Access-Control-Request-Method": "GET"}).status_code not in (200, 204) else "放行")
check("APIKey 模式下 admin token 亦放行", lambda: (_ for _ in ()).throw(AssertionError()) if client.get("/api/conversations", headers={"X-Admin-Token": "verify-secret"}).status_code != 200 else "200")
os.environ.pop("RAG_API_KEY", None)
os.environ.pop("RAG_ADMIN_TOKEN", None)

# --- CORS ---
def _cors():
    r = client.options("/api/conversations", headers={"Origin": "http://localhost:8002", "Access-Control-Request-Method": "GET"})
    acao = r.headers.get("access-control-allow-origin", "")
    assert "localhost" in acao, f"ACAO 缺失: {dict(r.headers)}"
    return f"ACAO={acao}"

check("CORS 白名单命中", _cors)

# --- 404 / 删除 ---
check("DELETE /api/conversations/{id}", lambda: (_ for _ in ()).throw(AssertionError()) if client.delete(f"/api/conversations/{conv_id}").status_code != 200 else "ok")
check("GET 已删除对话 → 404", lambda: (_ for _ in ()).throw(AssertionError()) if client.get(f"/api/conversations/{conv_id}").status_code != 404 else "404")

print()
print("=" * 60)
print("阶段4：降级行为确认（预期行为，非 bug）")
print("=" * 60)

check("POST /api/query（未启动 RAG）→ 500", lambda: (_ for _ in ()).throw(AssertionError(r.status_code)) if (r := client.post("/api/query", json={"message": "测试"})).status_code != 500 else "500（预期：lifespan 未运行）")

def _graph_degraded():
    r = client.get("/api/graph/node-counts")
    assert r.status_code == 200, r.status_code
    body = r.json()
    assert body.get("success") is False and "未初始化" in (body.get("message") or ""), body
    return "200 + success=False（优雅降级）"

check("GET /api/graph/node-counts（未启动 RAG）→ 优雅降级", _graph_degraded)

print()
print("=" * 60)
print(f"汇总：PASS {len(PASS)} / FAIL {len(FAIL)}")
print("=" * 60)
for name, err in FAIL:
    print(f"  [FAIL] {name}: {err}")
if not FAIL:
    print("全部通过")
sys.exit(1 if FAIL else 0)
