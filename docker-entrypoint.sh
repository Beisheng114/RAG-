#!/bin/sh
# 容器入口：等待依赖服务就绪后再启动应用
#
# 为什么需要：docker compose 的 depends_on 只保证"启动顺序"，不保证
# "服务就绪"。app 启动时会立即连接 Neo4j/Qdrant 构建知识库，若数据库
# 尚在初始化（Neo4j 首次启动通常需要 10-30 秒），应用会直接崩溃。
# 本脚本用轮询探测端口就绪后再 exec 启动，配合 compose 的
# condition: service_healthy 形成双保险。
#
# 可通过环境变量调整等待目标与超时：
#   NEO4J_WAIT_URL   默认 bolt://neo4j:7687
#   QDRANT_WAIT_URL  默认 http://qdrant:6333/healthz
#   WAIT_TIMEOUT     默认 180 秒

set -e

NEO4J_WAIT_URL="${NEO4J_WAIT_URL:-bolt://neo4j:7687}"
QDRANT_WAIT_URL="${QDRANT_WAIT_URL:-http://qdrant:6333/healthz}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-180}"

log() {
    echo "[entrypoint] $*"
}

# 用 Python 探测（镜像自带，无需 nc/curl 之外的工具依赖）
wait_for() {
    name="$1"
    url="$2"
    timeout="${3:-$WAIT_TIMEOUT}"
    python3 - "$name" "$url" "$timeout" <<'PYEOF'
import socket
import sys
import time
import urllib.parse

name, url, timeout = sys.argv[1], sys.argv[2], int(sys.argv[3])
parsed = urllib.parse.urlparse(url)
host = parsed.hostname or "localhost"
port = parsed.port or (443 if parsed.scheme == "https" else 80)

deadline = time.time() + timeout
attempt = 0
while time.time() < deadline:
    attempt += 1
    try:
        with socket.create_connection((host, port), timeout=3):
            print(f"[entrypoint] {name} 就绪 ({host}:{port})", flush=True)
            sys.exit(0)
    except OSError:
        if attempt == 1:
            print(f"[entrypoint] 等待 {name} 就绪: {host}:{port} ...", flush=True)
        time.sleep(min(2 ** min(attempt, 4), 10))  # 指数退避，最长10秒

print(f"[entrypoint] {name} 在 {timeout}s 内未就绪，放弃等待", file=sys.stderr, flush=True)
sys.exit(1)
PYEOF
}

log "启动前置检查"

# 工具类容器（模型下载等）不需要等数据库，设置 RAG_SKIP_WAIT=1 跳过
if [ "${RAG_SKIP_WAIT:-0}" != "1" ]; then
    wait_for "Neo4j" "$NEO4J_WAIT_URL"
    wait_for "Qdrant" "$QDRANT_WAIT_URL"
else
    log "RAG_SKIP_WAIT=1，跳过依赖等待"
fi

log "启动应用: $*"
exec "$@"
