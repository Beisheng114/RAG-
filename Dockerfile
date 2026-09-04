# 船舶故障诊断问答RAG系统 - 应用镜像
# 构建上下文需包含 models/（嵌入模型，建议挂载卷而不是打进镜像）
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Shanghai

# 系统依赖：build-essential（部分 Python 包编译）、curl（健康检查）、tzdata（时区）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl tzdata \
    && ln -fs /usr/share/zoneinfo/${TZ} /etc/localtime \
    && echo "${TZ}" > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装依赖（利用层缓存；源码变动不重装依赖）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 代码
COPY . .

# 入口脚本：等待 Neo4j/Qdrant 就绪后再启动
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# 运行时目录（卷挂载点）
RUN mkdir -p models source exports backups conversations bm25_cache context_cache data

EXPOSE 8002

# 健康检查（根路由重定向即可判定应用存活；start-period 覆盖知识库构建时长）
HEALTHCHECK --interval=60s --timeout=10s --start-period=300s --retries=5 \
    CMD curl -fsS http://localhost:8002/ >/dev/null || exit 1

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["python", "app.py"]
