# 船舶故障诊断问答RAG系统 - 应用镜像
# 构建上下文需包含 models/（嵌入模型，建议挂载卷而不是打进镜像）
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 系统依赖（部分 Python 包的编译依赖）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# 先装依赖（利用层缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 代码
COPY . .

# 运行时目录（卷挂载点）
RUN mkdir -p models source exports backups conversations

EXPOSE 8002

# 健康检查（页面路由）
HEALTHCHECK --interval=60s --timeout=10s --start-period=180s --retries=5 \
    CMD curl -fsS http://localhost:8002/ >/dev/null || exit 1

CMD ["python", "app.py"]
