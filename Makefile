# 船舶故障诊断问答RAG系统 - 常用命令
# 用法: make <target>（需要 docker compose v2；首次使用先 cp .env.example .env）

.PHONY: help env up down restart logs ps build rebuild \
        download-models init-csv eval test clean-soft clean-all

help: ## 显示所有可用命令
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

env: ## 初始化本地配置（从模板生成 .env）
	@test -f .env && echo ".env 已存在，跳过" || (cp .env.example .env && echo "已生成 .env，请编辑填写 NEO4J_PASSWORD")

up: ## 启动全部服务（后台）
	docker compose up -d

down: ## 停止并移除容器（数据卷保留）
	docker compose down

restart: ## 重启应用容器
	docker compose restart app

logs: ## 跟踪应用日志
	docker compose logs -f app

ps: ## 查看服务状态
	docker compose ps

build: ## 构建应用镜像
	docker compose build app

rebuild: ## 无缓存重建镜像
	docker compose build --no-cache app

# ---------- 工具（对应 compose 的 tools profile） ----------

download-models: ## 下载嵌入+精排模型到 ./models（容器内执行）
	docker compose run --rm model-download

init-csv: ## 清空 Neo4j 并导入 ./generate_csv 的 CSV（危险：会覆盖图谱）
	docker compose run --rm csv-import

eval: ## 运行检索评测（recall@k / MRR），报告存 evaluation/results/
	docker compose run --rm eval

test: ## 本地跑 pytest 冒烟测试（不走容器）
	python3 -m pytest tests/ -v

# ---------- 清理 ----------

clean-soft: ## 停止并删除容器与网络（保留全部数据卷）
	docker compose down --remove-orphans

clean-all: ## 危险：停止并删除容器+全部数据卷（图谱/向量/对话全丢）
	docker compose down -v --remove-orphans
