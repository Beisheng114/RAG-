# 检索质量评测

为 RAG 调参（top_k、chunk_size、路由阈值、是否开启 rerank 等）提供量化依据，
避免"盲调"。

## 评测集

`eval_queries.jsonl`（JSONL，每行一条）：

```json
{"query": "全船停电怎么办", "expected_keywords": ["停电", "失电", "配电板"], "note": "可选备注"}
```

当前内置 10 条**种子示例**（通用船舶故障问题）。建议结合自己知识库的实际
数据扩充到 50–100 条：从 Neo4j 中的故障现象/原因/维修措施节点出发，反向
构造"问题 → 应命中关键词"，覆盖现象定位、多原因枚举、处置步骤等不同类型。

## 指标

| 指标 | 含义 |
|---|---|
| recall@k | top-k 文档命中的期望关键词占比 |
| hit_rate@k | 至少命中一个关键词的问题占比 |
| MRR | 首个相关文档排名倒数的均值（衡量排序质量） |

## 用法

```bash
# 默认 top_k=5、intelligent 路由
python evaluation/run_eval.py

# 多个 k 值对比（调 top_k 前后各跑一次）
python evaluation/run_eval.py --k 3 --k 5 --k 10

# 指定检索模式
python evaluation/run_eval.py --mode traditional
python evaluation/run_eval.py --mode graph

# 保存报告（evaluation/results/，已被 .gitignore 忽略）
python evaluation/run_eval.py --save
```

前置条件：Neo4j / Qdrant / 嵌入模型已就绪且知识库已构建。评测只走检索
链路（含查询改写与 rerank 配置），不调用 LLM 生成，节省算力且可离线对比。

## 推荐流程

1. 基线：`python evaluation/run_eval.py --k 5 --save`
2. 开启/关闭某项能力（如 `enable_rerank`）或调整参数
3. 重跑并对比 `recall@k` / `MRR` 变化，保留更优配置
