# LightRAG 实现与验证记录

对应任务：[CESH-4](https://antmultica.alipay.com/ceshiqu/issues/CESH-4)，父任务 [CESH-1](https://antmultica.alipay.com/ceshiqu/issues/CESH-1)。
依据：CESH-3 评论附件提供的 `docs/design-docs/active/2026-10-08-lightrag-retrieval-upgrade-design.md`，未改写设计正文。

## 实现范围

- S1：`rag_modules/hybrid_retrieval.py` 恢复双层 LLM 关键词，规则快速路径、超时配置、JSON 校验、思考标签清洗、重试和失败降级；原规则词表不变。
- S2：`rag_modules/graph_indexing.py` 生成实体/关系 KV 向量记录，文本严格按设计 §4.3 拼接。`qdrant_index_construction.py` 复用现有 embedding、Qdrant 和索引参数，新增 KV 集合创建/删除/批量 upsert/分层过滤检索；支持旧 search 和新 query_points 客户端接口。
- 检索保留精确命中，叠加向量命中。精确实体/关系分数为 0.9/0.95，向量分数为相似度乘 0.9；重合标记 exact+vector。每次命中复制 metadata，避免污染共享 KV 对象。
- S3：low/high 作为独立通道进同一 RRF，召回各 2k 再融合。关系优先以 relation_id 去重，保留同节点的不同关系；其他结果以节点 ID 或稳定内容指纹去重。Document 保留来源、层级、实体与命中信息，交现有生成链组织上下文。
- `ragmain.py` 新增显式 dual 分支和 DUAL_LIGHTRAG 分析；仅 dual 精排路径保留 high-level 配额。为避免已有 rerank 的 top_k 截断先丢掉 high，单独精排 high 候选并预留 floor，再保留其余精排槽位。默认 floor=2，实际受 top_k 和可用 high 数量限制。
- S5：全量初始化重灌 KV 向量；增量刷新 KV store/BM25/缓存，限定受影响节点和一跳关系删除旧记录后 upsert，不全量重灌 KV 向量；失败仅告警。代码核对发现 graph_indexed 原本会短路 KV 刷新，因此初始化时清空旧 KV/倒排并重建，以落实设计要求的增量刷新。
- S6：`evaluation/run_eval.py --mode dual`，评测复用主入口同一精排/保底函数。另提供固定 mock 样例的离线回归评测，不增加运行依赖。

未修改 `app.py`、抽取 schema、图多跳引擎、路由选择策略、`core/rrf.py`、对话存储、导出或 UI。旧四种模式的路由、入口签名、默认值和返回形状保持不变；S1/S2 会按设计改善其共享 KV 检索入口，因此不声称旧模式的检索内容逐字不变。

## 验证与复现

```bash
python3 -m pytest tests/ -q
python3 -m compileall -q config.py ragmain.py rag_modules/hybrid_retrieval.py rag_modules/graph_indexing.py rag_modules/qdrant_index_construction.py rag_modules/intelligent_query_router.py evaluation tests
git diff --check
python3 evaluation/offline_retrieval_check.py --output docs/evaluation/lightrag-offline-comparison.json
```

新增离线测试覆盖关键词快速路径/失败/重试，实体与关系向量补召回、精确重合、文本和过滤、Qdrant 生命周期和操作顺序，双层 RRF/关系去重、精排 high-floor、五种模式的普通/流式返回、增量删除与失败告警、缓存失效与受影响一跳关系查询、评测指标聚合。

本地结果：`pytest tests/` 为 **73 passed, 1 skipped**（升级前为 41 passed, 1 skipped）；上述 compileall 与 git diff --check 均通过。

## 降级评测结果

真实基线代码来自提交 `7a614abf7723f0d18285589589cda66beda134f7` 的 `hybrid_retrieval.py`，从 Git 读取并执行实际类；升级端执行实际 `dual_fusion_search`。两者使用相同的两实体/一关系、5 条固定查询、固定双层关键词。chunk 向量/BM25/Neo4j 补充均置空，KV 向量相似度由 mock 给定，rerank 关闭。指标调用现有 `evaluation/run_eval.py` 的 evaluate_query/summarize 计算。

| k | baseline traditional recall@k | dual recall@k | baseline MRR | dual MRR |
|---|---:|---:|---:|---:|
| 3 | 0.4 | 0.8 | 0.4 | 0.8 |
| 5 | 0.4 | 0.8 | 0.4 | 0.8 |
| 10 | 0.4 | 0.8 | 0.4 | 0.8 |

JSON：`docs/evaluation/lightrag-offline-comparison.json`，含每条查询命中详情、基线 SHA、环境、覆盖与未覆盖清单。

这些数字是人为固定数据上的回归证据，只证明精确/向量补召回和指标计算控制流有效；不能证明实际语义召回质量提高或线上不劣化，也不是原 10 条评测集的成绩。

## 未覆盖与后续验收

**端到端未跑。** 升级前运行 `python3 evaluation/run_eval.py --k 3 --k 5 --k 10 --mode intelligent --save` 在导入阶段报 `ModuleNotFoundError: neo4j`。当前环境未配置真实 Neo4j、Qdrant、本地 embedding/LLM/reranker 模型。没有用零值或 mock 数字代替在线成绩，JSON 的 end_to_end_metrics 为 null。

准备在线依赖后，在基线提交与升级分支分别运行：

```bash
# 基线提交：intelligent 与 traditional
python evaluation/run_eval.py --k 3 --k 5 --k 10 --mode intelligent --save
python evaluation/run_eval.py --k 3 --k 5 --k 10 --mode traditional --save
# 升级分支：dual 与 intelligent，同一图数据、模型和评测集
python evaluation/run_eval.py --k 3 --k 5 --k 10 --mode dual --save
python evaluation/run_eval.py --k 3 --k 5 --k 10 --mode intelligent --save
```

Reviewer 仍需核实线上质量不劣化、真实 Qdrant/Neo4j 增量一致性及模型质量。原全量关系读取 LIMIT 1000 保留，受影响一跳查询不受该全局上限约束；大图全量覆盖上限属于现有约束，后续需与 Architect 确认分页方案。KV 写入失败会留下语义旁路不完整，精确/Neo4j 路径可用，恢复服务后需重建。

回滚：不使用 dual 即恢复旧路由体验；`qdrant_kv_collection_name=""` 关闭 KV 语义旁路；关键词失败自动回退规则，不变更数据库 schema。
