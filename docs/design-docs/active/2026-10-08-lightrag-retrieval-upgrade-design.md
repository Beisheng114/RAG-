# LightRAG 检索升级设计文档

| 项 | 值 |
|---|---|
| Issue | [CESH-3](https://antmultica.alipay.com/ceshiqu/issues/CESH-3)（父 issue [CESH-1](https://antmultica.alipay.com/ceshiqu/issues/CESH-1)）|
| 角色 | Stack Architect |
| 版本 | v1.0（2026-10-08）|
| 仓库 | https://github.com/Beisheng114/RAG- 分支 `agent/stack-architect/b90382100ed6` |
| 依据 | Stage 1 调研：Stack Scout 差距分析报告（父 issue 评论，2026-10-08T14:47:38）；本设计另逐行核对仓库代码 |
| 语雀副本 | 待接入 |

> 阅读对象：Stack Coder（据以实现）、Stack Reviewer（据以评审）、Stack Scribe（据以组织最终文档）。

---

## 0. TL;DR

现有检索链路对 LightRAG 的**结构形态对齐已超 80%**（双层 KV、邻居扩展、RRF 三路融合、去重、增量对齐、检索-生成解耦都在），唯一真正缺口在「双层语义入口的质量」：

1. `hybrid_retrieval.extract_query_keywords` 的 **LLM 双层关键词提取被整段注释**，双层入口只能走规则词表（`_fallback_keyword_extraction`），即「双层有名无实」。
2. KV 检索 `graph_indexing.get_entities_by_key` / `get_relations_by_key` 是**精确字典命中**（key 只有 `entity_name`），无向量/语义匹配，未对齐 LightRAG 的 L6 语义召回，漏召严重。
3. high-level 主题索引键由 `_generate_relation_index_keys` **硬编码关系类型→中文短语词表**生成，覆盖不到同义/近义查询。

本设计据此给出三项可落地改动 + 一个新增可评测模式，**全部落在检索逻辑以内，不动抽取 schema、不动 Qdrant/Neo4j、不破坏对外接口**：

- **S1 恢复并守护 LLM 双层关键词提取**（原始代码已在文件里，重新启用 + 快速路径 + 降级链）。
- **S2 给 KV 建向量匹配旁路**（新增一个 Qdrant KV collection，实体/关系 KV 文本向量化后语义召回，与 chunk collection 生命周期绑定）。
- **S3 新增 `dual` 统一双层融合模式**（low/high 进同一 RRF 而非各自截断；rerank 保留 high-level 最少配额）。
- 新增 `--mode dual` 评测入口与离线逻辑单测。

对开放问题的决策已写入第 8 节。

---

## 1. 背景与目标

### 1.1 要解决什么

仓库现存一个基于 RAG 的船舶故障诊断问答系统，已具备：向量（Qdrant）+ BM25 + 图（Neo4j）三路 RRF 融合、双层 KV 检索（`hybrid_retrieval`）、多跳/子图检索（`graph_rag_retrieval`）、智能查询路由、Rerank 精排、查询改写。父 issue 要求将其「合格地作为一个 LightRAG」——对齐 LightRAG（HKUDS/LightRAG，arXiv:2410.05779）核心闭环：

```
实体/关系抽取 → 图谱增量构建 → 双层检索(low-level 实体 + high-level 关系主题)
              → 检索结果去重与上下文组织 → 与查询融合后送 LLM 生成
```

强调**轻量、低资源、易部署、检索与生成可解耦**。

### 1.2 不在范围（明确划出）

| 不做 | 原因 / 归属 |
|---|---|
| 替换 Qdrant / Neo4j | 父 issue 范围外；继续复用现有栈 |
| 引入重依赖（新向量库框架/图算法库/大抽取模型） | 与「light」定位冲突 |
| 改抽取 schema（`graph_data_insert` 的强领域 schema、`find_or_create_entity`） | 船舶领域 schema 合理，**只对齐检索侧**（见决策 Q2）|
| 改 `context_manager` / `services` / 对话存储 / 导出 / UI | 与检索升级无关 |
| 改图谱可视化、CSV 导入 | 与检索升级无关 |
| 给 Neo4j 加新的全文/向量索引服务端能力（依赖服务端插件） | 父 issue 不替换图库；KV 向量化落在 Qdrant 侧 |

### 1.3 设计成功标准

本设计文档须可被 Coder 直接实现、Reviewer 直接对照评审、Scribe 直接组织最终文档，即：

- 每条改动精确到**文件名 + 函数名 + 行号锚点**；
- 给出**接口契约的不变点 / 变更点**（向后兼容）；
- 给出**逐条对照父 issue 5 条验收标准的满足路径**；
- 给出**评测跑法与无 Neo4j/Qdrant 时的降级评测覆盖度说明**；
- 给出**风险与回滚**。

---

## 2. 现状审计（本设计已逐行核对，非照搬调研结论）

> Stack Scout 调研报告与本审计一致；以下为代码锚点，供 Coder/Reviewer 直接定位。

### 2.1 双层 KV 引擎：`rag_modules/hybrid_retrieval.py`

- `HybridRetrievalModule.dual_level_retrieval()`（`hybrid_retrieval.py:774-841`）：提关键词 → `entity_level_retrieval` + `topic_level_retrieval` → 合并 → `_dedup_key` 去重 → 转 `Document`。
- `entity_level_retrieval()`（`503-547`）：逐关键词 `graph_indexing.get_entities_by_key(keyword)` → 取邻居 → 命中得分写死 `0.9`；不足时 `_neo4j_entity_level_search` 补。
- `topic_level_retrieval()`（`627-712`）：逐关键词 `get_relations_by_key(keyword)` + `get_entities_by_key(keyword)`（分类主题）→ 写死 `0.95`/`0.85`；不足时 `_neo4j_topic_level_search` 补。
- **缺陷锚点**：
  - `extract_query_keywords()`（`250-365`）：LLM 提示词整段注释，函数体直接 `return self._fallback_keyword_extraction(query)`（`365`）。**双层语义入口被关闭。**
  - `hybrid_search()`（`967-1103`）：三路 `vector / bm25 / graph` 走 `rrf_fuse`（`core/rrf.py`），但其中 `graph` 这一路就是 `dual_level_retrieval` 的输出——**low/high 在 `dual_level_retrieval` 内已被各自截断到 `top_k` 后再相加**（`784-785` 各取 `top_k`，`812` 再 `[:top_k]`），low/high 没有进同一 RRF。
- **可复用锚点**：`core/rrf.py:rrf_fuse`（纯函数，已单测覆盖）、`ThreadPoolExecutor` 三路并发（`1004-1014`）、`_dedup_key`（`794-801`，按 `relation_id` / `node_id` / 内容指纹三级去重）、`_bm25_signature` 缓存指纹（`122-133`）。

### 2.2 KV 存储：`rag_modules/graph_indexing.py`

- `entity_kv_store: Dict[str, EntityKeyValue]` / `relation_kv_store: Dict[str, RelationKeyValue]`（`54-55`）。
- `key_to_entities` / `key_to_relations`：`Dict[str, List[str]]` 倒排（`58-59`）。
- `get_entities_by_key(key)`（`743-746`）/ `get_relations_by_key(key)`（`748-751`）：**纯 `dict.get` 精确命中，无模糊/向量匹配**。这是「L6 语义召回」缺口的来源。
- `create_entity_key_values()`（`63-393`）：每实体 `index_keys=[entity_name]`（**单键 = 实体名**），所以查「主机」命中不了 `entity_name="主柴油机"`。
- `_generate_relation_index_keys()`（`395-530`）：按 `relation_type` **硬编码**一批中文主题短语作为 high-level 索引键（如 `HAS_FAULT → ["设备故障","故障现象",...]`）。这就是 high-level「主题硬编码词表」缺口的来源。
- `enable_llm_relation_keys`（`config.py:93`，默认 `False`）：留有 LLM 增强关系键的钩子（`graph_indexing.py:524-527, 621-673`），但默认关。
- `deduplicate_entities_and_relations()`（`675-726`）+ `_rebuild_key_mappings()`（`728-741`）：去重并重建倒排。

### 2.3 图 RAG 引擎：`rag_modules/graph_rag_retrieval.py`（与 KV 引擎并存，职责不同）

- `graph_rag_search()`（`952-999`）：意图分类 → `multi_hop_traversal` / `extract_knowledge_subgraph` → 排序。这是**图多跳/子图**引擎，不是 LightRAG 的「关系 KV 主题」引擎；它**已自带**一个带倒排索引 + degree 加权 + 上下文缓存评分的 `_fast_entity_match()`（`403-465`），即 Neo4j 侧的语义锚点能力**已具备**。
- **结论**：本引擎是「关系推理/路径发现」的 high-level 补强，不替代 KV 双层。LightRAG high-level 的「关系 KV 语义匹配」仍要在 **KV 引擎**补（S2 + S3），而非在此处动刀。

### 2.4 路由与对外入口

- `ragmain.ask_question_with_routing()`（`ragmain.py:291-461`）：`search_mode ∈ {intelligent, traditional, graph, combined}`。traditional → `hybrid_search`；graph → `graph_rag_search`；intelligent → `query_router.route_query`；combined → 两路各 `recall_k//2` 合并。
- 评测 `evaluation/run_eval.py:_retrieve_with_k()`（`118-126`）：`mode` 映射同上，但 `choices=["intelligent","traditional","graph"]`（**无 dual**，S6 落点）。
- FastAPI `app.py`：检索唯一入口是 `/api/query` 与 `/api/query/stream`（`app.py:240, 304`），`QueryRequest.search_mode`（`app.py:124-127`，字符串，默认 `"intelligent"`）透传给 `ask_question_with_routing`。`search_mode` 是**枚举式字符串**，新增取值不破坏旧取值。
- 增量：`ragmain.update_knowledge_base_incremental()`（`574-636`）：增量切块 → 按 `neo4j_node_id` 删旧向量点 → 加新向量点 → 最后 `_initialize_retrievers(full_chunks)`（`626`）→ 重建 BM25 + **KV store**（`_build_graph_index` → `create_entity_key_values`）。**KV store 随增量已刷新**；唯一缺口是「新的 KV 向量 collection 没有随增量 upsert」（S2/R3 补全闭环）。

### 2.5 抽取侧（不动，仅说明为何保留）

- `rag_modules/graph_data_insert.py:find_or_create_entity()`（`596`）等：强领域 schema 抽取 + 实体对齐（多处以 name 去重 MERGE）。船舶领域实体类型固定（Equipment/Component/FaultPhenomenon/FaultReason/MaintenanceAction/SafetyNotice/KnowledgeSource/EquipmentCategory/Fault），强 schema **合理且对检索有利**（Review/Coder 可对照 `get_statistics` 的类型分布，`graph_indexing.py:756-774`）。**决策：保留，不对齐抽取（见 Q2）。**

---

## 3. 架构与模块划分

### 3.1 整体结构（升级后）

```
                       ask_question_with_routing(search_mode)
                                  │
        ┌─────────────────────────┼────────────────────────────┐
   traditional              intelligent                 graph            combined   [新增] dual
        │                        │                        │                │          │
  hybrid_search          query_router.route_query   graph_rag_search   传统+图各k/2   dual_fusion_search  ← S3新增
        │                  (complexity→选其一)            │
        │                        │                        │
   ┌────┴─────────── RRF 三路融合 ──────────────────┐     │
 vector(channel1)  bm25(channel2)  dual_level  (channel3)│     │
                                 │                      │     │
                       extract_query_keywords ← S1 恢复 LLM + 快速路径
                       entity_level_retrieval  ── KV 向量旁路 ← S2 新增
                       topic_level_retrieval   ── KV 向量旁路 ← S2 新增
                                 │
                       graph_indexing (KV store)
                                 │
                 Qdrant KV collection  ← S2 新增（生命周期绑定 chunk collection）
```

升级后**检索主链路仍是 `traditional/hybrid_search`**（三路 RRF 不变），其中 `graph` 一路 = `dual_level_retrieval`（S1+S2 升级其入口与 KV 匹配质量）；新增独立 `dual` 模式（S3）专门暴露「low/high 统一进 RRF」的 LightRAG 一致形态用于评测与默认候选。两套并存，**向后兼容**。

### 3.2 关键模块与职责（升级后）

| 模块 | 职责 | 本设计改动 |
|---|---|---|
| `rag_modules/hybrid_retrieval.py` | 双层 KV 检索 + 三路 RRF 融合 | S1 恢复 LLM 关键词；S2 接 KV 向量旁路；`dual_fusion_search` 新方法（S3）|
| `rag_modules/graph_indexing.py` | 实体/关系 KV 存储 + 倒排 | KV 向量化接口（新增 `_build_kv_vector_index` / `query_kv_by_vector`），high-level 键**保留硬编码但叠加向量召回**（不删词表，避免回归）|
| `rag_modules/qdrant_index_construction.py` | Qdrant chunk 索引 | 新增 KV collection 管理（建/删/upsert/搜），与 chunk collection 同 host/embberding 模型、同生命周期 |
| `ragmain.py` | 装配 + 对外入口 | `search_mode` 新增 `dual` 分支调用 `dual_fusion_search`；增量路径补 KV 向量 upsert |
| `evaluation/run_eval.py` | 检索质量评测 | `--mode` 增 `dual`；`_retrieve_with_k` 增 `dual` 分支 |
| `config.py` | 配置 | 新增 KV 向量旁路开关与集合名、`dual` 高层保底配额、S1 快速路径开关（复用 `router_fast_path`）|
| `core/rrf.py` | RRF 纯函数 | **不动**（已被 `dual_fusion_search` 复用）|

### 3.3 边界

- **写入边界**：改动集中在 `rag_modules/hybrid_retrieval.py`、`rag_modules/graph_indexing.py`、`rag_modules/qdrant_index_construction.py`（KV 集合新增）、`ragmain.py`（入口 + 增量）、`evaluation/run_eval.py`、`config.py`、对应 `tests/`。**不动** `graph_rag_retrieval.py`（图多跳引擎保持）、`intelligent_query_router.py`（路由策略保持，`intelligent` 行为不变）、`context_manager.py`、`services/`、`routers/`、`app.py`（仅 `QueryRequest.search_mode` 多一个合法值，模型字段不变）、`graph_data_insert.py`（抽取 schema 保留）。
- **运行依赖边界**：新增 KV 向量旁路**复用现有 embedding 模型** `./models/bge-base-zh-v1.5`（`config.py:47`，768 维）与现有 Qdrant 实例（`config.py:34-44`），**不引入新模型/新服务**。

---

## 4. 关键接口与数据模型

### 4.1 与 LightRAG 原文的逐点对照

| LightRAG 原文要素 | 对应实现（升级前） | 升级后（本设计）| 对齐度变化 |
|---|---|---|---|
| **P(·)** Profiling：把实体/关系 profile 成 KV | `graph_indexing.create_entity_key_values` / `create_relation_key_values` 已建 KV；`entity_name` 单键 | 同上（key 仍保留便于精确命中/调试）| 形态已有 → 保持 |
| **Low-level 检索**：实体 KV + 邻居 `N_v ∪ N_e` | `entity_level_retrieval` 取 `get_entities_by_key` + `_get_node_neighbors`（max 2）| 关键词恢复 LLM（S1）；KV 命中补向量召回（S2）| 入口语义：无 → 有 |
| **High-level 检索**：关系 KV 按 query 语义召回 | `topic_level_retrieval` 取 `get_relations_by_key`，key=硬编码主题短语 | 高层键保留 + **关系 KV 向量召回**（S2）| 召回：精确 → 语义 |
| **KV 语义匹配（L6）** | 无（精确 `dict.get`）| 新增 Qdrant KV collection + `query_kv_by_vector`（S2）| 缺 → 补 |
| **双层融合** | low/high 各自 `[:top_k]` 后相加，未进同一 RRF | `dual_fusion_search` 把 low/high 作为两路进同一 `rrf_fuse`（S3）| 伪双层 → 真双层 |
| **去重 + context 组织** | `_dedup_key`（relation_id/node_id/内容指纹三级）| **沿用**，不改动 | 已对齐 |
| **增量更新** | `update_knowledge_base_incremental` 重建 BM25+KV store，但未刷 KV 向量 | 补 `KV 向量 upsert`（R3）| 闭环不完整 → 闭环 |
| **检索-生成解耦** | `run_eval` 不调生成；`ask_question_with_routing` 检索与生成分两段 | **不动** | 已对齐 |
| **实体/关系抽取** | 强领域 schema + `find_or_create_entity` 对齐 | **不动**（决策 Q2）| 不对齐到此层 |

### 4.2 修改后数据结构

**新增 dataclass（`graph_indexing.py`，与现有 `EntityKeyValue`/`RelationKeyValue` 并存）**：

```python
@dataclass
class KVVectorRecord:
    """KV 向量旁路的单条记录（用于在 Qdrant KV collection 里存实体/关系 KV 的向量）"""
    record_id: str            # "entity:<entity_id>" 或 "relation:<relation_id>"
    kv_kind: str              # "entity" | "relation"
    source_id: str             # entity_id 或 relation_id（回指 KV store）
    text: str                  # 用于向量的拼接文本（见 4.3）
    level: str                 # "low"(entity) | "high"(relation)
    entity_type: str           # 实体类型 / 关系类型
    metadata: Dict[str, Any]  # 透传 KV 的关键字段，便于结果回填
```

> 不修改 `EntityKeyValue`/`RelationKeyValue` 现有字段，避免影响 `dual_level_retrieval`/`get_statistics` 等既有读取方。

**`RetrievalResult`（`hybrid_retrieval.py:55-63`）新增可选字段**（向后兼容，带默认值）：

```python
@dataclass
class RetrievalResult:
    content: str
    neo4j_node_id: str
    entity_type: str
    relevance_score: float
    retrieval_level: str
    metadata: Dict[str, Any]
    # 新增（可空）：KV 向量召回时的相似度与来源，便于评测/调试
    kv_vector_score: float = 0.0
    kv_hit_source: str = ""    # "exact" | "vector" | "exact+vector"
```

### 4.3 KV 向量文本拼接规则（决定召回质量，需 Coder 严守）

为让一个 KV 条目对「实体查询」和「主题查询」都能被语义召回，向量化文本按层级拼接（复用现有 `value_content`，不引入新字段）：

- **low-level（entity）**：`text = entity_name + "\n" + value_content`（即「设备名称：主柴油机\n类型:...\n型号:...」）。只内联 KV 自身字段，不内联邻居（邻居在 `entity_level_retrieval` 里动态拼接，保持轻量）。
- **high-level（relation）**：`text = relation_type + "\n" + source_entity.entity_name + "→" + target_entity.entity_name + "\n" + relation 的 value_content 描述句`（即「HAS_FAULT\n主柴油机→排气温度过高\n关系类型:...设备故障:...存在...故障」）。

> 拼接复用 `graph_indexing.create_entity_key_values` / `create_relation_key_values` 已构造的 `value_content`，**不重新做抽取**。

### 4.4 接口契约：向后兼容点

**`ask_question_with_routing` 签名不变**（`ragmain.py:291`）：
```python
def ask_question_with_routing(self, question, stream=False, explain_routing=False,
                              conversation_history=None, search_mode="intelligent")
```
- **不变点**：`intelligent` / `traditional` / `graph` / `combined` 四种模式行为与返回签名 `(result, analysis)` 完全不变；`analysis` 在新分支补一个 `SearchStrategy.DUAL_LIGHTRAG`（见 6.x 枚举扩展）。
- **变更点（附加，非破坏）**：`search_mode` 合法值集合从 4 个扩到 5 个，新增 `"dual"`。FastAPI `QueryRequest.search_mode: Optional[str]`（`app.py:127`）**类型不变**，前端传 `"dual"` 即用新链路；不传则维持 `intelligent`。

**`run_eval.py` `--mode`（附加）**：`choices` 由 `["intelligent","traditional","graph"]` 扩到含 `"dual"`（`run_eval.py:152`）。`_retrieve_with_k`（`118-126`）增一个 `dual` 分支调用新方法，不影响既有分支。

**`HybridRetrievalModule` 公有方法**：保留 `hybrid_search` / `dual_level_retrieval` / `entity_level_retrieval` / `topic_level_retrieval` 签名；新增 `dual_fusion_search(self, query, top_k)`（S3）。`initialize()` 签名不变，但在其后侧触发 KV 向量旁路构建。

---

## 5. 分阶段实现建议（子任务边界与先后顺序）

> **顺序至关重要**：S1 复用现有注释代码，最低风险、最先做；S2 建立语义召回根基，S3 才能体现「真双层」价值；S4/S5/S6 顺势收口。每个阶段产出可单独 PR/可单独评测/可单独回滚。

### S1 — 恢复 LLM 双层关键词提取 + 快速路径（低风险，原代码已在）

**目标**：把 `extract_query_keywords`（`hybrid_retrieval.py:250-365`）从「整段注释→直接 fallback」改为「快速路径判否→走 LLM→解析失败降级」。

**改动点**：
1. 取消 `extract_query_keywords` 体内的注释（`250-365`），恢复 LLM 调用 + `<|think|>` 清洗 + JSON 提取 + 重试逻辑。
2. 新增**快速路径门控**（决策 R1）：入口先调 `_fallback_keyword_extraction(query)`（规则词表），若命中 ≥1 实体关键词且不含抽象推理类词（复用 `intelligent_query_router._rule_based_analysis` 的 `relation_keywords` 判定，`intelligent_query_router.py:194-198`），则**直接返回规则关键词跳过 LLM**；否则进 LLM。门控开关复用 `config.router_fast_path`（`config.py:97`），新增 `config.lightrag_llm_keyword_fast_path: bool = True`（默认开，与 router 一致体验）。
3. LLM 调用超时改用 `config.llm_timeout_chat`（`config.py:84`，60s，代替注释里硬编码 60s/`num_predict=500`，与全仓统一）。
4. 降级链不变：LLM 失败 → `_fallback_keyword_extraction`（`365`）。**`_fallback_keyword_extraction` 一字不改**（领域词表继续作为最终兜底）。

**验收**：`tests/` 新增 `test_keyword_extraction_fast_path`（mock LLM，断言快速路径不调 LLM；断言抽象查询触发 LLM 并能解析 JSON）。

### S2 — KV 向量匹配旁路（核心，立语义召回）

**目标**：给 KV 建一个 Qdrant KV collection，把实体/关系 KV 文本向量化后存入；`entity_level_retrieval`/`topic_level_retrieval` 在精确命中之外，叠加向量召回。

**改动点（`qdrant_index_construction.py` 新增 KV 集合管理）**：
1. `QdrantIndexConstructionModule` 新增 `ensure_kv_collection(name)` / `upsert_kv_vectors(records: List[KVVectorRecord])` / `search_kv_vectors(query_text, level, top_k)`。集合名 `config.qdrant_kv_collection_name = "ship_maintenance_kv"`（新增，默认值）；向量维度/距离/hnsw 复用现有 `qdrant_vector_size`/`qdrant_distance`/`qdrant_hnsw_*`（`config.py:39-44`）；embedding model 复用 `self._embedding`（已有，不新建）。
2. **生命周期绑定 chunk collection（决策 R2）**：用与 BM25 相同的指纹思路——`QdrantIndexConstructionModule.ensure_kv_collection` 在 `build_knowledge_base` 路径里，由 `HybridRetrievalModule.initialize` 末尾触发一次「KV store → 向量」全量构建；KV 集合不独立维护版本号，靠「全量重建即全量重灌 KV 向量」「增量即增量 upsert KV 向量」保持一致（与 BM25 缓存失效同模型）。
3. **`graph_indexing.py` 新增**：`_build_kv_vector_records()` 遍历 `entity_kv_store`/`relation_kv_store` 按 4.3 规则产出 `KVVectorRecord`；`get_entities_by_vector(query_text, top_k)` / `get_relations_by_vector(query_text, top_k)` 返回 `List[EntityKeyValue]` / `List[RelationKeyValue]`（带 `kv_vector_score`）。

**改动点（`hybrid_retrieval.py` 接入）**：
4. `entity_level_retrieval`：在「精确命中 `get_entities_by_key`」之外，**追加** `graph_indexing.get_entities_by_vector(keyword|query, top_k)`；精确命中标记 `kv_hit_source="exact"`，向量召回标记 `"vector"`，两路都命中同一 entity 标 `"exact+vector"` 并取较高分。
5. `topic_level_retrieval`：同理对关系 KV 追加 `get_relations_by_vector`。
6. **评分对齐**：精确命中沿用 `0.9`/`0.95`；向量召回分数 = `kv_vector_score * 0.9`（语义分乘衰减系数，低于精确命中、高于 Neo4j 补充 `0.7`/`0.75`，保证排序合理）。

**验收**：`tests/test_kv_vector_retrieval.py`（mock Qdrant client，断言：无精确命中时仍能通过向量召回 entity/relation；`kv_hit_source` 正确；分数单调）。

### S3 — 新增 `dual` 统一双层融合模式（落实「真双层」）

**目标**：让 low/high 进**同一 `rrf_fuse`** 而非各自截断；并作为可评测、可作默认候选的 LightRAG 一致模式。

**改动点**：
1. `hybrid_retrieval.py` 新增 `dual_fusion_search(self, query, top_k) -> List[Document]`：
   - `extract_query_keywords` 得 entity/topic 关键词（S1）；
   - `low_docs = entity_level_retrieval(..., top_k=recall)`、`high_docs = topic_level_retrieval(..., top_k=recall)`，其中 `recall = top_k*2`（多召回再融合）；
   - `ranked = rrf_fuse({"low": low_docs, "high": high_docs}, doc_id_fn=_doc_id)`（复用 `core/rrf.py` 与 `hybrid_search` 同款 `_doc_id`，`hybrid_retrieval.py:1061-1067`）；
   - 去重沿用 `_dedup_key`；
   - 返回 `ranked[:top_k]` 转 `Document`。
2. `intelligent_query_router.py`：`SearchStrategy` 枚举新增 `DUAL_LIGHTRAG = "dual_lightrag"`（`intelligent_query_router.py:22-26`）；路由 `analyze_query` **不改**（intelligent 仍只挑 traditional/graph/combined，决策 Q1——不替换 intelligent 旧语义）；`DUAL_LIGHTRAG` 仅由 `search_mode="dual"` 显式触发，不进 router 自动选择。
3. `ragmain.ask_question_with_routing`（`ragmain.py:331-397`）：在 `elif search_mode == "combined":` 之后插入 `elif search_mode == "dual":` 分支，调 `self.traditional_retrieval.dual_fusion_search(retrieval_query, recall_k)`，补一个 `QueryAnalysis(recommended_strategy=DUAL_LIGHTRAG, ...)`。
4. **rerank 保底配额（决策 Q3）**：`ragmain.py:399-400` 的 `rerank` 前后，仅当 `search_mode=="dual"` 且 `rerank` 启用时，对 `relevant_docs` 按 `retrieval_level` 标注后送 rerank，再合并时**强制保留至少 `config.dual_high_floor`（默认 2）条 high-level 结果**（从 rerank 输出中按 high-level 优先取够 floor，其余槽位按 rerank 分排序）。`dual_high_floor` 新增到 `config.py`（默认 2）。`traditional/graph/combined/intelligent` 路径**不引入**此逻辑，行为零变化。

**验收**：`tests/test_dual_fusion.py`（mock 三路，断言 low/high 进同一 RRF；断言 high-floor 在 rerank 后被保留）。

### S4 — `intelligent_query_router` 旧路径不动 + 传统/图逃生（决策 Q1 收口说明）

不写代码改动，仅作为边界说明：`intelligent` 模式继续「路由二选一」，`traditional`/`graph` 作为逃生路径保留。**`dual` 与 `intelligent` 并存**：`dual` 是「检索逻辑对齐 LightRAG」的明面形态与默认评测对象，`intelligent` 是「按复杂度路由到 traditional/graph」的既有体验。两者职责不冲突。

### S5 — 增量闭环：KV 向量随增量 upsert（决策 R3）

**目标**：补全 `update_knowledge_base_incremental` 对 KV 向量集合的副作用。

**改动点（`ragmain.py:update_knowledge_base_incremental`，`574-636`）**：
1. 在 `self._initialize_retrievers(full_chunks)`（`626`）之后，追加：若 `self.index_module` 提供 `upsert_kv_vectors`（S2 新增）且 `get_entities_by_node_ids(neo4j_node_ids)` 可得受影响 KV，则 **仅对这些节点的 entity/relation KV 调 `upsert_kv_vectors`**（先按 `record_id` 删旧再 upsert，与 chunk 向量同删-增顺序）。
2. 受影响关系集合：从 Neo4j 取这些节点的一跳关系（复用 `hybrid_retrieval._extract_relationships_from_graph()` 的 cypher，限定 `WHERE source_id IN $ids OR target_id IN $ids`）。
3. 兜底：若 KV 向量 upsert 失败，**仅告警不阻断**（与现有「增量失败回退全量重建」策略一致，`631-636`）。

**验收**：`tests/test_incremental_kv_upsert.py`（断言新增节点的 KV 向量被 upsert、旧向量被删；失败仅告警）。

### S6 — 评测补 `--mode dual` + 降级评测覆盖度文档

**改动点**：
1. `evaluation/run_eval.py:152` `choices` 增 `"dual"`；`run_eval.py:118-126` `_retrieve_with_k` 增 `dual` 分支调 `dual_fusion_search`。
2. 评测报告里记录 `metrics_by_mode`（baseline=traditional/intelligent 升级前 vs dual 升级后），落盘对比 JSON。
3. 无 Neo4j/Qdrant 时：跑离线逻辑单测（`test_kv_vector_retrieval`/`test_dual_fusion`/`test_keyword_extraction_fast_path`，均 mock 重依赖）；端到端 recall@k/MRR 需在线环境，在报告里**显式标注未覆盖**（见第 7 节覆盖度说明）。

---

## 6. 模块改动清单（精确到文件与函数）

| 文件 | 函数/符号 | 改动 | 阶段 |
|---|---|---|---|
| `rag_modules/hybrid_retrieval.py` | `extract_query_keywords` (`250-365`) | 取消注释、接 LLM、加快速路径门控、超时用 config | S1 |
| `rag_modules/hybrid_retrieval.py` | `_fallback_keyword_extraction` (`367-470`) | **不动**（最终兜底）| S1 |
| `rag_modules/hybrid_retrieval.py` | `entity_level_retrieval` (`503-547`) | 追加 KV 向量召回、标注 `kv_hit_source`、调分 | S2 |
| `rag_modules/hybrid_retrieval.py` | `topic_level_retrieval` (`627-712`) | 追加 KV 向量召回、标注 `kv_hit_source`、调分 | S2 |
| `rag_modules/hybrid_retrieval.py` | `RetrievalResult` (`55-63`) | 新增 `kv_vector_score`/`kv_hit_source`（带默认）| S2 |
| `rag_modules/hybrid_retrieval.py` | `dual_level_retrieval` (`774-841`) | 保留（向后兼容，旧形态）| — |
| `rag_modules/hybrid_retrieval.py` | 新增 `dual_fusion_search` | low/high 进同一 `rrf_fuse` | S3 |
| `rag_modules/graph_indexing.py` | 新增 `_build_kv_vector_records` | 遍历 KV store 产 `KVVectorRecord`（按 4.3）| S2 |
| `rag_modules/graph_indexing.py` | 新增 `get_entities_by_vector` / `get_relations_by_vector` | 向量召回返回 KV 对象 | S2 |
| `rag_modules/graph_indexing.py` | 新增 `KVVectorRecord` dataclass | 见 4.2 | S2 |
| `rag_modules/graph_indexing.py` | `get_entities_by_key`/`get_relations_by_key` (`743-751`) | **不动**（精确命中保留）| — |
| `rag_modules/qdrant_index_construction.py` | 新增 `ensure_kv_collection`/`upsert_kv_vectors`/`search_kv_vectors` | Qdrant KV collection 管理 | S2 |
| `ragmain.py` | `ask_question_with_routing` (`291-461`) | 新增 `dual` 分支 + rerank high-floor（仅 dual）| S3 |
| `ragmain.py` | `update_knowledge_base_incremental` (`574-636`) | 追加 KV 向量 upsert（受影响节点）| S5 |
| `ragmain.py` | `_initialize_retrievers` (`253-268`) | initialize 后触发 KV 向量全量构建（一次性）| S2 |
| `evaluation/run_eval.py` | `--mode choices` (`152`) + `_retrieve_with_k` (`118-126`) | 增 `dual` | S6 |
| `rag_modules/intelligent_query_router.py` | `SearchStrategy` (`22-26`) | 新增 `DUAL_LIGHTRAG`（仅 dual 模式用）| S3 |
| `config.py` | `GraphRAGConfig` 字段 | 新增 `qdrant_kv_collection_name`、`lightrag_llm_keyword_fast_path`、`dual_high_floor`（均带默认）、`to_dict` 同步 | S1/S2/S3 |
| `tests/` | 新增 test_keyword_extraction_fast_path / test_kv_vector_retrieval / test_dual_fusion / test_incremental_kv_upsert | 离线逻辑单测（mock 重依赖，复用 `test_graph_query.py` 的 mock 框架）| S1/S2/S3/S5 |
| `rag_modules/intelligent_query_router.py` `route_query` (`215-245`) | **不动** | intelligent 行为不变 | — |
| `app.py` `QueryRequest` (`124-127`) | **不动** | search_mode 类型不变（枚举扩容靠文档约定）| — |

---

## 7. 评测方案

### 7.1 基线 vs 升级后

评测集 `evaluation/eval_queries.jsonl`（现 10 条，覆盖故障处置/现象→原因/多原因/排查/部件维修/枚举/原因+措施/启动/参数异常/应急处置）。评测不调 LLM 生成，只算检索指标。

**基线（升级前，当前 HEAD）**：
```
python evaluation/run_eval.py --k 3 --k 5 --k 10 --mode intelligent --save
python evaluation/run_eval.py --k 3 --k 5 --k 10 --mode traditional --save
```
基线数字需在升级前先跑一轮**落盘为 `baseline_*.json`**，升级后用同评测集对照。

**升级后**：
```
python evaluation/run_eval.py --k 3 --k 5 --k 10 --mode dual --save          # 新链路
python evaluation/run_eval.py --k 3 --k 5 --k 10 --mode intelligent --save  # 验证旧链路不劣化
```
对照指标（`run_eval.py:summarize`，`129-145`）：`recall_at_k`、`hit_rate_at_k`、`mrr`、`elapsed_seconds`。

**通过判据（父 issue 验收 #2）**：升级后 `dual` 相对基线 `intelligent`/`traditional`，`recall@k`、`mrr` **不劣化**；若 `dual` 在多原因类/枚举类查询（`note` 标注）上 `recall@k` 上升，记为提升证据。

### 7.2 无 Neo4j / Qdrant 时的降级评测

离线可在 CI/无服务环境跑的纯逻辑单测（均 mock 重依赖，复用 `tests/test_graph_query.py` 的 `FakeNode`/`FakeRel`/`_load_ragmain` 注入假模块框架，`tests/test_graph_query.py:32-58`）：

| 单测 | 覆盖 | 不覆盖 |
|---|---|---|
| `test_keyword_extraction_fast_path` | S1 快速路径不调 LLM、抽象查询触发 LLM、降级返回词表 | LLM 输出质量 |
| `test_kv_vector_retrieval` | S2 无精确命中时向量召回成功、`kv_hit_source` 正确、分数单调 | 真实 embedding 召回质量 |
| `test_dual_fusion` | S3 low/high 进同一 RRF、high-floor 在 rerank 后被保留、去重生效 | 端到端 recall@k |
| `test_incremental_kv_upsert` | S5 受影响节点 KV 向量增/删、失败仅告警 | 增量后端到端一致性 |

### 7.3 覆盖度说明（必须写入最终交付文档）

- **离线可验证**：S1/S2/S3/S5 的**逻辑正确性**（控制流、评分、去重、保底、增量副作用）——纯函数 + mock，CI 可跑。
- **离线不可验证、需在线环境**：
  1. LLM 关键词的实际质量（依赖 ollama/vllm 服务）；
  2. KV 向量召回对真实 embedding 的 recall@k/MRR 影响；
  3. `dual` 相对 `intelligent` 的端到端 recall@k/MRR（依赖 Neo4j + Qdrant + 模型就绪，父 issue 验收 #2 依赖此）。
- **结论**：离线逻辑单测**不替代**端到端评测；端到端数字必须在线环境跑（`run_eval.py` 已支持），在最终交付文档里**显式标注「端到端已跑/未跑」与具体环境**。

---

## 8. 决策记录（回应 Stack Scout 提出的开放问题）

> 决策已落到第 3-6 节。此处集中陈述，便于评审。

**Q1：intelligent 是否从「路由二选一」改为「统一双层融合再融合」？**
决策：**否（不强改 intelligent），改为新增并行 `dual` 模式**。`intelligent` 的 `route_query` 仍按复杂度在 `traditional/graph/combined` 间路由（`route_query`、`_rule_based_analysis` 不动），向后兼容；新增 `dual` 直接走 `dual_fusion_search`，作为「检索逻辑对齐 LightRAG」的明面形态、默认评测对象、未来可作默认候选。理由：强改 intelligent 会改变既有路由统计与体验，且 `dual` 的价值在于「始终跑双层」，不应被复杂度路由稀释——并存更清晰、回滚也更易。

**Q2：强 schema 抽取保留、只对齐检索侧——确认「对齐到哪一层」？**
决策：**对齐到「检索与融合层」**，即 LightRAG 的 Profiling KV → 双层检索 → KV 语义匹配 → 去重 → 上下文组织 → 增量闭环。**不动抽取层**（`graph_data_insert.find_or_create_entity`、强领域 schema、实体/关系类型集合保持）。理由：船舶领域 schema 固定且合理，强 schema 对检索反而有利（类型可作过滤/排序信号）；LightRAG 的「light」体现在检索闭环与低资源部署，而非放弃领域 schema。边界：抽取产出什么 KV，检索就消费什么 KV，不反向改抽取。

**Q3：rerank 截断是否给 high-level 保留最少配额（防 CrossEncoder 挤占主题结果）？**
决策：**是，但仅限 `dual` 模式**。`dual` 模式 rerank 后强制保留 ≥ `config.dual_high_floor`（默认 2）条 high-level 结果，其余槽位按 rerank 分排序；`traditional/graph/combined/intelligent` **不引入**此逻辑，行为零变化。理由：CrossEncoder 倾向于奖励与 query 字面高度相关的 low-level 实体片段，high-level 关系主题易被打低而丢失；`dual` 的语义就是「保留主题层」，给 floor 是直接且可调的实现。

**R1：LLM 关键词对简单查询是否走快速路径跳过 LLM？**
决策：**是**。复用 `_fallback_keyword_extraction` 规则词表作为快速路径判据：命中 ≥1 实体关键词且无抽象推理类词时直接返回、不调 LLM；抽象/命中为零则进 LLM；LLM 失败再降级到词表。门控开关 `config.lightrag_llm_keyword_fast_path`（默认开）。理由：本地小模型上 LLM 调用占问答时延显著（ Scout 报告引述 `analyze_query` 同款优化在 `intelligent_query_router:71-88` 已落地），简单实体查找无需 LLM 往返。

**R2：KV collection 生命周期是否与 chunk collection 绑定？**
决策：**是**。KV collection 与 chunk collection **同一 Qdrant 实例、同一 embedding 模型、同一向量维度/距离/hnsw 参数**；不独立维护版本号，靠「全量重建即重灌 KV 向量」「增量即增量 upsert KV 向量」保持一致（与 BM25 缓存 `_bm25_signature` 同模型）。理由：避免新增独立生命周期写入点，复用现有 `build_knowledge_base`/`update_knowledge_base_incremental` 调度。

**R3：`update_knowledge_base_incremental` 是否同步刷新 KV store，需 trace 确认增量闭环？**
决策：**已 trace 确认**。现状：`update_knowledge_base_incremental`（`ragmain.py:574-636`）末尾 `_initialize_retrievers(full_chunks)`（`626`）→ `traditional_retrieval.initialize` → `_build_graph_index` → `create_entity_key_values` → **KV store 已随增量全量重建**。唯一缺口是「新的 KV Qdrant 集合没有随增量 upsert」。S5 补：对受影响节点 entity/relation KV 调 `upsert_kv_vectors`（先删 `record_id` 再 upsert），失败仅告警不阻断。闭环补全。

---

## 9. 验收标准对照（父 issue CESH-1 五条）

| 父验收标准 | 满足路径（本设计）| 证据 |
|---|---|---|
| 1. 检索链路具备 LightRAG 式双层检索（low + high），并说明与原文对应关系 | S1 恢复双层语义入口；S2 KV 向量召回；S3 `dual_fusion_search` 将 low/high 进同一 RRF = 真双层 | 第 4.1 节逐点对照；`dual_fusion_search` 实现 + `test_dual_fusion` |
| 2. `evaluation/` 跑基线 vs 升级后，核心指标不劣化，给出对比数字 | S6 增 `--mode dual`；升前后同评测集跑 `recall@k`/`hit_rate@k`/`mrr` 对比 | 第 7.1 节；落盘 `baseline_*.json` vs `dual_*.json` |
| 3. 现有对外接口（app.py/FastAPI + ragmain.py 入口）行为不破坏 | `ask_question_with_routing` 签名不变；`search_mode` 仅扩容不破坏；`QueryRequest` 不动；`intelligent/traditional/graph/combined` 行为不变 | 第 4.4 节契约；`tests/` 回归 |
| 4. `pytest tests/` 通过；新增/改动检索逻辑补测 | S1/S2/S3/S5 各阶段离线逻辑单测（mock 重依赖）| 第 7.2 节单测清单 |
| 5. 设计文档与最终交付文档产出（本地 docs），含改了什么/为什么/验证证据/后续建议 | 本设计文档 + 实现 PR + 评测报告 + 最终交付文档 | 本文档；第 10 节后续扩展 |

---

## 10. 风险与取舍 / 后续扩展

### 10.1 风险与回滚

| 风险 | 影响 | 缓解 / 回滚 |
|---|---|---|
| S1 恢复 LLM 关键词后，LLM 输出解析失败率高于预期 → 关键词质量回退词表 | 双层入口退化为现状 | 快速路径门控（默认走词表，仅复杂查询走 LLM）+ LLM 失败必降级词表；回滚=重新注释回退 |
| S2 KV 向量集合随全量构建耗时增加启动时间 | 冷启动变慢 | KV 向量构建并发化（复用 `ThreadPoolExecutor`）；落盘可选；回滚=`config` 关 `qdrant_kv_collection_name` 即退回纯精确命中 |
| S2 向量召回引入噪声（同义词过多反而拉低 high-level）| recall 略降 | 向量召回分数乘 0.9 衰减低于精确命中；`dual_high_floor` 兜底；评测对照可定位 |
| S3 `dual` 改变排序后 rerank 与 high-floor 交互异常 | dual 模式结果异常 | `dual` 为新增模式，不影响既有四模式；可单独回滚 `search_mode=="dual"` 分支 |
| S5 增量 upsert KV 向量失败 | 增量后 KV 向量与 store 不一致 | 失败仅告警不阻断 + 增量失败已回退全量重建（`ragmain.py:631-636`）；运维可手动触发 `/api` 重建 |
| 引入新 Qdrant collection 占额外内存 | 资源上升（与「light」张力）| KV 向量集合只存 entity/relation KV 文本向量，量级 ≪ chunk 集合； hnsw 参数复用；可控 |

### 10.2 取舍

- **为什么不删 `_generate_relation_index_keys` 的硬编码主题词表？** 它对 `relation_type` 是稳定且可解释的兜底；删了会回归（现状靠它 high-level 才有召回）。S2 的向量化是**叠加**而非替换——精确主题短语命中优先、向量召回补充。
- **为什么不把 `dual` 设为 `intelligent` 默认？** 强改既有路由体验风险高、统计语义变化、回滚成本高；`dual` 先作为可评测/可选模式上线，验证 recall 持平或上升后再议默认切换（见 10.3）。
- **为什么 KV 向量落在 Qdrant 而非 Neo4j 侧？** 父 issue 不换图库、不引图端向量插件；Qdrant 已是向量库、embedding 模型已在用，复用最轻。

### 10.3 后续扩展建议（不在本次范围）

1. 验证 `dual` 的 `recall@k`/`mrr` 持平或优于 `intelligent` 后，可把 `search_mode` 默认从 `"intelligent"` 切到 `"dual"`（仅一行 `QueryRequest.search_mode` 默认值 + 评测证据）。
2. `enable_llm_relation_keys`（`config.py:93`，默认关）可在 S2 验证后择机开启，让关系索引键由 LLM 生成更自然语言化的主题，进一步对齐 LightRAG high-level。
3. 把 `_fast_entity_match`（`graph_rag_retrieval`，已带 degree 加权）与 KV 向量召回做一次对齐评测，决定是否合并两端重复的实体锚点逻辑。
4. 评测集扩充：当前 10 条偏少，建议补到覆盖每个 `entity_type` 与每个 `relation_type` 各 ≥2 条，使 recall@k 对升级效果更敏感。

---

## 附录 A：本设计对调研结论的复核差异（供 Reviewer 参考）

- Stack Scout 报告路径声明为 `docs/design-docs/active/2026-10-08-lightrag-alignment-gap-analysis.md`，但代码库各分支均未发现该文件（仅 `optimize/kg-visualization` 分支有 KG 可视化设计文档）。本设计**不依赖该文件**，已逐行复核 `hybrid_retrieval.py` / `graph_indexing.py` / `graph_rag_retrieval.py` / `ragmain.py` / `evaluation/run_eval.py` / `config.py`，调研结论与代码事实一致。
- 补充调研未点明但本设计认为关键的点：`graph_rag_retrieval._fast_entity_match`（`rag_modules/graph_rag_retrieval.py:403-465`）**已自带** degree 加权 + 倒排索引 + 缓存的语义式实体锚点，故 LightRAG 的「KV 语义匹配」缺口**专指 KV 引擎（`rag_modules/graph_indexing.py:743-751` 的精确字典命中）**，不应在 graph RAG 引擎重复造轮子——S2 落在 KV 侧正合此判断。

语雀副本：待接入。
