"""
检索质量评测脚本

对评测集逐条执行检索（不调用LLM生成，节省算力且可离线评估），
输出 recall@k / 命中率@k / MRR 指标，为 top_k、chunk_size、路由阈值等
调参提供量化依据（配合调大/调小参数对比运行）。

用法（在项目根目录执行）：
    python evaluation/run_eval.py                 # 默认 top_k=5，intelligent 路由
    python evaluation/run_eval.py --k 3 --k 5     # 多个k对比
    python evaluation/run_eval.py --mode traditional
    python evaluation/run_eval.py --queries evaluation/eval_queries.jsonl

评测集格式（JSONL，每行一条）：
    {"query": "...", "expected_keywords": ["关键词1", "关键词2"], "note": "可选备注"}

结果输出到 evaluation/results/eval_report.json（目录已加入 .gitignore）。

依赖：需先完成知识库构建（Neo4j + Qdrant + 模型就绪）。
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# 支持从项目根目录或 evaluation/ 目录运行
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from ragmain import AdvancedGraphRAGSystem  # noqa: E402


def load_queries(path: str):
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"⚠️ 第 {line_no} 行解析失败，已跳过: {e}")
                continue
            if "query" not in item or "expected_keywords" not in item:
                print(f"⚠️ 第 {line_no} 行缺少 query/expected_keywords 字段，已跳过")
                continue
            queries.append(item)
    return queries


def doc_text(doc) -> str:
    """拼接文档正文与元数据实体名，用于关键词命中判定"""
    parts = [getattr(doc, "page_content", "")]
    metadata = getattr(doc, "metadata", {}) or {}
    parts.append(str(metadata.get("entity_name", "")))
    return "\n".join(parts)


def evaluate_query(system, query_item: dict, k: int, mode: str) -> dict:
    """对单条评测项执行检索并计算指标"""
    query = query_item["query"]
    keywords = [kw for kw in query_item["expected_keywords"] if kw]
    expected = set(keywords)

    docs, _analysis = _retrieve(system, query, k, mode)

    ranked_texts = [doc_text(d) for d in (docs or [])]

    # 每个关键词首次命中的排名（1-based）
    first_hit_rank = {}
    for kw in expected:
        for rank, text in enumerate(ranked_texts, start=1):
            if kw in text:
                first_hit_rank[kw] = rank
                break

    hit_keywords = set(first_hit_rank.keys())
    # 首个命中的文档排名 → MRR
    doc_first_hit = None
    for rank, text in enumerate(ranked_texts, start=1):
        if any(kw in text for kw in expected):
            doc_first_hit = rank
            break

    return {
        "query": query,
        "k": k,
        "returned": len(ranked_texts),
        "keyword_hits": len(hit_keywords),
        "keyword_total": len(expected),
        "hit_keywords": sorted(hit_keywords),
        "missed_keywords": sorted(expected - hit_keywords),
        "first_hit_doc_rank": doc_first_hit,
    }


def _retrieve(system, query: str, k: int, mode: str):
    """直接走检索链路（复用 ask_question_with_routing 的路由+改写+精排逻辑，
    但避免触发生成）。检索分支与主链路共享同一实现。"""
    saved_top_k = system.config.top_k
    try:
        # 临时调整 top_k 以评测不同 k 值
        system.config.top_k = k
        if system.rerank_module and system.rerank_module.is_enabled():
            recall_k = max(k, int(getattr(system.config, "rerank_candidate_k", 20)))
            docs, analysis = _retrieve_with_k(system, query, recall_k, mode)
            docs = system.rerank_module.rerank(query, docs, k)
            return docs, analysis
        return _retrieve_with_k(system, query, k, mode)
    finally:
        system.config.top_k = saved_top_k


def _retrieve_with_k(system, query: str, k: int, mode: str):
    if mode == "traditional":
        docs = system.traditional_retrieval.hybrid_search(query, k)
        return docs, None
    if mode == "graph":
        docs = system.graph_rag_retrieval.graph_rag_search(query, k)
        return docs, None
    docs, analysis = system.query_router.route_query(query, k)
    return docs, analysis


def summarize(results) -> dict:
    """聚合指标：recall@k / 命中率@k / MRR"""
    n = len(results)
    if n == 0:
        return {}
    total_kw = sum(r["keyword_total"] for r in results)
    hit_kw = sum(r["keyword_hits"] for r in results)
    hit_queries = sum(1 for r in results if r["keyword_hits"] > 0)
    rr_sum = sum(
        1.0 / r["first_hit_doc_rank"] for r in results if r["first_hit_doc_rank"]
    )
    return {
        "num_queries": n,
        "recall_at_k": round(hit_kw / total_kw, 4) if total_kw else 0.0,
        "hit_rate_at_k": round(hit_queries / n, 4),
        "mrr": round(rr_sum / n, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="检索质量评测（recall@k / MRR）")
    parser.add_argument("--queries", default="evaluation/eval_queries.jsonl", help="评测集路径")
    parser.add_argument("--k", type=int, action="append", default=None, help="评测的k值，可多次指定对比")
    parser.add_argument("--mode", default="intelligent", choices=["intelligent", "traditional", "graph"], help="检索模式")
    parser.add_argument("--save", action="store_true", help="保存报告到 evaluation/results/")
    args = parser.parse_args()

    queries = load_queries(args.queries)
    if not queries:
        print("评测集为空，请检查 JSONL 文件格式")
        sys.exit(1)
    print(f"加载评测集: {len(queries)} 条（来源: {args.queries}）")

    k_list = args.k or [5]

    print("初始化RAG系统...")
    system = AdvancedGraphRAGSystem()
    system.initialize_system()
    system.build_knowledge_base()

    all_results = {}
    for k in k_list:
        print(f"\n===== 评测 k={k} (模式: {args.mode}) =====")
        results = []
        start = time.time()
        for item in queries:
            r = evaluate_query(system, item, k, args.mode)
            results.append(r)
            marker = "✓" if r["keyword_hits"] else "✗"
            print(f"{marker} {r['query']}  命中 {r['keyword_hits']}/{r['keyword_total']}  首中排名: {r['first_hit_doc_rank']}")
        elapsed = time.time() - start
        summary = summarize(results)
        summary["elapsed_seconds"] = round(elapsed, 1)
        summary["k"] = k
        summary["mode"] = args.mode
        all_results[f"k={k}"] = {"summary": summary, "details": results}
        print(f"--- 汇总(k={k}) ---")
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.save:
        out_dir = Path("evaluation/results")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"eval_report_{ts}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n报告已保存: {out_path}")


if __name__ == "__main__":
    main()
