"""Controlled retrieval regression fixture, NOT an end-to-end quality benchmark.

Uses real baseline/current retrieval and run_eval metrics, but mocked vector
scores, query keywords, graph responses, and import-only heavy dependencies.
"""
import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from lightrag_helpers import load_modules, make_hybrid, seed_graph
from evaluation.run_eval import evaluate_query, summarize

BASELINE = "7a614abf7723f0d18285589589cda66beda134f7"


def compare():
    modules = load_modules()
    source = subprocess.check_output(
        ["git", "show", BASELINE + ":rag_modules/hybrid_retrieval.py"], cwd=ROOT, text=True
    )
    tree = ast.parse(source)
    baseline_class = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                          and node.name == "HybridRetrievalModule")
    namespace = dict(vars(modules.hybrid_retrieval))
    exec(compile(ast.Module(body=[baseline_class], type_ignores=[]), "baseline_hybrid.py", "exec"), namespace)
    baseline_modules = SimpleNamespace(hybrid_retrieval=SimpleNamespace(
        HybridRetrievalModule=namespace["HybridRetrievalModule"]))
    baseline = make_hybrid(baseline_modules)
    upgraded = make_hybrid(modules)
    scenarios = [
        ("主柴油机", ["主柴油机"], [], "主柴油机"),
        ("主机", ["主机"], [], "主柴油机"),
        ("设备故障", [], ["设备故障"], "排气温度过高"),
        ("热负荷异常", [], ["热负荷异常"], "排气温度过高"),
        ("无关问题", ["无关问题"], [], "不存在的知识"),
    ]
    keywords = {query: (low, high) for query, low, high, _ in scenarios}
    for hybrid in (baseline, upgraded):
        graph = seed_graph(modules, hybrid)
        hybrid.extract_query_keywords = lambda query: keywords[query]
        hybrid.vector_module.hybrid_search.return_value = []
        hybrid.vector_module.search_kv_vectors.side_effect = lambda query, level, k: (
            [{"source_id": "e1", "score": 0.98}] if level == "low" and query in ("主机", "主柴油机") else
            [{"source_id": "e1_HAS_FAULT_e2", "score": 0.97}] if level == "high" and query in ("热负荷异常", "设备故障") else []
        )
    queries = [{"query": query, "expected_keywords": [expected]} for query, _, _, expected in scenarios]
    metrics = {}
    for mode, hybrid in (("baseline_traditional", baseline), ("upgraded_dual", upgraded)):
        system = SimpleNamespace(config=hybrid.config, traditional_retrieval=hybrid, rerank_module=None)
        metrics[mode] = {}
        for k in (3, 5, 10):
            results = [evaluate_query(system, item, k, "traditional" if mode.startswith("baseline") else "dual")
                       for item in queries]
            metrics[mode][str(k)] = {"summary": summarize(results), "details": results}
    report = {
        "kind": "controlled_mock_regression_not_end_to_end",
        "baseline_commit": BASELINE,
        "environment": {"python": sys.version.split()[0], "neo4j": "mock", "qdrant": "mock",
                        "embedding": "fixed_scores", "llm_keywords": "fixed", "rerank": "disabled"},
        "coverage": ["exact entity/relation", "vector-only entity/relation", "empty result", "real RRF", "run_eval metrics"],
        "not_covered": ["evaluation/eval_queries.jsonl", "real embedding quality", "LLM quality",
                        "online intelligent/traditional/dual non-regression", "real service consistency"],
        "end_to_end_metrics": None,
        "online_blocker": "baseline run_eval failed with ModuleNotFoundError: neo4j; local models/services not provisioned",
        "metrics_by_mode": metrics,
    }
    for hybrid in (baseline, upgraded):
        hybrid.executor.shutdown(wait=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"kind": report["kind"], "metrics_by_mode": {
        mode: {k: v["summary"] for k, v in runs.items()} for mode, runs in report["metrics_by_mode"].items()
    }}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
