from evaluation.offline_retrieval_check import compare


def test_baseline_snapshot_and_real_metric_aggregation():
    report = compare()
    assert report["end_to_end_metrics"] is None
    for k in ("3", "5", "10"):
        baseline = report["metrics_by_mode"]["baseline_traditional"][k]["summary"]
        dual = report["metrics_by_mode"]["upgraded_dual"][k]["summary"]
        assert baseline["num_queries"] == dual["num_queries"] == 5
        assert baseline["recall_at_k"] == baseline["mrr"] == 0.4
        assert dual["recall_at_k"] == dual["mrr"] == 0.8
