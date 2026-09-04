"""
RRF（Reciprocal Rank Fusion）融合的纯函数实现

从 hybrid_retrieval 中抽出，便于单元测试与复用。
输入多路已排序的候选列表，输出按融合分数排序的结果。
"""
from typing import Any, Callable, Dict, List


def rrf_score(rank: int, k: int = 60) -> float:
    """单个候选在单路结果中的 RRF 得分（rank 从 1 开始）"""
    return 1.0 / (k + rank)


def rrf_fuse(
    channel_docs: Dict[str, List[Any]],
    doc_id_fn: Callable[[Any], str],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """多路候选 RRF 融合

    Args:
        channel_docs: {"vector": [doc...], "bm25": [doc...], "graph": [doc...]}
            每路列表按该路自身的相关性降序排列
        doc_id_fn: 文档唯一标识函数（如 neo4j_node_id）
        k: RRF 平滑常数，常用 60

    Returns:
        [{"doc": 原文档, "score": 融合分, "channels": ["vector", "bm25"]}, ...]
        按融合分降序
    """
    fused: Dict[str, Dict[str, Any]] = {}

    for channel, docs in channel_docs.items():
        for rank, doc in enumerate(docs, start=1):
            doc_id = doc_id_fn(doc)
            if doc_id not in fused:
                fused[doc_id] = {"doc": doc, "score": 0.0, "channels": set()}
            fused[doc_id]["score"] += rrf_score(rank, k)
            fused[doc_id]["channels"].add(channel)

    ranked = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
    for item in ranked:
        item["channels"] = sorted(item["channels"])
    return ranked
