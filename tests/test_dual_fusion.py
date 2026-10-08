from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from lightrag_helpers import Document, load_modules, make_hybrid


def test_same_rrf_preserves_relations_and_deduplicates():
    modules = load_modules()
    hybrid = make_hybrid(modules)
    result = modules.hybrid_retrieval.RetrievalResult
    hybrid.extract_query_keywords = MagicMock(return_value=(["engine"], ["fault"]))
    low = result("engine", "e1", "Equipment", 0.9, "entity", {})
    high1 = result("cause1", "e1", "FaultReason", 0.95, "topic", {"relation_id": "r1"})
    high2 = result("cause2", "e1", "FaultReason", 0.95, "topic", {"relation_id": "r2"})
    hybrid.entity_level_retrieval = MagicMock(return_value=[low, low])
    hybrid.topic_level_retrieval = MagicMock(return_value=[high1, high2, high1])
    docs = hybrid.dual_fusion_search("query", 3)
    hybrid.entity_level_retrieval.assert_called_once_with(["engine"], 6)
    hybrid.topic_level_retrieval.assert_called_once_with(["fault"], 6)
    assert len(docs) == 3
    assert {d.metadata.get("relation_id") for d in docs} == {None, "r1", "r2"}
    assert {tuple(d.metadata["recall_channels"]) for d in docs} == {("low",), ("high",)}
    assert all(d.metadata["rrf_score"] > 0 for d in docs)
    assert hybrid.dual_fusion_search("query", 0) == []


@pytest.mark.parametrize("floor,k,expected", [(2, 3, 2), (9, 1, 1), (0, 3, 0)])
def test_high_floor_after_rerank(floor, k, expected):
    modules = load_modules()
    system = object.__new__(modules.ragmain.AdvancedGraphRAGSystem)
    system.config = SimpleNamespace(dual_high_floor=floor)
    docs = [Document(str(i), {"retrieval_level": "entity"}) for i in range(4)]
    docs += [Document("high" + str(i), {"retrieval_level": "topic"}) for i in range(2)]
    system.rerank_module = SimpleNamespace(rerank=lambda query, candidates, limit: candidates[:limit])
    selected = system._rerank_retrieval("query", docs, k, "dual")
    assert len(selected) == k
    assert sum(d.metadata["retrieval_level"] == "topic" for d in selected) == expected
    assert len({id(d) for d in selected}) == len(selected)
    assert system._rerank_retrieval("query", docs, k, "traditional") == docs[:k]


def test_real_reranker_with_high_candidates_below_cutoff():
    modules = load_modules()
    system = object.__new__(modules.ragmain.AdvancedGraphRAGSystem)
    from config import GraphRAGConfig
    system.config = GraphRAGConfig(dual_high_floor=1)
    system.rerank_module = modules.reranker.RerankModule(system.config)
    scores = {"l0": 20, "l1": 19, "l2": 18, "h0": 1, "h1": 2}
    model = MagicMock()
    model.predict.side_effect = lambda pairs: [scores[text] for query, text in pairs]
    system.rerank_module._model = model
    docs = [Document(name, {"retrieval_level": "topic" if name.startswith("h") else "entity"}) for name in scores]
    result = system._rerank_retrieval("query", docs, 3, "dual")
    assert [d.page_content for d in result] == ["l0", "l1", "h1"]
    assert model.predict.call_count == 2


@pytest.mark.parametrize("mode", ["traditional", "graph", "combined", "intelligent", "dual"])
def test_ask_contract_and_old_routes(mode):
    modules = load_modules()
    system = object.__new__(modules.ragmain.AdvancedGraphRAGSystem)
    from config import GraphRAGConfig
    system.config = GraphRAGConfig(enable_query_rewrite=False)
    system.system_ready = True
    doc = Document("answer", {"retrieval_level": "topic"})
    system.rerank_module = SimpleNamespace(is_enabled=lambda: False)
    system.traditional_retrieval = MagicMock()
    system.traditional_retrieval.hybrid_search.return_value = [doc]
    system.traditional_retrieval.dual_fusion_search.return_value = [doc]
    system.graph_rag_retrieval = MagicMock()
    system.graph_rag_retrieval.graph_rag_search.return_value = [doc]
    system.query_router = MagicMock()
    analysis = modules.intelligent_query_router.QueryAnalysis(0.3, 0.3, False, 1,
        modules.intelligent_query_router.SearchStrategy.HYBRID_TRADITIONAL, 1.0, "test")
    system.query_router.route_query.return_value = ([doc], analysis)
    system.generation_module = MagicMock()
    system.generation_module.generate_adaptive_answer.return_value = "answer"
    system.generation_module.generate_adaptive_answer_stream.return_value = iter(["a", "b"])
    with patch.dict("sys.modules", {"rag_modules.intelligent_query_router": modules.intelligent_query_router}):
        answer, analysis = system.ask_question_with_routing("query", search_mode=mode)
        assert answer == "answer" and analysis is not None
        if mode == "dual":
            assert analysis.recommended_strategy.value == "dual_lightrag"
        stream, _ = system.ask_question_with_routing("query", stream=True, search_mode=mode)
        assert list(stream) == ["a", "b"]
    if mode == "dual":
        system.traditional_retrieval.hybrid_search.assert_not_called()
        system.query_router.route_query.assert_not_called()
    else:
        system.traditional_retrieval.dual_fusion_search.assert_not_called()
