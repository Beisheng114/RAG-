from unittest.mock import MagicMock

from lightrag_helpers import load_modules, make_hybrid, seed_graph


def test_vector_only_and_exact_overlap():
    modules = load_modules()
    hybrid = make_hybrid(modules)
    graph = seed_graph(modules, hybrid)
    graph.vector_module.search_kv_vectors.return_value = [{"source_id": "e1", "score": 0.98}]
    result = hybrid.entity_level_retrieval(["主机"], 1)[0]
    assert result.kv_hit_source == "vector"
    assert result.kv_vector_score == 0.98
    assert result.relevance_score == 0.98 * 0.9
    result = hybrid.entity_level_retrieval(["主柴油机", "主机"], 2)
    assert len(result) == 1
    assert result[0].kv_hit_source == "exact+vector"
    assert result[0].relevance_score == 0.9
    assert "kv_vector_score" not in graph.entity_kv_store["e1"].metadata


def test_relation_vector_and_exact_overlap():
    modules = load_modules()
    hybrid = make_hybrid(modules)
    graph = seed_graph(modules, hybrid)
    rid = next(iter(graph.relation_kv_store))
    graph.vector_module.search_kv_vectors.return_value = [{"source_id": rid, "score": 0.97}]
    result = hybrid.topic_level_retrieval(["引擎热负荷异常"], 1)[0]
    assert result.metadata["relation_id"] == rid
    assert result.relevance_score == 0.97 * 0.9
    assert result.kv_hit_source == "vector"
    keyword = graph.relation_kv_store[rid].index_keys[0]
    exact = hybrid.topic_level_retrieval([keyword], 1)[0]
    assert exact.kv_hit_source == "exact+vector"
    assert exact.relevance_score == 0.95


def test_vector_failure_and_disabled_bypass():
    modules = load_modules()
    hybrid = make_hybrid(modules)
    graph = seed_graph(modules, hybrid)
    graph.vector_module.search_kv_vectors.side_effect = RuntimeError("offline")
    assert hybrid.entity_level_retrieval(["主柴油机"], 1)[0].kv_hit_source == "exact"
    hybrid.config.qdrant_kv_collection_name = ""
    graph.vector_module.search_kv_vectors.reset_mock()
    assert graph.get_entities_by_vector("主机") == []
    graph.vector_module.search_kv_vectors.assert_not_called()


def test_kv_record_text_and_affected_filter():
    modules = load_modules()
    hybrid = make_hybrid(modules)
    graph = seed_graph(modules, hybrid)
    records = graph._build_kv_vector_records(["e1"])
    assert len(records) == 2
    assert records[0].text == "主柴油机\n主柴油机说明"
    assert records[1].text.startswith("HAS_FAULT\n主柴油机→排气温度过高\n")
    assert graph._build_kv_vector_records(["missing"]) == []


def test_qdrant_collection_embedding_upsert_delete_and_filter():
    modules = load_modules()
    hybrid = make_hybrid(modules)
    graph = seed_graph(modules, hybrid)
    qm = modules.qdrant_index_construction
    index = object.__new__(qm.QdrantIndexConstructionModule)
    index.config = qm.QdrantConfig(vector_size=2)
    index.client = MagicMock()
    index.client.collection_exists.return_value = False
    index.embeddings = MagicMock()
    records = graph._build_kv_vector_records()
    index.embeddings.embed_documents.return_value = [[1.0, 0.0]] * len(records)
    assert index.ensure_kv_collection(force_recreate=True)
    index.upsert_kv_vectors(records)
    points = index.client.upsert.call_args.kwargs["points"]
    assert len(points) == 3
    assert points[0].id == index._kv_point_id(records[0].record_id)
    assert points[0].payload["source_id"] == "e1"
    index.delete_kv_vectors([records[0].record_id])
    assert index.client.delete.call_args.kwargs["points_selector"].points == [points[0].id]
    index.client.query_points.return_value.points = [type("Hit", (), {"payload": {"source_id": "e1"}, "score": 0.98})()]
    assert index.search_kv_vectors("主机", "low", 1) == [{"source_id": "e1", "score": 0.98}]
    condition = index.client.query_points.call_args.kwargs["query_filter"].must[0]
    assert condition.key == "level" and condition.match.value == "low"


def test_collection_recreation_and_same_name_guard():
    import pytest
    qm = load_modules().qdrant_index_construction
    index = object.__new__(qm.QdrantIndexConstructionModule)
    index.config = qm.QdrantConfig()
    index.client = MagicMock()
    index.client.collection_exists.return_value = True
    index.ensure_kv_collection(force_recreate=True)
    index.client.delete_collection.assert_called_once_with(index.config.kv_collection_name)
    with pytest.raises(ValueError):
        index.ensure_kv_collection(index.config.collection_name, force_recreate=True)


def test_delete_chunk_collection_also_deletes_kv():
    qm = load_modules().qdrant_index_construction
    index = object.__new__(qm.QdrantIndexConstructionModule)
    index.config = qm.QdrantConfig()
    index.client = MagicMock()
    index.client.collection_exists.return_value = True
    assert index.delete_collection()
    deleted = [c.args[0] for c in index.client.delete_collection.call_args_list]
    assert deleted == [index.config.collection_name, index.config.kv_collection_name]


def test_legacy_qdrant_search_and_config_roundtrip():
    from types import SimpleNamespace
    from config import GraphRAGConfig
    qm = load_modules().qdrant_index_construction
    index = object.__new__(qm.QdrantIndexConstructionModule)
    index.config = qm.QdrantConfig()
    index.client = SimpleNamespace(search=MagicMock(return_value=[SimpleNamespace(
        payload={"source_id": "r1"}, score=0.9)]))
    index.embeddings = MagicMock()
    assert index.search_kv_vectors("query", "high", 2) == [{"source_id": "r1", "score": 0.9}]
    assert index.client.search.call_args.kwargs["query_vector"] is index.embeddings.embed_query.return_value
    config = GraphRAGConfig(qdrant_kv_collection_name="custom_kv", dual_high_floor=1,
                            lightrag_llm_keyword_fast_path=False)
    roundtrip = GraphRAGConfig.from_dict(config.to_dict())
    assert roundtrip.qdrant_kv_collection_name == "custom_kv"
    assert roundtrip.dual_high_floor == 1 and not roundtrip.lightrag_llm_keyword_fast_path
