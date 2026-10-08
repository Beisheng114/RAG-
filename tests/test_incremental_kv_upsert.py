from unittest.mock import MagicMock

from lightrag_helpers import load_modules, make_hybrid, seed_graph


def make_system():
    modules = load_modules()
    hybrid = make_hybrid(modules)
    seed_graph(modules, hybrid)
    system = object.__new__(modules.ragmain.AdvancedGraphRAGSystem)
    system.config = hybrid.config
    system.traditional_retrieval = hybrid
    system.data_module = MagicMock()
    system.index_module = MagicMock()
    system.index_module.has_collection.return_value = True
    system._initialize_retrievers = MagicMock()
    system._rebuild_knowledge_base = MagicMock()
    return system, hybrid


def test_affected_records_delete_before_upsert():
    system, hybrid = make_system()
    system.update_knowledge_base_incremental(["e1"])
    index = system.index_module
    ids = index.delete_kv_vectors.call_args.args[0]
    assert "entity:e1" in ids
    assert "relation:e1_HAS_FAULT_e2" in ids
    assert "entity:e2" not in ids
    records = index.upsert_kv_vectors.call_args.args[0]
    assert len(records) == 2
    names = [call[0] for call in index.mock_calls]
    assert names.index("delete_kv_vectors") < names.index("upsert_kv_vectors")
    assert hybrid._kv_incremental is False


def test_removed_relation_deleted_and_failure_warns(caplog):
    system, hybrid = make_system()
    def refresh(chunks):
        assert hybrid._kv_incremental
        hybrid.graph_indexing.relation_kv_store.clear()
    system._initialize_retrievers.side_effect = refresh
    system.index_module.upsert_kv_vectors.side_effect = RuntimeError("offline")
    system.update_knowledge_base_incremental(["e1"])
    assert "relation:e1_HAS_FAULT_e2" in system.index_module.delete_kv_vectors.call_args.args[0]
    assert "增量 KV 向量更新失败" in caplog.text
    system._rebuild_knowledge_base.assert_not_called()


def test_initialize_rebuilds_kv_and_invalidates_caches():
    modules = load_modules()
    hybrid = make_hybrid(modules)
    seed_graph(modules, hybrid)
    hybrid.graph_indexed = True
    hybrid.query_cache["old"] = []
    hybrid.node_info_cache["old"] = "stale"
    def rebuild():
        assert not hybrid.graph_indexed
        assert not hybrid.graph_indexing.entity_kv_store
        hybrid.graph_indexed = True
    hybrid._build_graph_index = MagicMock(side_effect=rebuild)
    hybrid.initialize([])
    assert not hybrid.query_cache and not hybrid.node_info_cache
    hybrid.vector_module.ensure_kv_collection.assert_called_once_with(
        hybrid.config.qdrant_kv_collection_name, force_recreate=True)
    hybrid.vector_module.ensure_kv_collection.reset_mock()
    hybrid._kv_incremental = True
    hybrid.initialize([])
    hybrid.vector_module.ensure_kv_collection.assert_not_called()


def test_full_build_failure_is_nonfatal(caplog):
    system, hybrid = make_system()
    hybrid.graph_indexed = True
    hybrid.vector_module.ensure_kv_collection.side_effect = RuntimeError("offline")
    hybrid._build_kv_vector_index()
    assert "KV 向量索引构建失败" in caplog.text


def test_filtered_one_hop_query_does_not_use_global_limit():
    modules = load_modules()
    hybrid = make_hybrid(modules)
    hybrid.driver = MagicMock()
    session = hybrid.driver.session.return_value.__enter__.return_value
    session.run.return_value = [{"source_id": "e1", "relation_type": "HAS_FAULT", "target_id": "e2"}]
    assert hybrid._extract_relationships_from_graph(["e1"]) == [("e1", "HAS_FAULT", "e2")]
    query, params = session.run.call_args.args
    assert params == {"ids": ["e1"]}
    assert "toString(source_id) IN $ids" in query and "toString(target_id) IN $ids" in query
    assert "LIMIT 1000" not in query


def test_new_entity_and_incident_relation_upsert():
    system, hybrid = make_system()
    modules = load_modules()
    def refresh(chunks):
        hybrid.graph_indexing.entity_kv_store["e3"] = modules.graph_indexing.EntityKeyValue(
            "bearing", ["bearing"], "new bearing", "Component", {"neo4j_node_id": "e3"})
    system._initialize_retrievers.side_effect = refresh
    hybrid._extract_relationships_from_graph = MagicMock(return_value=[("e1", "HAS_COMPONENT", "e3")])
    system.update_knowledge_base_incremental(["e3"])
    records = system.index_module.upsert_kv_vectors.call_args.args[0]
    assert {r.record_id for r in records} == {"entity:e3", "relation:e1_HAS_COMPONENT_e3"}
    assert "bearing" in hybrid.graph_indexing.key_to_entities
