"""图谱查询/邻居接口单测（mock driver，不依赖 Neo4j/Qdrant）

覆盖：
1. query_graph 分页（offset 透传、负数 clamp、has_more 判定、SKIP 进 Cypher）
2. get_node_neighbors（参数构造、limit clamp、三元组去重）
3. 路由层（graph_routes 直载，FastAPI 局部 app + mock get_rag_system）

运行：pytest tests/test_graph_query.py -v
"""
import importlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from conftest import PROJECT_ROOT


# ---------- 测试基础设施 ----------

class FakeNode:
    """模拟 neo4j.Node 的最小接口（id/labels/get）"""

    def __init__(self, nid, labels, props=None):
        self.id = nid
        self._labels = set(labels)
        self._props = props or {}

    @property
    def labels(self):
        return self._labels

    def get(self, key, default=None):
        return self._props.get(key, default)


class FakeRel:
    """模拟 neo4j.Relationship（start_node/end_node/type）"""

    def __init__(self, start, end, rtype):
        self.start_node = start
        self.end_node = end
        self.type = rtype


def _load_ragmain():
    """离线加载 ragmain：预置假的重依赖模块（rag_modules 全家桶）。

    仅注入 sys.modules 假模块使 import 成功；被测方法
    （query_graph / get_node_neighbors / _ingest_triplet）均为纯逻辑 +
    mock driver，不受假模块影响。CI/容器中有真实依赖时也兼容
    （仅当模块缺失时才注入）。
    """
    needed = [
        "rag_modules",
        "rag_modules.qdrant_index_construction",
        "rag_modules.hybrid_retrieval",
        "rag_modules.graph_rag_retrieval",
        "rag_modules.intelligent_query_router",
        "rag_modules.graph_data_insert",
        "rag_modules.reranker",
    ]
    injected = {}
    for name in needed:
        if name not in sys.modules:
            injected[name] = MagicMock()
            sys.modules[name] = injected[name]
    import ragmain  # noqa: E402
    return ragmain


def make_rag(records):
    """构造绕过 __init__ 的 RAG 系统实例（mock driver 返回给定 records）"""
    ragmain = _load_ragmain()
    sys_obj = object.__new__(ragmain.AdvancedGraphRAGSystem)
    sys_obj._calls = []

    session = MagicMock()

    def run_side_effect(cypher, params=None, **kw):
        sys_obj._calls.append((cypher, params or kw))
        return iter(list(records))

    session.run.side_effect = run_side_effect
    driver = MagicMock()
    driver.session.return_value.__enter__.return_value = session
    sys_obj.data_module = SimpleNamespace(driver=driver)
    sys_obj.config = SimpleNamespace(max_graph_depth=4)
    return sys_obj


def chain_records(pairs, rtype="HAS_FAULT"):
    """由 (a, b) 节点 id 对生成三元组 records（节点自带 name 属性）"""
    node_cache = {}

    def node(nid):
        if nid not in node_cache:
            node_cache[nid] = FakeNode(nid, ["Equipment"], {"name": f"设备{nid}", "system_name": "动力系统"})
        return node_cache[nid]

    return [
        {"n": node(a), "r": FakeRel(node(a), node(b), rtype), "m": node(b)}
        for a, b in pairs
    ]


# ---------- query_graph 分页 ----------

class TestQueryGraphPagination:
    def test_offset_passed_and_skip_in_cypher(self):
        sys_obj = make_rag(chain_records([(1, 2), (3, 4)]))
        nodes, edges, stats = sys_obj.query_graph("", "all", 100, "all", 150)
        cypher, params = sys_obj._calls[0]
        assert params["offset"] == 150
        assert "SKIP $offset" in cypher
        assert len(nodes) == 4 and len(edges) == 2

    def test_negative_offset_clamped_to_zero(self):
        sys_obj = make_rag(chain_records([(1, 2)]))
        sys_obj.query_graph("", "all", 100, "all", -50)
        _, params = sys_obj._calls[0]
        assert params["offset"] == 0

    def test_has_more_true_when_full_page(self):
        # node_limit=50 → browse_edge_lim = min(4000, max(200, 250)) = 250
        records = chain_records([(i, i + 1) for i in range(250)])
        sys_obj = make_rag(records)
        _, _, stats = sys_obj.query_graph("", "all", 50, "all", 0)
        assert stats["has_more"] is True

    def test_has_more_false_when_partial_page(self):
        records = chain_records([(i, i + 1) for i in range(10)])
        sys_obj = make_rag(records)
        _, _, stats = sys_obj.query_graph("", "all", 50, "all", 0)
        assert stats["has_more"] is False

    def test_query_mode_not_paginated_has_more_false(self):
        records = chain_records([(1, 2), (2, 3)])
        sys_obj = make_rag(records)
        _, _, stats = sys_obj.query_graph("电机", "all", 100, "all", 100)
        assert stats["has_more"] is False
        # 关键词模式 Cypher 不带 SKIP 分页
        cypher, params = sys_obj._calls[0]
        assert "SKIP $offset" not in cypher


# ---------- get_node_neighbors ----------

class TestNodeNeighbors:
    def test_params_and_result(self):
        records = chain_records([(7, 8), (7, 9)])
        sys_obj = make_rag(records)
        nodes, edges, stats = sys_obj.get_node_neighbors(7, 30)
        cypher, params = sys_obj._calls[0]
        assert "id(n) = $nid" in cypher
        assert params == {"nid": 7, "lim": 30}
        assert len(nodes) == 3 and len(edges) == 2
        assert stats["anchor_id"] == "7"
        assert stats["total_nodes"] == 3 and stats["total_edges"] == 2

    def test_limit_clamped_to_100(self):
        sys_obj = make_rag([])
        sys_obj.get_node_neighbors(1, 500)
        _, params = sys_obj._calls[0]
        assert params["lim"] == 100

    def test_dedup_bidirectional_edges(self):
        """A->B 与 B->A 视为同一条边（无向可视化去重）"""
        a = FakeNode(1, ["Equipment"], {"name": "主机"})
        b = FakeNode(2, ["Fault"], {"name": "故障A"})
        records = [
            {"n": a, "r": FakeRel(a, b, "HAS_FAULT"), "m": b},
            {"n": b, "r": FakeRel(b, a, "HAS_FAULT"), "m": a},
        ]
        sys_obj = make_rag(records)
        nodes, edges, stats = sys_obj.get_node_neighbors(1, 20)
        assert len(nodes) == 2
        assert len(edges) == 1  # 去重后仅一条
        assert edges[0]["label"] == "HAS_FAULT"

    def test_node_to_item_fields(self):
        a = FakeNode(11, ["FaultReason"], {"cause_name": "燃油杂质"})
        b = FakeNode(12, ["Fault"], {"name": "主机故障"})
        records = [{"n": a, "r": FakeRel(a, b, "CAUSED_BY"), "m": b}]
        sys_obj = make_rag(records)
        nodes, _, _ = sys_obj.get_node_neighbors(11, 20)
        reason = next(n for n in nodes if n["type"] == "FaultReason")
        assert reason["label"] == "燃油杂质"
        assert reason["id"] == "11"
        assert "燃油杂质 (FaultReason)" in reason["title"]


# ---------- 路由层（FastAPI 局部 app + mock get_rag_system） ----------

class TestGraphRoutes:
    @pytest.fixture()
    def client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        import routers.graph_routes as gr
        app = FastAPI()
        app.include_router(gr.router)
        return TestClient(app, raise_server_exceptions=False)

    @staticmethod
    def _mock_rag_system():
        rag = MagicMock()
        rag.query_graph.return_value = (
            [{"id": "1", "label": "主机", "type": "Equipment", "title": "t", "system_name": None}],
            [{"from": "1", "to": "2", "label": "HAS_FAULT"}],
            {"total_nodes": 1, "total_edges": 1, "has_more": False, "node_types": {"Equipment": 1}},
        )
        rag.get_node_neighbors.return_value = (
            [{"id": "1", "label": "主机", "type": "Equipment", "title": "t", "system_name": None}],
            [{"from": "1", "to": "2", "label": "HAS_FAULT"}],
            {"total_nodes": 1, "total_edges": 1, "anchor_id": "1"},
        )
        return rag

    def test_query_route_passes_offset(self, client):
        rag = self._mock_rag_system()
        with patch("services.graph_service.get_rag_system", return_value=rag):
            resp = client.post("/api/graph/query", json={
                "query": "", "entity_type": "all", "node_limit": 100, "offset": 120,
            })
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["stats"]["has_more"] is False
        assert len(body["nodes"]) == 1 and body["nodes"][0]["id"] == "1"
        # offset 透传到 rag_system.query_graph 第 5 个位置参数
        args = rag.query_graph.call_args.args
        assert args[4] == 120

    def test_query_route_clamps_negative_offset(self, client):
        rag = self._mock_rag_system()
        with patch("services.graph_service.get_rag_system", return_value=rag):
            resp = client.post("/api/graph/query", json={
                "query": "", "offset": -9,
            })
        assert resp.status_code == 200
        assert rag.query_graph.call_args.args[4] == 0

    def test_neighbors_route(self, client):
        rag = self._mock_rag_system()
        with patch("services.graph_service.get_rag_system", return_value=rag):
            resp = client.get("/api/graph/node/42/neighbors", params={"limit": 15})
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["stats"]["anchor_id"] == "1"
        assert body["edges"][0]["label"] == "HAS_FAULT"
        rag.get_node_neighbors.assert_called_once_with(42, 15)

    def test_query_route_uninitialized_returns_failure(self, client):
        with patch("services.graph_service.get_rag_system",
                   side_effect=RuntimeError("RAG 系统尚未初始化")):
            resp = client.post("/api/graph/query", json={"query": ""})
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is False
        assert body["nodes"] == []
