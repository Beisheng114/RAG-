"""
最小冒烟测试（不依赖 Neo4j/Qdrant/模型等外部服务）

覆盖 issue #4 建议的三类核心逻辑：
1. 配置加载与约束（config）
2. 安全模块（admin token / API key / CORS / 打码）
3. RRF 融合纯函数（core/rrf）
4. SQLite 对话存储（services/conversation_store，含旧 JSON 迁移）

运行：pytest tests/ -v
"""
import json
import os
import sqlite3
import threading

import pytest


# ---------- config ----------

class TestConfig:
    def test_default_config_loads(self):
        from config import GraphRAGConfig, DEFAULT_CONFIG
        assert DEFAULT_CONFIG.top_k == 5
        assert DEFAULT_CONFIG.qdrant_vector_size == 768  # bge-base-zh-v1.5

    def test_milvus_rejected(self):
        from config import GraphRAGConfig
        with pytest.raises(ValueError):
            GraphRAGConfig(vector_index_type="milvus")

    def test_no_hardcoded_password(self):
        """修复 issue #1：代码中不得保留默认密码"""
        import config
        from config import GraphRAGConfig
        src = open(config.__file__, encoding="utf-8").read()
        assert "myrag123456" not in src
        assert GraphRAGConfig().neo4j_password == ""

    def test_to_dict_fields(self):
        from config import GraphRAGConfig
        d = GraphRAGConfig().to_dict()
        # 已删除的 milvus/faiss 字段不应回归
        for k in ("milvus_host", "milvus_dimension", "faiss_index_path"):
            assert k not in d
        # issue #3 新增的精排/改写配置应导出
        for k in ("enable_rerank", "rerank_candidate_k", "enable_query_rewrite"):
            assert k in d


# ---------- security ----------

class TestSecurity:
    def test_admin_disabled_without_env(self, monkeypatch):
        monkeypatch.delenv("RAG_ADMIN_TOKEN", raising=False)
        from core import security
        assert security.get_admin_token() is None
        assert security.is_admin_enabled() is False

    def test_admin_503_when_not_configured(self, monkeypatch):
        monkeypatch.delenv("RAG_ADMIN_TOKEN", raising=False)
        from fastapi import HTTPException
        from core import security
        with pytest.raises(HTTPException) as exc:
            security.require_admin(x_admin_token=None)
        assert exc.value.status_code == 503

    def test_admin_token_verification(self, monkeypatch):
        monkeypatch.setenv("RAG_ADMIN_TOKEN", "secret-token")
        from fastapi import HTTPException
        from core import security
        assert security.require_admin(x_admin_token="secret-token") is True
        with pytest.raises(HTTPException) as exc:
            security.require_admin(x_admin_token="wrong")
        assert exc.value.status_code == 401

    def test_api_key_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("RAG_API_KEY", raising=False)
        from core import security

        class EmptyHeaders:
            def get(self, name):
                return None

        assert security.check_api_access(EmptyHeaders()) is True

    def test_api_key_guard(self, monkeypatch):
        monkeypatch.setenv("RAG_API_KEY", "key-123")
        from core import security

        class Headers:
            def __init__(self, **kw):
                self._kw = kw

            def get(self, name):
                return self._kw.get(name)

        # 正确 key / 管理 token / 错误 key
        assert security.check_api_access(Headers(**{"X-API-Key": "key-123"})) is True
        assert security.check_api_access(Headers()) is False

    def test_cors_default_whitelist(self, monkeypatch):
        monkeypatch.delenv("RAG_CORS_ORIGINS", raising=False)
        from core import security
        origins = security.get_cors_origins()
        assert "*" not in origins
        assert any("localhost" in o for o in origins)

    def test_cors_env_override(self, monkeypatch):
        monkeypatch.setenv("RAG_CORS_ORIGINS", "https://a.com, https://b.com")
        from core import security
        assert security.get_cors_origins() == ["https://a.com", "https://b.com"]

    def test_mask_secret(self):
        from core.security import mask_secret
        assert mask_secret("myrag123456") == "myr******56"
        assert mask_secret("") == ""
        assert mask_secret("ab") == "**"
        assert mask_secret("abc") == "***"


# ---------- RRF ----------

class TestRRF:
    def test_rrf_score_value(self):
        from core.rrf import rrf_score
        assert rrf_score(1) == 1.0 / 61
        assert rrf_score(1, k=60) > rrf_score(2, k=60)

    def test_fuse_multi_channel_beats_single(self):
        """多路同时命中的文档应排在单路命中的前面"""
        from core.rrf import rrf_fuse

        class Doc:
            def __init__(self, i):
                self.id = f"d{i}"

        docs_v = [Doc(1), Doc(2)]
        docs_b = [Doc(3), Doc(1)]  # Doc1 两路都命中
        docs_g = [Doc(4)]

        ranked = rrf_fuse(
            {"vector": docs_v, "bm25": docs_b, "graph": docs_g},
            doc_id_fn=lambda d: d.id,
        )
        assert ranked[0]["doc"].id == "d1"
        assert set(ranked[0]["channels"]) == {"vector", "bm25"}
        # 分数应为两路之和
        from core.rrf import rrf_score
        expected = rrf_score(1) + rrf_score(2)
        assert abs(ranked[0]["score"] - expected) < 1e-12

    def test_fuse_empty_channels(self):
        from core.rrf import rrf_fuse
        assert rrf_fuse({}, doc_id_fn=str) == []

    def test_fuse_dedup_by_doc_id(self):
        """同 id 在不同 rank 合并累加，不重复占位"""
        from core.rrf import rrf_fuse

        class Doc:
            def __init__(self, i):
                self.id = str(i)

        ranked = rrf_fuse(
            {"a": [Doc(1), Doc(1)], "b": [Doc(2)]},  # 同路同 id 重复也合并
            doc_id_fn=lambda d: d.id,
        )
        ids = [r["doc"].id for r in ranked]
        assert len(ids) == len(set(ids))


# ---------- SQLite 对话存储 ----------

class TestConversationStore:
    @pytest.fixture()
    def store(self, tmp_path):
        from services.conversation_store import ConversationStore
        s = ConversationStore(str(tmp_path / "test.db"))
        yield s
        s.close()

    def test_crud_roundtrip(self, store):
        conv = {
            "id": "c1",
            "title": "测试对话",
            "created_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:01",
            "messages": [{"role": "user", "content": "主机过热", "timestamp": "t"}],
            "case_state": {"status": "in_progress"},
        }
        store.upsert(conv)
        got = store.get("c1")
        assert got == conv
        assert store.count() == 1
        assert store.total_messages() == 1

    def test_update_via_upsert(self, store):
        store.upsert({"id": "c1", "title": "v1", "messages": [], "case_state": None})
        store.upsert({"id": "c1", "title": "v2", "messages": [{"role": "user", "content": "x", "timestamp": "t"}], "case_state": {"a": 1}})
        got = store.get("c1")
        assert got["title"] == "v2"
        assert got["case_state"] == {"a": 1}

    def test_delete(self, store):
        store.upsert({"id": "c1", "title": "t", "messages": [], "case_state": None})
        assert store.delete("c1") is True
        assert store.delete("c1") is False
        assert store.get("c1") is None

    def test_thread_safety(self, store):
        """并发写不丢数据（issue #4：原 JSON 文件无锁会互相覆盖）"""
        n_threads, n_writes = 8, 20
        errors = []

        def worker(tid):
            try:
                for i in range(n_writes):
                    store.upsert({
                        "id": f"t{tid}-{i}",
                        "title": f"conv-{tid}-{i}",
                        "messages": [],
                        "case_state": None,
                    })
            except Exception as e:  # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert store.count() == n_threads * n_writes

    def test_legacy_json_migration(self, store, tmp_path):
        legacy = tmp_path / "conversations"
        legacy.mkdir()
        old = {"id": "old1", "title": "旧对话", "messages": [{"role": "user", "content": "x", "timestamp": "t"}], "case_state": None}
        (legacy / "old1.json").write_text(json.dumps(old, ensure_ascii=False), encoding="utf-8")

        migrated = store.migrate_legacy_json(str(legacy))
        assert migrated == 1
        assert store.get("old1")["title"] == "旧对话"
        # 幂等：再跑一次不重复
        assert store.migrate_legacy_json(str(legacy)) == 0


# ---------- case_state 兼容逻辑 ----------

class TestCaseStateCompat:
    def test_default_and_ensure(self, tmp_path, monkeypatch):
        """ensure_conversation_case_state 对旧数据补齐结构"""
        # 使用独立实例避免污染模块级单例
        from services.conversation_store import ConversationStore
        from services.conversation_service import ConversationService
        from services import case_state_service as css

        svc = ConversationService(str(tmp_path / "case.db"))
        # 替换模块级引用
        monkeypatch.setattr(css.conversation_service, "_cache", svc.store and {})
        # 直接构造干净服务替换
        monkeypatch.setattr(css, "conversation_service", svc)

        svc.create("t", case_state=None)
        conv_id = svc.list_all()[0]["id"]

        # case_state 为 None → ensure 后补齐默认结构
        css.ensure_conversation_case_state(conv_id)
        state = svc.get(conv_id)["case_state"]
        assert state["status"] == "in_progress"
        assert isinstance(state["fault_context"], dict)
        assert state["fault_context"]["confirmed"] is False

        # 旧结构（缺新字段）也能补齐
        conv = svc.get(conv_id)
        conv["case_state"] = {"status": "in_progress"}  # 极简旧数据
        svc.save_conv(conv)
        css.ensure_conversation_case_state(conv_id)
        state = svc.get(conv_id)["case_state"]
        assert "draft" in state and "todo" in state
        svc.close()
