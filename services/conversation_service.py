"""
对话管理服务

- 内存缓存（读多写少）+ SQLite 持久化（写穿透），线程安全
- 替代原 app.py 中的全局 conversations dict + 手写 JSON 文件
- 启动时自动迁移旧版 conversations/*.json
"""
import logging
import os
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from services.conversation_store import ConversationStore

logger = logging.getLogger(__name__)


def generate_conversation_title(first_message: str) -> str:
    """根据第一条消息生成对话标题"""
    return first_message[:30] + "..." if len(first_message) > 30 else first_message


class ConversationService:
    """对话 CRUD（内存缓存 + SQLite 持久化）"""

    DEFAULT_DB_PATH = os.getenv("CONVERSATIONS_DB_PATH", "conversations.db")

    def __init__(self, db_path: Optional[str] = None):
        self.store = ConversationStore(db_path or self.DEFAULT_DB_PATH)
        self._lock = threading.RLock()
        # 内存缓存：id -> conversation dict
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._load_all()

    # ---------- 初始化 ----------

    def _load_all(self) -> None:
        """启动加载：先迁移旧 JSON，再全量载入缓存"""
        self.store.migrate_legacy_json()
        with self._lock:
            for conv in self.store.list_all():
                self._cache[conv["id"]] = conv
        logger.info(f"对话缓存加载完成，共 {len(self._cache)} 条")

    # ---------- CRUD ----------

    def create(self, title: str, case_state: Optional[Dict[str, Any]] = None,
               conversation_id: Optional[str] = None) -> Dict[str, Any]:
        conversation_id = conversation_id or str(uuid.uuid4())
        now = datetime.now().isoformat()
        conv = {
            "id": conversation_id,
            "title": title,
            "messages": [],
            "created_at": now,
            "updated_at": now,
            "case_state": case_state,
        }
        with self._lock:
            self._cache[conversation_id] = conv
            self.store.upsert(conv)
        return conv

    def get(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._cache.get(conversation_id)

    def list_all(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._cache.values())

    def exists(self, conversation_id: str) -> bool:
        with self._lock:
            return conversation_id in self._cache

    def save(self, conversation_id: str) -> None:
        """把内存中的对话写穿到 SQLite"""
        with self._lock:
            conv = self._cache.get(conversation_id)
            if conv is None:
                raise KeyError(f"对话不存在: {conversation_id}")
            self.store.upsert(conv)

    def save_conv(self, conv: Dict[str, Any]) -> None:
        """直接保存（并缓存）给定对话对象"""
        with self._lock:
            self._cache[conv["id"]] = conv
            self.store.upsert(conv)

    def delete(self, conversation_id: str) -> bool:
        with self._lock:
            existed = self._cache.pop(conversation_id, None) is not None
        self.store.delete(conversation_id)
        return existed

    # ---------- 统计 ----------

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "conversation_count": len(self._cache),
                "total_messages": sum(len(c.get("messages", [])) for c in self._cache.values()),
            }

    def close(self) -> None:
        self.store.close()


# ---------- 惰性单例 ----------
# 不在模块级直接实例化：import 本模块不再产生创建 SQLite/迁移旧 JSON 的副作用，
# 首次真正需要读写对话时才初始化（测试也因此无需 monkeypatch 模块属性）。

_conversation_service: Optional[ConversationService] = None
_singleton_lock = threading.Lock()


def get_conversation_service() -> ConversationService:
    """获取全局对话服务单例（首次调用时初始化，线程安全）"""
    global _conversation_service
    if _conversation_service is None:
        with _singleton_lock:
            if _conversation_service is None:
                _conversation_service = ConversationService()
    return _conversation_service


def reset_conversation_service() -> None:
    """重置并关闭单例（主要用于测试）"""
    global _conversation_service
    with _singleton_lock:
        if _conversation_service is not None:
            try:
                _conversation_service.close()
            except Exception:  # noqa: BLE001
                pass
        _conversation_service = None
