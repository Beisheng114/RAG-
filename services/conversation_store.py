"""
对话存储层（SQLite，替代内存 dict + 手写 JSON 文件）

修复的问题：
- 原实现无并发锁，多请求/多 worker 同时写 JSON 文件会互相覆盖丢数据
- 重启依赖目录里散落的 JSON 文件兜底，无事务保证

实现要点：
- sqlite3 标准库，零部署成本；WAL 模式提升并发读写能力
- threading.RLock 保护跨线程访问（FastAPI 线程池 + 检索线程池）
- messages / case_state 以 JSON 文本存储
- 启动时自动把旧版 conversations/*.json 迁移进库（一次性，幂等）
"""
import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL DEFAULT '',
    updated_at  TEXT NOT NULL DEFAULT '',
    messages    TEXT NOT NULL DEFAULT '[]',
    case_state  TEXT
)
"""


class ConversationStore:
    """SQLite 对话存储（线程安全）"""

    def __init__(self, db_path: str = "conversations.db"):
        self.db_path = db_path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.execute(_SCHEMA)
            self._conn.commit()
        logger.info(f"SQLite 对话存储就绪: {db_path}")

    # ---------- 内部工具 ----------

    @staticmethod
    def _row_to_conv(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "title": row["title"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "messages": json.loads(row["messages"] or "[]"),
            "case_state": json.loads(row["case_state"]) if row["case_state"] else None,
        }

    @staticmethod
    def _conv_to_row(conv: Dict[str, Any]):
        return (
            str(conv.get("id", "")),
            str(conv.get("title", "")),
            str(conv.get("created_at", "")),
            str(conv.get("updated_at", "")),
            json.dumps(conv.get("messages", []), ensure_ascii=False),
            json.dumps(conv.get("case_state"), ensure_ascii=False) if conv.get("case_state") is not None else None,
        )

    # ---------- CRUD ----------

    def upsert(self, conv: Dict[str, Any]) -> None:
        """插入或整体替换一条对话（与内存缓存同步后调用）"""
        with self._lock:
            self._conn.execute(
                """INSERT INTO conversations (id, title, created_at, updated_at, messages, case_state)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                     title=excluded.title,
                     created_at=excluded.created_at,
                     updated_at=excluded.updated_at,
                     messages=excluded.messages,
                     case_state=excluded.case_state""",
                self._conv_to_row(conv),
            )
            self._conn.commit()

    def get(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
        return self._row_to_conv(row) if row else None

    def list_all(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM conversations ORDER BY updated_at DESC"
            ).fetchall()
        return [self._row_to_conv(r) for r in rows]

    def delete(self, conversation_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM conversations WHERE id = ?", (conversation_id,)
            )
            self._conn.commit()
        return cur.rowcount > 0

    def count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]

    def total_messages(self) -> int:
        with self._lock:
            rows = self._conn.execute("SELECT messages FROM conversations").fetchall()
        return sum(len(json.loads(r["messages"] or "[]")) for r in rows)

    # ---------- 旧数据迁移 ----------

    def migrate_legacy_json(self, legacy_dir: str = "conversations") -> int:
        """把旧版 conversations/*.json 一次性导入 SQLite（已存在同 id 的跳过）"""
        legacy_path = Path(legacy_dir)
        if not legacy_path.exists():
            return 0
        migrated = 0
        for json_file in sorted(legacy_path.glob("*.json")):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    conv = json.load(f)
                conv_id = str(conv.get("id") or json_file.stem)
                conv["id"] = conv_id
                if self.get(conv_id) is None:
                    self.upsert(conv)
                    migrated += 1
            except Exception as e:
                logger.warning(f"迁移旧对话文件失败 {json_file}: {e}")
        if migrated:
            logger.info(f"已从 {legacy_dir}/ 迁移 {migrated} 条旧对话到 SQLite")
        return migrated

    def close(self) -> None:
        with self._lock:
            self._conn.close()
