"""
对话导出服务（从 app.py 下沉）

- export_json: 导出原始对话 JSON
- export_markdown: 导出人类可读的 Markdown（UTF-8-SIG 兼容 Windows Word）
"""
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict

logger = logging.getLogger(__name__)

EXPORT_DIR = "exports"


def _ensure_export_dir() -> None:
    os.makedirs(EXPORT_DIR, exist_ok=True)


def export_json(conversation: Dict[str, Any]) -> str:
    """导出对话为 JSON 文件，返回可访问的相对 URL"""
    _ensure_export_dir()
    conversation_id = conversation["id"]
    export_file = f"{EXPORT_DIR}/conversation_{conversation_id}.json"
    with open(export_file, "w", encoding="utf-8") as f:
        json.dump(conversation, f, ensure_ascii=False, indent=2)
    return f"/exports/{os.path.basename(export_file)}"


def export_markdown(conversation: Dict[str, Any]) -> str:
    """导出对话为 Markdown 文档，返回可访问的相对 URL"""
    _ensure_export_dir()
    conversation_id = conversation["id"]
    export_file = f"{EXPORT_DIR}/conversation_{conversation_id}.md"
    conversation_title = conversation.get("title") or f"conversation_{conversation_id}"

    lines = []
    lines.append(f"# {conversation_title}")
    lines.append("")
    lines.append(f"> 导出时间：{datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"> 对话 ID：{conversation_id}")
    lines.append("")
    lines.append("---")
    lines.append("")

    messages = conversation.get("messages", []) or []
    for i, message in enumerate(messages, start=1):
        role = message.get("role", "assistant")
        content = message.get("content", "") or ""
        timestamp = message.get("timestamp", "") or ""

        role_cn = "用户" if role == "user" else "助手"
        lines.append(f"## {i}. {role_cn}")
        if timestamp:
            lines.append(f"**时间**：{timestamp}")
        lines.append("")
        # content is already markdown-ish (the UI uses it as markdown), so we keep it as-is.
        lines.append(content)
        lines.append("")
        lines.append("---")
        lines.append("")

    # Use UTF-8-SIG for better Windows Word/Markdown app compatibility.
    with open(export_file, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(lines).rstrip() + "\n")
    return f"/exports/{os.path.basename(export_file)}"
