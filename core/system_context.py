"""
全局系统上下文模块

持有 RAG 系统单例，供 routers/services 层在请求期访问，
避免各模块循环依赖 app.py。
"""

from __future__ import annotations

from typing import Optional

# 全局 RAG 系统实例（由 app.py 启动事件注入）
_rag_system = None


def set_rag_system(system) -> None:
    """注入全局 RAG 系统实例（应用启动时调用一次）"""
    global _rag_system
    _rag_system = system


def get_rag_system():
    """获取全局 RAG 系统实例

    Raises:
        RuntimeError: 系统尚未初始化（应用启动事件未完成）时抛出
    """
    if _rag_system is None:
        raise RuntimeError("RAG 系统尚未初始化，请等待应用启动完成")
    return _rag_system


def is_system_ready() -> bool:
    """系统是否已初始化"""
    return _rag_system is not None
