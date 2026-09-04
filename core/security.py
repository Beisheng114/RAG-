"""
应用安全模块（统一管理凭证校验，修复双份 ADMIN_TOKEN 实现不一致问题）

约定：
- 管理接口凭证：环境变量 RAG_ADMIN_TOKEN。未设置时管理接口直接禁用（503），
  不回退任何弱默认值。
- 普通接口凭证：环境变量 RAG_API_KEY（可选）。设置后所有 /api/* 请求
  需携带 X-API-Key（或携带有效 X-Admin-Token）才可访问；未设置时维持原有
  本地部署的无鉴权行为。
- CORS 白名单：环境变量 RAG_CORS_ORIGINS（逗号分隔），默认仅本机来源。
"""

from __future__ import annotations

import hmac
import os
from typing import List, Optional

from fastapi import Header, HTTPException


def get_admin_token() -> Optional[str]:
    """读取管理口令；未配置时返回 None（管理接口应禁用）"""
    return os.getenv("RAG_ADMIN_TOKEN") or None


def is_admin_enabled() -> bool:
    return get_admin_token() is not None


def require_admin(
    x_admin_token: Optional[str] = Header(default=None, alias="X-Admin-Token"),
) -> bool:
    """管理接口依赖：未配置 RAG_ADMIN_TOKEN 时禁用接口，而非回退弱默认值"""
    expected = get_admin_token()
    if expected is None:
        raise HTTPException(
            status_code=503,
            detail="管理接口未启用：请设置环境变量 RAG_ADMIN_TOKEN 后重启服务",
        )
    if not x_admin_token or not hmac.compare_digest(str(x_admin_token), expected):
        raise HTTPException(status_code=401, detail="管理员认证失败")
    return True


def get_api_key() -> Optional[str]:
    """读取普通 API 访问密钥；未配置时返回 None（不启用接口鉴权）"""
    return os.getenv("RAG_API_KEY") or None


def check_api_access(headers) -> bool:
    """校验请求头是否携带有效的 X-API-Key 或 X-Admin-Token

    供 API key 中间件调用；未启用 API key 时始终放行。
    """
    expected_api_key = get_api_key()
    if expected_api_key is None:
        return True
    provided = headers.get("X-API-Key")
    if provided and hmac.compare_digest(str(provided), expected_api_key):
        return True
    # 携带有效管理口令的请求（管理页面）同样放行
    expected_admin = get_admin_token()
    provided_admin = headers.get("X-Admin-Token")
    if (
        expected_admin
        and provided_admin
        and hmac.compare_digest(str(provided_admin), expected_admin)
    ):
        return True
    return False


def get_cors_origins() -> List[str]:
    """CORS 白名单：RAG_CORS_ORIGINS（逗号分隔），默认仅本机来源

    注意：不要与 allow_credentials=True 一起使用 "*"（Starlette 会在该组合下
    回显任意 Origin，等价于全开）。
    """
    raw = os.getenv("RAG_CORS_ORIGINS", "")
    if raw.strip():
        return [o.strip() for o in raw.split(",") if o.strip()]
    return [
        "http://localhost:8002",
        "http://127.0.0.1:8002",
    ]


def mask_secret(value: str, left: int = 3, right: int = 2) -> str:
    """敏感值打码，用于配置预览"""
    v = str(value or "")
    if not v:
        return ""
    if len(v) <= left + right:
        return "*" * len(v)
    return f"{v[:left]}{'*' * (len(v) - left - right)}{v[-right:]}"


def is_sensitive_config_key(key: str) -> bool:
    lk = key.lower()
    sensitive_tokens = ["password", "secret", "token", "api_key"]
    return any(t in lk for t in sensitive_tokens)
