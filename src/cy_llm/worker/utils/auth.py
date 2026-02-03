"""
auth.py
[认证工具] 统一的内部 Token 验证逻辑

用于 Gateway -> Worker 之间的安全通信认证。
支持 gRPC 同步/异步上下文。
"""

from __future__ import annotations

import hmac
import logging
import os
from typing import Optional, Tuple
import uuid

LOGGER = logging.getLogger("cy_llm.worker.auth")

# 内部认证 Token（Gateway -> Worker）
# 建议在生产环境中通过环境变量设置强随机值
INTERNAL_TOKEN: str = os.getenv("CY_LLM_INTERNAL_TOKEN", "")
ENV_ALLOW_INSECURE = "CY_LLM_ALLOW_INSECURE_INTERNAL_TOKEN"
ENV_APP_ENV = "CY_LLM_ENV"

# Token 验证相关常量
AUTH_HEADER_PREFIX = "Bearer "
AUTH_METADATA_KEY = "authorization"


def verify_token(provided_token: Optional[str], expected_token: Optional[str] = None) -> Tuple[bool, str]:
    """
    使用常数时间比较验证 Token（防止时序攻击）。
    
    Args:
        provided_token: 请求中提供的 Token
        expected_token: 期望的 Token，默认使用环境变量配置
        
    Returns:
        (is_valid, error_message) 元组
    """
    expected = expected_token or INTERNAL_TOKEN
    
    # 未配置 Token 时允许开发环境放行
    if not expected:
        if is_insecure_internal_token_allowed():
            LOGGER.warning(
                "INTERNAL_TOKEN 未配置，开发环境放行认证验证。"
                "生产环境请务必设置 CY_LLM_INTERNAL_TOKEN 环境变量！"
            )
            return True, ""
        return False, "Internal token required"
    
    # 无提供 Token
    if not provided_token:
        return False, "Missing authorization token"
    
    # 格式化期望值
    expected_header = f"{AUTH_HEADER_PREFIX}{expected}"
    
    # 使用 hmac.compare_digest 进行常数时间比较，防止时序攻击
    try:
        is_valid = hmac.compare_digest(
            provided_token.encode('utf-8'),
            expected_header.encode('utf-8')
        )
    except (UnicodeDecodeError, AttributeError):
        return False, "Invalid token format"
    
    if not is_valid:
        LOGGER.warning("内部认证失败: %s", provided_token[:20] if provided_token else "(空)")
        return False, "Invalid internal token"
    
    return True, ""


def extract_token_from_metadata(metadata: dict, key: str = AUTH_METADATA_KEY) -> Optional[str]:
    """
    从 gRPC metadata 中提取认证 Token。
    
    Args:
        metadata: gRPC 调用的 metadata 字典
        
    Returns:
        提取的 Token 字符串，或 None
    """
    return metadata.get(key, None)


def get_internal_token() -> str:
    """
    获取内部通信 token。

    如果 `CY_LLM_INTERNAL_TOKEN` 已配置则返回该值，否则生成并缓存一个随机 token 以便测试和本地开发使用。
    """
    global INTERNAL_TOKEN
    if INTERNAL_TOKEN:
        return INTERNAL_TOKEN
    if is_insecure_internal_token_allowed():
        token = uuid.uuid4().hex
        INTERNAL_TOKEN = token
        return token
    raise RuntimeError("CY_LLM_INTERNAL_TOKEN is required in production")


def is_insecure_internal_token_allowed() -> bool:
    env_value = os.getenv(ENV_APP_ENV, "development").lower().strip()
    allow_flag = os.getenv(ENV_ALLOW_INSECURE, "false").lower().strip() in {"1", "true", "yes"}
    return env_value != "production" or allow_flag


def enforce_internal_token_policy() -> None:
    if INTERNAL_TOKEN:
        return
    if is_insecure_internal_token_allowed():
        LOGGER.warning(
            "CY_LLM_INTERNAL_TOKEN 未配置，允许开发模式启动。"
            "生产环境请设置 CY_LLM_INTERNAL_TOKEN 并禁用 CY_LLM_ALLOW_INSECURE_INTERNAL_TOKEN。"
        )
        return
    raise RuntimeError("CY_LLM_INTERNAL_TOKEN is required in production")


def verify_grpc_context(context) -> bool:
    """
    验证 gRPC 上下文中的内部 Token。
    
    支持同步 grpc.ServicerContext 和异步 grpc.aio.ServicerContext。
    
    Args:
        context: gRPC ServicerContext
        
    Returns:
        验证是否通过
    """
    import grpc
    
    try:
        # 获取 metadata
        invocation_metadata = context.invocation_metadata()
        if invocation_metadata is None:
            metadata = {}
        else:
            metadata = dict(invocation_metadata)
    except Exception as e:
        LOGGER.error("获取 gRPC metadata 失败: %s", e)
        metadata = {}
    
    provided_token = extract_token_from_metadata(metadata)
    is_valid, error_msg = verify_token(provided_token)
    
    if not is_valid:
        context.abort(grpc.StatusCode.UNAUTHENTICATED, error_msg)
        return False
    
    return True


async def verify_grpc_context_async(context) -> bool:
    """
    异步验证 gRPC 上下文中的内部 Token。
    
    Args:
        context: grpc.aio.ServicerContext
        
    Returns:
        验证是否通过
    """
    import grpc
    
    try:
        invocation_metadata = context.invocation_metadata()
        if invocation_metadata is None:
            metadata = {}
        else:
            metadata = dict(invocation_metadata)
    except Exception as e:
        LOGGER.error("获取 gRPC metadata 失败: %s", e)
        metadata = {}
    
    provided_token = extract_token_from_metadata(metadata)
    is_valid, error_msg = verify_token(provided_token)
    
    if not is_valid:
        await context.abort(grpc.StatusCode.UNAUTHENTICATED, error_msg)
        return False
    
    return True


# 便捷导出
__all__ = [
    "INTERNAL_TOKEN",
    "verify_token",
    "verify_grpc_context",
    "verify_grpc_context_async",
    "extract_token_from_metadata",
    "get_internal_token",
    "is_insecure_internal_token_allowed",
    "enforce_internal_token_policy",
]
