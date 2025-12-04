"""
utils - Worker 工具模块

包含：
  - auth: 认证工具
  - path_utils: 路径安全工具
  - stream_buffer: 流式缓冲
  - vram_optimizer: 显存优化
  - model_manager: 模型管理器（检测/下载/验证/加载）
"""

from .auth import (
    verify_token,
    verify_grpc_context,
    verify_grpc_context_async,
)
from .path_utils import (
    PathTraversalError,
    safe_join,
    validate_model_path,
    is_safe_filename,
    sanitize_filename,
)

# 延迟导入，避免循环依赖
def get_model_manager():
    """获取 ModelManager 类（延迟导入）"""
    from .model_manager import ModelManager
    return ModelManager

def prepare_model(*args, **kwargs):
    """便捷函数：准备模型（延迟导入）"""
    from .model_manager import prepare_model as _prepare_model
    return _prepare_model(*args, **kwargs)

__all__ = [
    # auth
    "verify_token",
    "verify_grpc_context",
    "verify_grpc_context_async",
    # path_utils
    "PathTraversalError",
    "safe_join",
    "validate_model_path",
    "is_safe_filename",
    "sanitize_filename",
    # model_manager
    "get_model_manager",
    "prepare_model",
]
