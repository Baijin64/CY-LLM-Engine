"""
path_utils.py
[路径工具] 安全的路径处理，防止路径遍历攻击

使用示例：
    >>> from worker.utils.path_utils import safe_join, validate_model_path
    >>> safe_path = safe_join("/models", user_input)
    >>> validate_model_path("/models/llama", allowed_roots=["/models", "/data"])
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Union

LOGGER = logging.getLogger("cy_llm.worker.utils.path")


class PathTraversalError(ValueError):
    """路径遍历攻击检测异常"""
    pass


def safe_join(base: Union[str, Path], *parts: str) -> Path:
    """
    安全地拼接路径，防止路径遍历攻击。
    
    确保最终路径始终在 base 目录下。
    
    Args:
        base: 基础路径
        *parts: 要拼接的路径部分
        
    Returns:
        安全的绝对路径
        
    Raises:
        PathTraversalError: 如果检测到路径遍历尝试
        
    Examples:
        >>> safe_join("/models", "llama", "weights.bin")
        PosixPath('/models/llama/weights.bin')
        
        >>> safe_join("/models", "../etc/passwd")
        PathTraversalError: Path traversal detected
    """
    base_path = Path(base).resolve()
    
    # 过滤并正规化 parts: 去除前导斜杠、空部分与 '.'
    sanitized_parts = []
    for p in parts:
        if not p:
            continue
        p = str(p).lstrip('/')
        if p == '.':
            continue
        sanitized_parts.append(p)
    # 拼接并解析路径
    joined = base_path.joinpath(*sanitized_parts).resolve()
    
    # 检查是否仍在 base 下
    try:
        joined.relative_to(base_path)
    except ValueError:
        raise PathTraversalError(
            f"Path traversal detected: '{'/'.join(parts)}' escapes base '{base}'"
        )
    
    # 返回字符串以兼容旧测试（也可以将其当作 Path 使用）
    return str(joined)


def validate_model_path(
    path: Union[str, Path],
    allowed_roots: Optional[List[Union[str, Path]]] = None,
    must_exist: bool = False,
) -> Path:
    """
    验证模型路径的安全性。
    
    Args:
        path: 要验证的路径
        allowed_roots: 允许的根目录列表，None 表示不检查
        must_exist: 是否要求路径必须存在
        
    Returns:
        验证后的绝对路径
        
    Raises:
        PathTraversalError: 如果路径不在允许的根目录下
        FileNotFoundError: 如果 must_exist=True 且路径不存在
    """
    resolved = Path(path).resolve()
    
    # 检查是否在允许的根目录下
    if allowed_roots:
        is_allowed = False
        for root in allowed_roots:
            root_path = Path(root).resolve()
            try:
                resolved.relative_to(root_path)
                is_allowed = True
                break
            except ValueError:
                continue
        
        if not is_allowed:
            raise PathTraversalError(
                f"Path '{path}' is not under any allowed root: {allowed_roots}"
            )
    
    # 检查是否存在
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")
    
    # 兼容旧测试：返回字符串路径
    return str(resolved)


def is_safe_filename(filename: str) -> bool:
    """
    检查文件名是否安全（不包含路径分隔符或特殊字符）。
    
    Args:
        filename: 文件名
        
    Returns:
        是否安全
    """
    # 禁止的字符
    dangerous_chars = {'/', '\\', '..', '\x00', '<', '>', ':', '"', '|', '?', '*'}
    
    for char in dangerous_chars:
        if char in filename:
            return False
    
    # 不允许空文件名
    if not filename or not filename.strip():
        return False
    
    # 不允许以 . 开头（隐藏文件）
    if filename.startswith('.'):
        return False
    
    return True


def sanitize_filename(filename: str, replacement: str = "_") -> str:
    """
    清理文件名，替换危险字符。
    
    Args:
        filename: 原始文件名
        replacement: 用于替换危险字符的字符
        
    Returns:
        安全的文件名
    """
    # 替换危险字符（包含常见 Windows/Unix 禁用字符）
    dangerous = ['/', '\\', '..', '\x00', '<', '>', ':', '"', '|', '?', '*']
    result = filename
    for d in dangerous:
        result = result.replace(d, replacement if d != '\x00' else '')
    
    # 去除前导点
    result = result.lstrip('.')
    
    # 如果结果为空，使用默认名称
    if not result or not result.strip():
        result = "unnamed"
    
    return result


def get_relative_path(path: Union[str, Path], base: Union[str, Path]) -> str:
    """
    获取相对于 base 的相对路径。
    
    Args:
        path: 目标路径
        base: 基础路径
        
    Returns:
        相对路径字符串
    """
    try:
        return str(Path(path).resolve().relative_to(Path(base).resolve()))
    except ValueError:
        # 不是子路径，返回绝对路径
        return str(Path(path).resolve())


# 便捷导出
__all__ = [
    "PathTraversalError",
    "safe_join",
    "validate_model_path",
    "is_safe_filename",
    "sanitize_filename",
    "get_relative_path",
    "is_safe_path",
]


def is_safe_path(path: Union[str, Path], base: Union[str, Path]) -> bool:
    """检查指定路径是否在 base 下并且是安全的路径（不会越权）。
    
    Args:
        path: 要检查的路径
        base: 基础路径
        
    Returns:
        如果 path 在 base 目录下，返回 True；否则返回 False
    """
    try:
        resolved_path = Path(path).resolve()
        resolved_base = Path(base).resolve()
        
        # 检查 resolved_path 是否在 resolved_base 目录下
        resolved_path.relative_to(resolved_base)
        return True
    except ValueError:
        return False
