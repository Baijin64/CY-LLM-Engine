"""
exceptions.py
[异常层次] 定义 Worker 模块的自定义异常

异常层次结构：
    EWWorkerError (基类)
    ├── EngineError (引擎相关)
    │   ├── EngineNotFoundError
    │   ├── EngineLoadError
    │   ├── EngineInferenceError
    │   └── EngineTimeoutError
    ├── ModelError (模型相关)
    │   ├── ModelNotFoundError
    │   ├── ModelLoadError
    │   └── ModelUnloadError
    ├── ConfigError (配置相关)
    │   ├── ConfigNotFoundError
    │   ├── ConfigValidationError
    │   └── ConfigReloadError
    ├── TrainingError (训练相关)
    │   ├── TrainingConfigError
    │   ├── TrainingDataError
    │   ├── TrainingInterruptedError
    │   └── TrainingCheckpointError
    ├── ResourceError (资源相关)
    │   ├── GPUMemoryError
    │   ├── ResourceExhaustedError
    │   └── QuotaExceededError
    └── CommunicationError (通信相关)
        ├── GRPCError
        └── CacheError
"""

from __future__ import annotations

from typing import Any, Dict, Optional


# ============================================================================
# 基类
# ============================================================================

class CYLLMWorkerError(Exception):
    """
    CY-LLM Worker 异常基类。

    所有 Worker 模块的异常都应继承此类。为向后兼容，保留 `EWWorkerError`/`EWWorkerException` 别名。
    """
    
    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        """
        初始化异常。
        
        Args:
            message: 错误消息
            code: 错误代码（用于程序化处理）
            details: 额外的错误详情
            cause: 原始异常（用于异常链）
        """
        super().__init__(message)
        self.message = message
        self.code = code or self.__class__.__name__
        self.details = details or {}
        self.cause = cause
    
    def __str__(self) -> str:
        parts = [f"[{self.code}] {self.message}"]
        if self.details:
            parts.append(f"详情: {self.details}")
        if self.cause:
            parts.append(f"原因: {self.cause}")
        return " | ".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于 JSON 序列化）"""
        return {
            "error": self.code,
            "message": self.message,
            "details": self.details,
        }


# 兼容别名: 保持对 EWWorkerError 的向后兼容
EWWorkerException = CYLLMWorkerError
EWWorkerError = CYLLMWorkerError


# ============================================================================
# 引擎相关异常
# ============================================================================

class EngineError(CYLLMWorkerError):
    """引擎相关异常基类"""
    pass


class EngineNotFoundError(EngineError):
    """引擎未找到"""
    
    def __init__(self, engine_type: str, available: Optional[list] = None):
        details = {"engine_type": engine_type}
        if available:
            details["available_engines"] = available
        super().__init__(
            f"未知的引擎类型: '{engine_type}'",
            code="ENGINE_NOT_FOUND",
            details=details,
        )


class EngineLoadError(EngineError):
    """引擎加载失败"""
    
    def __init__(self, engine_type: str, reason: str, cause: Optional[Exception] = None):
        super().__init__(
            f"引擎 '{engine_type}' 加载失败: {reason}",
            code="ENGINE_LOAD_ERROR",
            details={"engine_type": engine_type, "reason": reason},
            cause=cause,
        )


class EngineInferenceError(EngineError):
    """推理执行失败"""
    
    def __init__(self, engine_type: str, reason: str, cause: Optional[Exception] = None):
        super().__init__(
            f"推理失败 ({engine_type}): {reason}",
            code="ENGINE_INFERENCE_ERROR",
            details={"engine_type": engine_type, "reason": reason},
            cause=cause,
        )


class EngineTimeoutError(EngineError):
    """引擎操作超时"""
    
    def __init__(self, operation: str, timeout_seconds: float):
        super().__init__(
            f"操作 '{operation}' 超时 ({timeout_seconds}s)",
            code="ENGINE_TIMEOUT",
            details={"operation": operation, "timeout_seconds": timeout_seconds},
        )


# ============================================================================
# 模型相关异常
# ============================================================================

class ModelError(CYLLMWorkerError):
    """模型相关异常基类"""
    pass


class ModelNotFoundError(ModelError):
    """模型未找到"""
    
    def __init__(self, model_id: str, registry_path: Optional[str] = None):
        details = {"model_id": model_id}
        if registry_path:
            details["registry_path"] = registry_path
        super().__init__(
            f"模型 '{model_id}' 未在注册表中找到",
            code="MODEL_NOT_FOUND",
            details=details,
        )
        self.model_id = model_id  # 兼容旧测试


class ModelLoadError(ModelError):
    """模型加载失败"""
    
    def __init__(self, message: str = "", model_id: Optional[str] = None, model_path: Optional[str] = None, reason: Optional[str] = None, cause: Optional[Exception] = None, **kwargs):
        # 支持两种调用方式：
        # 1. ModelLoadError("message", model_path="/path")
        # 2. ModelLoadError(model_id="x", model_path="/path", reason="xxx")
        if reason:
            msg = f"模型 '{model_id}' 加载失败: {reason}"
        else:
            msg = message or "模型加载失败"
        super().__init__(
            msg,
            code="MODEL_LOAD_ERROR",
            details={"model_id": model_id, "model_path": model_path, "reason": reason, **kwargs},
            cause=cause,
        )
        self.model_path = model_path


class ModelUnloadError(ModelError):
    """模型卸载失败"""
    
    def __init__(self, model_id: str, reason: str, cause: Optional[Exception] = None):
        super().__init__(
            f"模型 '{model_id}' 卸载失败: {reason}",
            code="MODEL_UNLOAD_ERROR",
            details={"model_id": model_id, "reason": reason},
            cause=cause,
        )


# ============================================================================
# 配置相关异常
# ============================================================================

class ConfigError(CYLLMWorkerError):
    """配置相关异常基类"""
    pass


class ConfigNotFoundError(ConfigError):
    """配置文件未找到"""
    
    def __init__(self, config_path: str):
        super().__init__(
            f"配置文件未找到: {config_path}",
            code="CONFIG_NOT_FOUND",
            details={"config_path": config_path},
        )


class ConfigValidationError(ConfigError):
    """配置验证失败"""
    
    def __init__(self, errors: list, config_path: Optional[str] = None):
        details = {"errors": errors}
        if config_path:
            details["config_path"] = config_path
        super().__init__(
            f"配置验证失败: {len(errors)} 个错误",
            code="CONFIG_VALIDATION_ERROR",
            details=details,
        )


class ConfigReloadError(ConfigError):
    """配置热重载失败"""
    
    def __init__(self, reason: str, cause: Optional[Exception] = None):
        super().__init__(
            f"配置重载失败: {reason}",
            code="CONFIG_RELOAD_ERROR",
            details={"reason": reason},
            cause=cause,
        )


# ============================================================================
# 训练相关异常
# ============================================================================

class TrainingError(CYLLMWorkerError):
    """训练相关异常基类"""
    
    def __init__(self, message: str = "", epoch: Optional[int] = None, **kwargs):
        super().__init__(message, code="TRAINING_ERROR", details={"epoch": epoch, **kwargs})
        self.epoch = epoch


class TrainingConfigError(TrainingError):
    """训练配置错误"""
    
    def __init__(self, reason: str, field: Optional[str] = None):
        details = {"reason": reason}
        if field:
            details["field"] = field
        super().__init__(
            f"训练配置错误: {reason}",
            code="TRAINING_CONFIG_ERROR",
            details=details,
        )


class TrainingDataError(TrainingError):
    """训练数据错误"""
    
    def __init__(self, reason: str, data_path: Optional[str] = None):
        details = {"reason": reason}
        if data_path:
            details["data_path"] = data_path
        super().__init__(
            f"训练数据错误: {reason}",
            code="TRAINING_DATA_ERROR",
            details=details,
        )


class TrainingInterruptedError(TrainingError):
    """训练被中断"""
    
    def __init__(self, job_id: str, reason: str, checkpoint: Optional[str] = None):
        details = {"job_id": job_id, "reason": reason}
        if checkpoint:
            details["last_checkpoint"] = checkpoint
        super().__init__(
            f"训练任务 '{job_id}' 被中断: {reason}",
            code="TRAINING_INTERRUPTED",
            details=details,
        )


class TrainingCheckpointError(TrainingError):
    """检查点保存/加载失败"""
    
    def __init__(self, checkpoint_path: str, operation: str, reason: str, cause: Optional[Exception] = None):
        super().__init__(
            f"检查点 {operation} 失败: {reason}",
            code="TRAINING_CHECKPOINT_ERROR",
            details={"checkpoint_path": checkpoint_path, "operation": operation, "reason": reason},
            cause=cause,
        )


# ============================================================================
# 资源相关异常
# ============================================================================

class ResourceError(CYLLMWorkerError):
    """资源相关异常基类"""
    pass


class GPUMemoryError(ResourceError):
    """GPU 显存不足"""
    
    def __init__(
        self,
        required_mb: float,
        available_mb: float,
        device_id: int = 0,
    ):
        super().__init__(
            f"GPU 显存不足: 需要 {required_mb:.0f}MB，可用 {available_mb:.0f}MB",
            code="GPU_MEMORY_ERROR",
            details={
                "required_mb": required_mb,
                "available_mb": available_mb,
                "device_id": device_id,
            },
        )


class ResourceExhaustedError(ResourceError):
    """资源耗尽"""
    
    def __init__(self, message: str = "", resource_type: Optional[str] = None, reason: Optional[str] = None, **kwargs):
        # 支持两种调用方式
        if resource_type and reason:
            msg = f"资源耗尽 ({resource_type}): {reason}"
        elif resource_type:
            msg = f"资源耗尽 ({resource_type}): {message}"
        else:
            msg = message or "资源耗尽"
        super().__init__(
            msg,
            code="RESOURCE_EXHAUSTED",
            details={"resource_type": resource_type, "reason": reason, **kwargs},
        )
        self.resource_type = resource_type


class QuotaExceededError(ResourceError):
    """配额超限"""
    
    def __init__(self, quota_type: str, limit: int, current: int):
        super().__init__(
            f"配额超限 ({quota_type}): 当前 {current}，限制 {limit}",
            code="QUOTA_EXCEEDED",
            details={"quota_type": quota_type, "limit": limit, "current": current},
        )


# ============================================================================
# 通信相关异常
# ============================================================================

class CommunicationError(CYLLMWorkerError):
    """通信相关异常基类"""
    pass


class GRPCError(CommunicationError):
    """gRPC 通信错误"""
    
    def __init__(self, status_code: str, message: str, cause: Optional[Exception] = None):
        super().__init__(
            f"gRPC 错误 [{status_code}]: {message}",
            code="GRPC_ERROR",
            details={"grpc_status": status_code, "message": message},
            cause=cause,
        )


class CacheError(CommunicationError):
    """缓存操作错误"""
    
    def __init__(self, operation: str, reason: str, cause: Optional[Exception] = None):
        super().__init__(
            f"缓存 {operation} 失败: {reason}",
            code="CACHE_ERROR",
            details={"operation": operation, "reason": reason},
            cause=cause,
        )


# ============================================================================
# 异常处理工具
# ============================================================================

def wrap_exception(exc: Exception, context: str = "") -> EWWorkerError:
    """
    将标准异常包装为 EWWorkerError。
    
    Args:
        exc: 原始异常
        context: 上下文描述
        
    Returns:
        包装后的 EWWorkerError
    """
    if isinstance(exc, EWWorkerError):
        return exc
    
    message = f"{context}: {exc}" if context else str(exc)
    return EWWorkerError(
        message=message,
        code="WRAPPED_ERROR",
        details={"original_type": type(exc).__name__},
        cause=exc,
    )


# 兼容别名（放在文件末尾，避免引用未定义的类）

# InferenceError 兼容类
class InferenceError(EngineError):
    """推理错误（兼容旧 API）"""
    def __init__(self, message: str = "", model_id: Optional[str] = None, prompt_length: Optional[int] = None, **kwargs):
        super().__init__(message, code="INFERENCE_ERROR", details={"model_id": model_id, "prompt_length": prompt_length, **kwargs})
        self.model_id = model_id
        self.prompt_length = prompt_length

# ConfigurationError 兼容类  
class ConfigurationError(ConfigError):
    """配置错误（兼容旧 API）"""
    def __init__(self, message: str = "", key: Optional[str] = None, **kwargs):
        super().__init__(message, code="CONFIG_ERROR", details={"key": key, **kwargs})
        self.key = key

# 认证错误：覆盖旧的名称并继承自 CommunicationError
class AuthenticationError(CommunicationError):
    def __init__(self, message: str, *, code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code=code or "AUTH_ERROR", details=details)

# 兼容别名（重复声明避免冲突）
EWWorkerException = EWWorkerError


__all__ = [
    # 基类
    "EWWorkerError",
    
    # 引擎异常
    "EngineError",
    "EngineNotFoundError",
    "EngineLoadError",
    "EngineInferenceError",
    "EngineTimeoutError",
    
    # 模型异常
    "ModelError",
    "ModelNotFoundError",
    "ModelLoadError",
    "ModelUnloadError",
    
    # 配置异常
    "ConfigError",
    "ConfigNotFoundError",
    "ConfigValidationError",
    "ConfigReloadError",
    
    # 训练异常
    "TrainingError",
    "TrainingConfigError",
    "TrainingDataError",
    "TrainingInterruptedError",
    "TrainingCheckpointError",
    
    # 资源异常
    "ResourceError",
    "GPUMemoryError",
    "ResourceExhaustedError",
    "QuotaExceededError",
    
    # 通信异常
    "CommunicationError",
    "GRPCError",
    "CacheError",
    
    # 工具函数
    "wrap_exception",
]
