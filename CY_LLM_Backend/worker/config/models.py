"""
config.models - Pydantic 配置模型
使用 Pydantic v2 定义配置数据模型
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
    from pydantic_settings import BaseSettings, SettingsConfigDict
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    # 提供 fallback - 在 pydantic 不可用时使用简单的 dataclass-like 基类
    from dataclasses import dataclass, field as dataclass_field
    
    class _FallbackBase:
        """Pydantic 不可用时的 fallback 基类"""
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                if hasattr(self.__class__, key):
                    setattr(self, key, value)
        
        def model_dump(self) -> dict:
            """兼容 Pydantic model_dump"""
            return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    BaseModel = _FallbackBase
    BaseSettings = _FallbackBase
    
    def Field(default=None, *, description: str = "", **kwargs):
        """Pydantic Field 的 fallback"""
        return default
    
    def field_validator(*args, **kwargs):
        """Pydantic field_validator 的 fallback"""
        def decorator(func):
            return func
        return decorator
    
    def model_validator(*args, **kwargs):
        """Pydantic model_validator 的 fallback"""
        def decorator(func):
            return func
        return decorator
    
    class SettingsConfigDict(dict):
        """Pydantic SettingsConfigDict 的 fallback"""
        pass
    
    # Provide a lightweight fallback for ConfigDict used in Pydantic v2
    class ConfigDict(dict):
        """Fallback for pydantic.ConfigDict used when Pydantic is unavailable."""
        pass

LOGGER = logging.getLogger("cy_llm.worker.config.models")


class EngineType(str, Enum):
    """支持的推理引擎类型"""
    # NVIDIA CUDA
    CUDA_VLLM = "cuda-vllm"
    CUDA_TRT = "cuda-trt"
    CUDA_VLLM_ASYNC = "cuda-vllm-async"
    
    # 华为 Ascend
    ASCEND_VLLM = "ascend-vllm"
    ASCEND_MINDIE = "ascend-mindie"
    
    # 兼容别名
    NVIDIA = "nvidia"
    CUDA = "cuda"
    ASCEND = "ascend"
    NPU = "npu"
    HYBRID = "hybrid"


class DeviceType(str, Enum):
    """设备类型"""
    CUDA = "cuda"
    ASCEND = "ascend"
    CPU = "cpu"


class HardwareProfile(BaseModel):
    """硬件配置"""
    has_cuda: bool = False
    cuda_device_count: int = 0
    cuda_device_name: Optional[str] = None
    cuda_memory_gb: float = 0.0
    
    has_ascend: bool = False
    ascend_device_count: int = 0
    
    bf16_supported: bool = False
    
    @property
    def primary_device(self) -> DeviceType:
        """获取主设备类型"""
        if self.has_cuda:
            return DeviceType.CUDA
        elif self.has_ascend:
            return DeviceType.ASCEND
        return DeviceType.CPU

    # Use Pydantic v2 style 'model_config' to avoid deprecation warnings
    model_config = ConfigDict(frozen=True)


class ModelSpec(BaseModel):
    """模型规格配置"""
    model_path: str = Field(..., description="模型路径或 HuggingFace ID")
    adapter_path: Optional[str] = Field(None, description="LoRA 适配器路径")
    engine: Optional[EngineType] = Field(None, description="指定引擎类型")

    # 量化配置（新增标准字段）
    quantization: Optional[str] = Field(
        None,
        description="量化方法: awq, gptq, bitsandbytes, fp8"
    )
    use_4bit: Optional[bool] = Field(
        None,
        description="[已废弃] 使用 quantization='bitsandbytes' 替代"
    )

    # vLLM 配置
    max_model_len: Optional[int] = Field(None, description="最大模型长度")
    tensor_parallel_size: Optional[int] = Field(None, ge=1, description="张量并行度")
    gpu_memory_utilization: Optional[float] = Field(None, ge=0.0, le=1.0, description="GPU 显存利用率")

    # KV Cache 配置
    enable_prefix_caching: Optional[bool] = Field(None, description="启用前缀缓存")
    kv_cache_dtype: Optional[str] = Field(None, description="KV Cache 数据类型")

    # Prompt 缓存
    enable_prompt_cache: Optional[bool] = Field(None, description="启用 Prompt 缓存")
    prompt_cache_ttl: Optional[int] = Field(None, ge=0, description="Prompt 缓存 TTL (秒)")

    @field_validator("quantization", mode="before")
    @classmethod
    def validate_quantization(cls, v):
        """验证量化方法"""
        if v is None:
            return None
        v_lower = str(v).lower().strip()
        valid_methods = ["awq", "gptq", "bitsandbytes", "fp8", "fp8_e5m2", "none"]
        if v_lower not in valid_methods:
            raise ValueError(
                f"不支持的量化方法: {v}. 支持: {', '.join(valid_methods)}"
            )
        return None if v_lower == "none" else v_lower

    @field_validator("gpu_memory_utilization", mode="before")
    @classmethod
    def validate_gpu_memory(cls, v):
        """验证并警告不安全的显存配置"""
        if v is not None and v > 0.90:
            LOGGER.warning(
                "gpu_memory_utilization=%.2f 过高，可能导致 OOM。推荐值: 0.70-0.85",
                v
            )
        return v

    # Use Pydantic v2 style 'model_config' to avoid deprecation warnings
    model_config = ConfigDict(frozen=True, use_enum_values=True)


class WorkerConfig(BaseModel):
    """Worker 配置"""
    preferred_backend: EngineType = Field(
        default=EngineType.CUDA_VLLM,
        description="默认推理引擎"
    )
    hardware: HardwareProfile = Field(default_factory=HardwareProfile)
    model_registry: Dict[str, ModelSpec] = Field(
        default_factory=dict,
        description="模型注册表"
    )
    
    # 服务配置
    grpc_port: int = Field(default=50051, ge=1, le=65535)
    max_workers: int = Field(default=10, ge=1)
    queue_size: int = Field(default=128, ge=1)
    
    # 缓存配置
    prompt_cache_enabled: bool = Field(default=True)
    prompt_cache_max_size: int = Field(default=1000, ge=0)
    prompt_cache_ttl: int = Field(default=3600, ge=0)

    def get_engine_type(self) -> str:
        """获取引擎类型字符串"""
        if isinstance(self.preferred_backend, EngineType):
            return self.preferred_backend.value
        return str(self.preferred_backend)

    @field_validator("preferred_backend", mode="before")
    @classmethod
    def validate_backend(cls, v):
        if isinstance(v, str):
            v = v.lower().strip()
            try:
                return EngineType(v)
            except ValueError:
                # 尝试匹配
                for engine in EngineType:
                    if engine.value == v:
                        return engine
                raise ValueError(f"不支持的引擎类型: {v}")
        return v


class TrainingConfig(BaseModel):
    """训练配置"""
    # 基础
    output_dir: str = Field(default="./output", description="输出目录")
    checkpoint_dir: str = Field(default="./checkpoints", description="检查点目录")
    
    # 并发
    max_concurrent_jobs: int = Field(default=1, ge=1)
    job_retention_days: int = Field(default=7, ge=1)
    
    # LoRA 默认值
    default_lora_r: int = Field(default=64, ge=1)
    default_lora_alpha: int = Field(default=16, ge=1)
    default_lora_dropout: float = Field(default=0.05, ge=0.0, le=1.0)
    
    # 训练默认值
    default_epochs: int = Field(default=3, ge=1)
    default_batch_size: int = Field(default=2, ge=1)
    default_learning_rate: float = Field(default=2e-4, gt=0)
    default_max_seq_length: int = Field(default=2048, ge=1)


class ServerConfig(BaseModel):
    """服务器配置"""
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=50051, ge=1, le=65535)
    
    # gRPC
    grpc_max_workers: int = Field(default=10, ge=1)
    grpc_max_message_size: int = Field(default=100 * 1024 * 1024)  # 100MB
    grpc_keepalive_time_ms: int = Field(default=30000)
    grpc_keepalive_timeout_ms: int = Field(default=10000)
    
    # 安全
    enable_auth: bool = Field(default=True)
    internal_token: Optional[str] = Field(default=None)


class AppConfig(BaseModel):
    """应用总配置"""
    worker: WorkerConfig = Field(default_factory=WorkerConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    
    # 日志
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="[%(levelname)s] %(name)s: %(message)s")


# ============================================================================
# Settings (从环境变量加载)
# ============================================================================

if PYDANTIC_AVAILABLE:
    class WorkerSettings(BaseSettings):
        """Worker 环境变量配置"""
        
        # 引擎
        engine: Optional[str] = Field(default=None, alias="CY_LLM_ENGINE")
        backend: Optional[str] = Field(default=None, alias="CY_LLM_BACKEND")
        
        # 模型
        model_registry: Optional[str] = Field(default=None, alias="CY_LLM_MODEL_REGISTRY")
        model_registry_path: Optional[str] = Field(default=None, alias="CY_LLM_MODEL_REGISTRY_PATH")
        default_model: str = Field(
            default="deepseek-ai/deepseek-llm-7b-chat",
            alias="CY_LLM_DEFAULT_MODEL"
        )
        default_adapter: Optional[str] = Field(default=None, alias="CY_LLM_DEFAULT_ADAPTER")
        
        # 服务
        port: int = Field(default=50051, alias="CY_LLM_PORT")
        internal_token: Optional[str] = Field(default=None, alias="CY_LLM_INTERNAL_TOKEN")
        
        # 缓存
        prompt_cache_enabled: bool = Field(default=True, alias="CY_LLM_PROMPT_CACHE_ENABLED")
        prompt_cache_ttl: int = Field(default=3600, alias="CY_LLM_PROMPT_CACHE_TTL")
        
        model_config = SettingsConfigDict(
            env_prefix="CY_LLM_",
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )
        
        def get_engine_type(self) -> Optional[str]:
            """获取引擎类型"""
            return self.engine or self.backend
else:
    WorkerSettings = None


# ============================================================================
# 辅助函数
# ============================================================================

def get_supported_engines() -> List[str]:
    """获取支持的引擎列表"""
    return [e.value for e in EngineType]


def is_valid_engine(engine_type: str) -> bool:
    """检查引擎类型是否有效"""
    try:
        EngineType(engine_type.lower().strip())
        return True
    except ValueError:
        return False


def get_default_engine_for_device(device: DeviceType) -> EngineType:
    """根据设备类型获取默认引擎"""
    defaults = {
        DeviceType.CUDA: EngineType.CUDA_VLLM,
        DeviceType.ASCEND: EngineType.ASCEND_VLLM,
        DeviceType.CPU: EngineType.NVIDIA,
    }
    return defaults.get(device, EngineType.NVIDIA)
