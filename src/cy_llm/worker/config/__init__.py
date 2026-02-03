"""
config - Worker 配置模块

提供配置加载、验证和管理功能。

模块结构:
    config/
    ├── config_loader.py   # 原有配置加载器（向后兼容）
    ├── models.py          # Pydantic 配置模型（新）
    └── validator.py       # 配置验证器（新）

使用示例:
    # 方式一：使用原有 API（向后兼容）
    from worker.config import load_worker_config, WorkerConfig
    config = load_worker_config()
    
    # 方式二：使用 Pydantic 模型（推荐）
    from worker.config import AppConfig, WorkerSettings
    settings = WorkerSettings()
    config = AppConfig(worker=WorkerConfig(...))
"""

# 原有 API（向后兼容）
from .config_loader import (
    # 数据类
    HardwareProfile,
    ModelSpec,
    WorkerConfig,
    
    # 函数
    detect_hardware,
    determine_backend,
    load_model_registry,
    load_worker_config,
    print_config_help,
    validate_engine_type,
    
    # 常量
    SUPPORTED_ENGINES,
    ENV_PREFERRED_BACKEND,
    ENV_ENGINE_TYPE,
    ENV_MODEL_REGISTRY,
    ENV_MODEL_REGISTRY_PATH,
    ENV_DEFAULT_MODEL,
    ENV_DEFAULT_ADAPTER,
)

# 新 Pydantic API
from .models import (
    # 状态标记
    PYDANTIC_AVAILABLE,
    
    # 枚举
    EngineType,
    DeviceType,
    
    # 配置模型
    HardwareProfile as HardwareProfileModel,
    ModelSpec as ModelSpecModel,
    WorkerConfig as WorkerConfigModel,
    TrainingConfig,
    ServerConfig,
    AppConfig,
    WorkerSettings,
    
    # 辅助函数
    get_supported_engines,
    is_valid_engine,
    get_default_engine_for_device,
)

from .validator import (
    ConfigValidator,
    ConfigValidationError,
    validate_config,
    suggest_engine,
)

__all__ = [
    # 原有 API
    "HardwareProfile",
    "ModelSpec",
    "WorkerConfig",
    "detect_hardware",
    "determine_backend",
    "load_model_registry",
    "load_worker_config",
    "print_config_help",
    "validate_engine_type",
    "SUPPORTED_ENGINES",
    
    # 环境变量
    "ENV_PREFERRED_BACKEND",
    "ENV_ENGINE_TYPE",
    "ENV_MODEL_REGISTRY",
    "ENV_MODEL_REGISTRY_PATH",
    "ENV_DEFAULT_MODEL",
    "ENV_DEFAULT_ADAPTER",
    
    # 状态标记
    "PYDANTIC_AVAILABLE",
    
    # Pydantic 模型
    "EngineType",
    "DeviceType",
    "HardwareProfileModel",
    "ModelSpecModel",
    "WorkerConfigModel",
    "TrainingConfig",
    "ServerConfig",
    "AppConfig",
    "WorkerSettings",
    
    # 验证
    "ConfigValidator",
    "ConfigValidationError",
    "validate_config",
    "suggest_engine",
    
    # 辅助
    "get_supported_engines",
    "is_valid_engine",
    "get_default_engine_for_device",
]