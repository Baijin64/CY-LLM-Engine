"""
training.model - 模型加载和配置模块
"""

from .setup import (
    ModelSetup,
    ModelConfig,
    DeviceConfig,
    DeviceType,
    QuantizationConfig,
)
from .lora import (
    LoRASetup,
    LoRAConfig,
    LoRAConfigBuilder,
    TaskType,
    TARGET_MODULES_PRESETS,
)

__all__ = [
    # 模型加载
    "ModelSetup",
    "ModelConfig",
    "DeviceConfig",
    "DeviceType",
    "QuantizationConfig",
    
    # LoRA
    "LoRASetup",
    "LoRAConfig",
    "LoRAConfigBuilder",
    "TaskType",
    "TARGET_MODULES_PRESETS",
]
