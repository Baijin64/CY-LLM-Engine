"""
training/__init__.py
[训练模块] 导出训练相关组件

模块结构:
    training/
    ├── data/           # 数据加载和格式化
    │   ├── loader.py       # DatasetLoader
    │   └── formatter.py    # PromptFormatter
    ├── model/          # 模型加载和配置
    │   ├── setup.py        # ModelSetup, DeviceConfig
    │   └── lora.py         # LoRASetup, LoRAConfig
    ├── loop/           # 训练循环和回调
    │   ├── trainer.py      # TrainerFactory
    │   └── callbacks.py    # ProgressCallback
    ├── engine.py       # TrainingEngine 核心类
    └── custom_script_runner.py  # 自定义脚本执行器
"""

# 核心引擎
from .engine import (
    TrainingEngine,
    TrainingJob,
    TrainingRequest,
    get_training_engine,
)

# 数据模块
# from .data import (
#     DatasetLoader,
#     DatasetConfig,
#     PromptFormatter,
#     InstructionFormatter,
#     ChatMLFormatter,
#     FormatterConfig,
#     get_formatter,
# )

# 模型模块
from .model import (
    ModelSetup,
    ModelConfig,
    DeviceConfig,
    QuantizationConfig,
    LoRASetup,
    LoRAConfig,
    LoRAConfigBuilder,
)

# 训练循环模块
from .loop import (
    TrainerFactory,
    TrainerConfig,
    TrainerConfigBuilder,
    ProgressCallback,
    CheckpointCallback,
    TrainingProgress,
    JobStatus,
)

# 自定义脚本执行器
from .custom_script_runner import (
    CustomScriptRunner,
    ScriptJob,
    ScriptProgress,
    ScriptStatus,
    get_custom_script_runner,
)

__all__ = [
    # 核心
    "TrainingEngine",
    "TrainingJob",
    "TrainingRequest",
    "get_training_engine",
    
    # 数据
    "DatasetLoader",
    "DatasetConfig",
    "PromptFormatter",
    "InstructionFormatter",
    "ChatMLFormatter",
    "FormatterConfig",
    "get_formatter",
    
    # 模型
    "ModelSetup",
    "ModelConfig",
    "DeviceConfig",
    "QuantizationConfig",
    "LoRASetup",
    "LoRAConfig",
    "LoRAConfigBuilder",
    
    # 训练循环
    "TrainerFactory",
    "TrainerConfig",
    "TrainerConfigBuilder",
    "ProgressCallback",
    "CheckpointCallback",
    "TrainingProgress",
    "JobStatus",
    
    # 自定义脚本
    "CustomScriptRunner",
    "ScriptJob",
    "ScriptProgress",
    "ScriptStatus",
    "get_custom_script_runner",
]
