"""
training.model.lora - LoRA/PEFT 配置
负责配置和应用 LoRA 适配器
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

LOGGER = logging.getLogger("cy_llm.worker.training.model.lora")

# 延迟导入
_peft_imported = False
_LoraConfig = None
_get_peft_model = None
_prepare_model_for_kbit_training = None
_TaskType = None


def _ensure_peft_imported():
    """确保 PEFT 已导入"""
    global _peft_imported, _LoraConfig, _get_peft_model, _prepare_model_for_kbit_training, _TaskType
    if not _peft_imported:
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
            _LoraConfig = LoraConfig
            _get_peft_model = get_peft_model
            _prepare_model_for_kbit_training = prepare_model_for_kbit_training
            _TaskType = TaskType
            _peft_imported = True
            LOGGER.debug("PEFT 库加载成功")
        except ImportError as e:
            raise ImportError(
                "PEFT 未安装。请运行: pip install peft\n"
                f"原始错误: {e}"
            ) from e


class TaskType(Enum):
    """任务类型"""
    CAUSAL_LM = "CAUSAL_LM"
    SEQ_2_SEQ_LM = "SEQ_2_SEQ_LM"
    SEQ_CLS = "SEQ_CLS"
    TOKEN_CLS = "TOKEN_CLS"
    QUESTION_ANS = "QUESTION_ANS"
    FEATURE_EXTRACTION = "FEATURE_EXTRACTION"


# 常用模型的 target_modules 预设
TARGET_MODULES_PRESETS: Dict[str, List[str]] = {
    # LLaMA / Llama 2 / Llama 3
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    
    # Qwen / Qwen2
    "qwen": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    
    # DeepSeek
    "deepseek": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    
    # Mistral / Mixtral
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    
    # ChatGLM
    "chatglm": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    
    # Baichuan
    "baichuan": ["W_pack", "o_proj", "gate_proj", "up_proj", "down_proj"],
    
    # InternLM
    "internlm": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    
    # Yi
    "yi": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    
    # Bloom
    "bloom": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    
    # Falcon
    "falcon": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    
    # 默认（适用于大多数 LLaMA 架构模型）
    "default": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}


# Bias 类型定义
BiasType = Literal["none", "all", "lora_only"]


@dataclass
class LoRAConfig:
    """
    LoRA 配置
    
    参数说明:
    - r: LoRA 秩，越大表达能力越强但参数量越多（推荐 8-64）
    - lora_alpha: 缩放因子，通常设为 r 的 1-2 倍
    - target_modules: 要应用 LoRA 的模块名
    - lora_dropout: Dropout 率（训练时）
    - bias: 是否训练 bias ("none", "all", "lora_only")
    """
    r: int = 64
    lora_alpha: int = 16
    target_modules: Optional[List[str]] = None
    lora_dropout: float = 0.05
    bias: BiasType = "none"
    task_type: TaskType = TaskType.CAUSAL_LM
    
    # 高级选项
    modules_to_save: Optional[List[str]] = None  # 完整训练的模块
    fan_in_fan_out: bool = False
    init_lora_weights: bool = True

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = TARGET_MODULES_PRESETS["default"]


class LoRAConfigBuilder:
    """
    LoRA 配置构建器
    
    提供便捷的配置构建方法
    
    示例:
        >>> config = (LoRAConfigBuilder()
        ...     .for_model("qwen")
        ...     .with_rank(32)
        ...     .with_alpha(64)
        ...     .build())
    """

    def __init__(self):
        self._config = LoRAConfig()

    def for_model(self, model_type: str) -> "LoRAConfigBuilder":
        """根据模型类型设置 target_modules"""
        model_type = model_type.lower()
        
        # 尝试匹配预设
        for key in TARGET_MODULES_PRESETS:
            if key in model_type:
                self._config.target_modules = TARGET_MODULES_PRESETS[key]
                LOGGER.debug("使用 %s 预设的 target_modules", key)
                return self
        
        # 使用默认
        self._config.target_modules = TARGET_MODULES_PRESETS["default"]
        LOGGER.debug("使用默认 target_modules")
        return self

    def with_rank(self, r: int) -> "LoRAConfigBuilder":
        """设置 LoRA 秩"""
        self._config.r = r
        return self

    def with_alpha(self, alpha: int) -> "LoRAConfigBuilder":
        """设置 LoRA alpha"""
        self._config.lora_alpha = alpha
        return self

    def with_dropout(self, dropout: float) -> "LoRAConfigBuilder":
        """设置 dropout"""
        self._config.lora_dropout = dropout
        return self

    def with_target_modules(self, modules: List[str]) -> "LoRAConfigBuilder":
        """设置 target_modules"""
        self._config.target_modules = modules
        return self

    def with_bias(self, bias: BiasType) -> "LoRAConfigBuilder":
        """设置 bias 训练模式"""
        self._config.bias = bias
        return self

    def with_task_type(self, task_type: TaskType) -> "LoRAConfigBuilder":
        """设置任务类型"""
        self._config.task_type = task_type
        return self

    def build(self) -> LoRAConfig:
        """构建配置"""
        return self._config


class LoRASetup:
    """
    LoRA 设置器
    
    职责:
    1. 创建 PEFT LoraConfig
    2. 应用 LoRA 到模型
    3. 准备量化模型训练
    
    示例:
        >>> setup = LoRASetup(LoRAConfig(r=32))
        >>> model = setup.apply(model, is_quantized=True)
    """

    def __init__(self, config: LoRAConfig):
        _ensure_peft_imported()
        self.config = config

    def apply(self, model: Any, is_quantized: bool = False) -> Any:
        """
        应用 LoRA 到模型
        
        Args:
            model: 基础模型
            is_quantized: 是否为量化模型
            
        Returns:
            应用了 LoRA 的模型
        """
        LOGGER.info(
            "应用 LoRA: r=%d, alpha=%d, modules=%s",
            self.config.r,
            self.config.lora_alpha,
            self.config.target_modules,
        )
        
        # 量化模型需要特殊准备
        if is_quantized:
            LOGGER.debug("准备量化模型训练")
            model = _prepare_model_for_kbit_training(model)
        
        # 创建 PEFT LoraConfig
        peft_config = self._create_peft_config()
        
        # 应用 LoRA
        model = _get_peft_model(model, peft_config)
        
        # 打印可训练参数统计
        trainable, total = self._count_parameters(model)
        LOGGER.info(
            "LoRA 应用完成: 可训练参数 %s / %s (%.2f%%)",
            self._format_number(trainable),
            self._format_number(total),
            100 * trainable / total,
        )
        
        return model

    def _create_peft_config(self) -> Any:
        """创建 PEFT LoraConfig"""
        # 映射任务类型
        task_type_map = {
            TaskType.CAUSAL_LM: _TaskType.CAUSAL_LM,
            TaskType.SEQ_2_SEQ_LM: _TaskType.SEQ_2_SEQ_LM,
            TaskType.SEQ_CLS: _TaskType.SEQ_CLS,
            TaskType.TOKEN_CLS: _TaskType.TOKEN_CLS,
        }
        
        peft_task_type = task_type_map.get(self.config.task_type, _TaskType.CAUSAL_LM)
        
        return _LoraConfig(
            r=self.config.r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias=self.config.bias,
            task_type=peft_task_type,
            modules_to_save=self.config.modules_to_save,
            fan_in_fan_out=self.config.fan_in_fan_out,
            init_lora_weights=self.config.init_lora_weights,
        )

    @staticmethod
    def _count_parameters(model: Any) -> tuple:
        """统计参数量"""
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        return trainable, total

    @staticmethod
    def _format_number(num: int) -> str:
        """格式化数字"""
        if num >= 1e9:
            return f"{num / 1e9:.2f}B"
        elif num >= 1e6:
            return f"{num / 1e6:.2f}M"
        elif num >= 1e3:
            return f"{num / 1e3:.2f}K"
        return str(num)

    @staticmethod
    def get_lora_info(model: Any) -> Dict[str, Any]:
        """获取 LoRA 模型信息"""
        info = {}
        
        if hasattr(model, "peft_config"):
            config = model.peft_config.get("default")
            if config:
                info["r"] = config.r
                info["lora_alpha"] = config.lora_alpha
                info["target_modules"] = list(config.target_modules) if config.target_modules else []
                info["lora_dropout"] = config.lora_dropout
        
        trainable, total = LoRASetup._count_parameters(model)
        info["trainable_parameters"] = trainable
        info["total_parameters"] = total
        info["trainable_ratio"] = trainable / total if total > 0 else 0
        
        return info


# 兼容老 API: create_lora_config 与 detect_target_modules
def create_lora_config(rank: int = 64, alpha: int = 16, dropout: float = 0.05, target_modules: Optional[List[str]] = None) -> LoRAConfig:
    return LoRAConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=target_modules)


def detect_target_modules(model: Any) -> List[str]:
    """尝试从模型名称中检测目标模块名的简单方法 (q_proj/k_proj/v_proj 等)"""
    modules = set()
    if hasattr(model, "named_modules"):
        for name, _mod in model.named_modules():
            if any(key in name for key in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "W_pack", "query_key_value", "dense"]):
                # 根据路径名截取模块关键标识
                for key in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "W_pack", "query_key_value", "dense"]:
                    if key in name:
                        modules.add(key)
    return list(modules)
