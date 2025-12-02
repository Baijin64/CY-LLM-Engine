"""
training.model.setup - 模型加载和设备配置
负责加载基础模型、配置设备和量化设置
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

LOGGER = logging.getLogger("ew.worker.training.model.setup")

# 延迟导入
_torch_imported = False
_transformers_imported = False
_torch = None
_AutoModelForCausalLM = None
_AutoTokenizer = None
_BitsAndBytesConfig = None


def _ensure_torch_imported():
    """确保 PyTorch 已导入"""
    global _torch_imported, _torch
    if not _torch_imported:
        try:
            import torch
            _torch = torch
            _torch_imported = True
        except ImportError as e:
            raise ImportError(f"PyTorch 未安装: {e}") from e


def _ensure_transformers_imported():
    """确保 transformers 已导入"""
    global _transformers_imported, _AutoModelForCausalLM, _AutoTokenizer, _BitsAndBytesConfig
    if not _transformers_imported:
        _ensure_torch_imported()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            _AutoModelForCausalLM = AutoModelForCausalLM
            _AutoTokenizer = AutoTokenizer
            _transformers_imported = True
            
            # 可选：BitsAndBytes
            try:
                from transformers import BitsAndBytesConfig
                _BitsAndBytesConfig = BitsAndBytesConfig
            except ImportError:
                _BitsAndBytesConfig = None
                LOGGER.debug("BitsAndBytesConfig 不可用")
                
        except ImportError as e:
            raise ImportError(f"transformers 未安装: {e}") from e


class DeviceType(Enum):
    """设备类型"""
    CUDA = "cuda"
    CPU = "cpu"
    MPS = "mps"  # Apple Silicon
    AUTO = "auto"


@dataclass
class DeviceConfig:
    """
    设备配置
    
    描述训练/推理使用的硬件配置
    """
    device_type: DeviceType = DeviceType.AUTO
    device_ids: Optional[list] = None  # 指定 GPU ID
    
    # CUDA 相关
    cuda_available: bool = False
    cuda_device_count: int = 0
    cuda_device_name: Optional[str] = None
    cuda_memory_gb: float = 0.0
    
    # 量化支持
    bitsandbytes_available: bool = False
    bf16_supported: bool = False
    
    @classmethod
    def detect(cls) -> "DeviceConfig":
        """自动检测设备配置"""
        _ensure_torch_imported()
        
        config = cls()
        
        # 检测 CUDA
        config.cuda_available = _torch.cuda.is_available()
        if config.cuda_available:
            config.cuda_device_count = _torch.cuda.device_count()
            config.cuda_device_name = _torch.cuda.get_device_name(0)
            config.cuda_memory_gb = _torch.cuda.get_device_properties(0).total_memory / (1024**3)
            config.bf16_supported = _torch.cuda.is_bf16_supported()
            config.device_type = DeviceType.CUDA
        else:
            config.device_type = DeviceType.CPU
        
        # 检测 BitsAndBytes
        try:
            import bitsandbytes
            config.bitsandbytes_available = config.cuda_available  # BnB 需要 CUDA
        except ImportError:
            config.bitsandbytes_available = False
        
        return config

    def get_device_map(self) -> str:
        """获取 device_map 参数"""
        if self.device_type == DeviceType.CUDA:
            return "auto"
        elif self.device_type == DeviceType.CPU:
            return "cpu"
        else:
            return "auto"

    def get_torch_dtype(self):
        """获取推荐的 torch dtype"""
        _ensure_torch_imported()
        
        if self.cuda_available:
            if self.bf16_supported:
                return _torch.bfloat16
            return _torch.float16
        return _torch.float32


@dataclass
class QuantizationConfig:
    """量化配置"""
    enabled: bool = True
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    bnb_4bit_quant_type: str = "nf4"  # nf4 或 fp4
    bnb_4bit_compute_dtype: str = "bfloat16"  # float16, bfloat16, float32
    bnb_4bit_use_double_quant: bool = True

    def to_bnb_config(self, device_config: DeviceConfig) -> Optional[Any]:
        """转换为 BitsAndBytesConfig"""
        _ensure_transformers_imported()
        
        if not self.enabled:
            return None
        
        if _BitsAndBytesConfig is None:
            LOGGER.warning("BitsAndBytesConfig 不可用，跳过量化")
            return None
        
        if not device_config.bitsandbytes_available:
            LOGGER.warning("BitsAndBytes 不可用（需要 CUDA），跳过量化")
            return None
        
        compute_dtype = getattr(_torch, self.bnb_4bit_compute_dtype, _torch.bfloat16)
        
        return _BitsAndBytesConfig(
            load_in_4bit=self.load_in_4bit,
            load_in_8bit=self.load_in_8bit,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
        )


@dataclass
class ModelConfig:
    """模型加载配置"""
    model_path: str
    trust_remote_code: bool = True
    use_cache: bool = False
    low_cpu_mem_usage: bool = True
    
    # 量化
    quantization: Optional[QuantizationConfig] = None
    
    # Tokenizer
    padding_side: str = "right"
    use_fast_tokenizer: bool = True


class ModelSetup:
    """
    模型设置器
    
    职责:
    1. 检测设备能力
    2. 配置量化参数
    3. 加载模型和 Tokenizer
    
    示例:
        >>> setup = ModelSetup(ModelConfig(model_path="deepseek-ai/deepseek-llm-7b"))
        >>> model, tokenizer = setup.load()
    """

    def __init__(
        self,
        config: ModelConfig,
        device_config: Optional[DeviceConfig] = None,
    ):
        _ensure_transformers_imported()
        self.config = config
        self.device_config = device_config or DeviceConfig.detect()

    def load(self) -> Tuple[Any, Any]:
        """
        加载模型和 Tokenizer
        
        Returns:
            (model, tokenizer) 元组
        """
        model = self._load_model()
        tokenizer = self._load_tokenizer()
        return model, tokenizer

    def _load_model(self) -> Any:
        """加载模型"""
        LOGGER.info("加载模型: %s", self.config.model_path)
        LOGGER.info(
            "设备配置: type=%s, cuda=%s, bnb=%s",
            self.device_config.device_type.value,
            self.device_config.cuda_available,
            self.device_config.bitsandbytes_available,
        )
        
        # 构建加载参数
        model_kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
            "low_cpu_mem_usage": self.config.low_cpu_mem_usage,
        }
        
        # 量化配置
        if self.config.quantization:
            bnb_config = self.config.quantization.to_bnb_config(self.device_config)
            if bnb_config:
                model_kwargs["quantization_config"] = bnb_config
                model_kwargs["device_map"] = "auto"
                LOGGER.info("使用 4bit 量化加载")
        
        # 非量化时的设备配置
        if "quantization_config" not in model_kwargs:
            model_kwargs["device_map"] = self.device_config.get_device_map()
            model_kwargs["torch_dtype"] = self.device_config.get_torch_dtype()
        
        # 尝试加载
        try:
            model = _AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                **model_kwargs
            )
        except Exception as e:
            LOGGER.warning("首次加载失败: %s，尝试 CPU fallback", e)
            model = _AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                device_map="cpu",
                torch_dtype=_torch.float32,
                trust_remote_code=self.config.trust_remote_code,
                low_cpu_mem_usage=True,
            )
        
        # 禁用缓存（训练时）
        if not self.config.use_cache:
            model.config.use_cache = False
        
        LOGGER.info("模型加载完成: %s", type(model).__name__)
        return model

    def _load_tokenizer(self) -> Any:
        """加载 Tokenizer"""
        LOGGER.info("加载 Tokenizer: %s", self.config.model_path)
        
        tokenizer = _AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=self.config.trust_remote_code,
            use_fast=self.config.use_fast_tokenizer,
        )
        
        # 设置 pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            LOGGER.debug("设置 pad_token = eos_token")
        
        tokenizer.padding_side = self.config.padding_side
        
        LOGGER.info(
            "Tokenizer 加载完成: vocab_size=%d, pad_token=%s",
            tokenizer.vocab_size,
            tokenizer.pad_token,
        )
        return tokenizer

    @staticmethod
    def get_model_info(model: Any) -> Dict[str, Any]:
        """获取模型信息"""
        info = {
            "type": type(model).__name__,
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
        
        if hasattr(model, "config"):
            info["hidden_size"] = getattr(model.config, "hidden_size", None)
            info["num_layers"] = getattr(model.config, "num_hidden_layers", None)
            info["vocab_size"] = getattr(model.config, "vocab_size", None)
        
        return info
