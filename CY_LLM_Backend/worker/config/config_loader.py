"""
config_loader.py
[配置] Worker 运行时配置加载器
说明：加载硬件检测、后端选择、模型注册表等配置。

支持的引擎类型：
  - cuda-vllm:      NVIDIA GPU + vLLM（通用，推荐）
  - cuda-trt:       NVIDIA GPU + TensorRT-LLM（极致性能）
  - ascend-vllm:    华为 NPU + vLLM-Ascend（通用）
  - ascend-mindie:  华为 NPU + MindIE Turbo（极致性能）
  - nvidia:         旧版兼容（transformers）
  - ascend:         旧版兼容（transformers）
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

LOGGER = logging.getLogger("cy_llm.worker.config")

# (已完成迁移) 统一使用 CY_LLM_* 环境变量，不再做向后兼容的 EW_* 映射

# 延迟导入 torch，避免在无 GPU 环境下报错
_torch = None
_torch_npu = None


def _ensure_torch_imported():
    """延迟导入 torch"""
    global _torch, _torch_npu
    if _torch is None:
        try:
            import torch
            _torch = torch
        except ImportError:
            LOGGER.warning("PyTorch 未安装，硬件检测将受限")
            return False
            
        try:
            import torch_npu
            _torch_npu = torch_npu
        except ImportError:
            pass  # Ascend 可选
            
    return True


# ============================================================================
# 环境变量定义
# ============================================================================

ENV_PREFERRED_BACKEND = "CY_LLM_BACKEND"           # 引擎类型，如 "cuda-vllm"
ENV_ENGINE_TYPE = "CY_LLM_ENGINE"                  # 别名，与 CY_LLM_BACKEND 等效
ENV_MODEL_REGISTRY = "CY_LLM_MODEL_REGISTRY"       # JSON 格式的模型注册表
ENV_MODEL_REGISTRY_PATH = "CY_LLM_MODEL_REGISTRY_PATH"  # 模型注册表文件路径
ENV_DEFAULT_MODEL = "CY_LLM_DEFAULT_MODEL"         # 默认模型路径
ENV_DEFAULT_ADAPTER = "CY_LLM_DEFAULT_ADAPTER"     # 默认 LoRA 适配器路径

# 引擎类型默认值（按硬件自动选择）
DEFAULT_ENGINE_BY_HARDWARE = {
    "cuda": "cuda-vllm",       # NVIDIA 默认使用 vLLM
    "ascend": "ascend-vllm",   # Ascend 默认使用 vLLM-Ascend
    "cpu": "nvidia",           # CPU 回退到 transformers
}

# 所有支持的引擎类型
SUPPORTED_ENGINES: List[str] = [
    "cuda-vllm",
    "cuda-trt",
    "ascend-vllm",
    "ascend-mindie",
    "nvidia",
    "cuda",
    "ascend",
    "npu",
    "hybrid",
]


# ============================================================================
# 数据类定义
# ============================================================================

@dataclass(frozen=True)
class HardwareProfile:
    """描述当前节点的硬件能力。"""

    has_cuda: bool
    cuda_device_count: int
    has_ascend: bool
    ascend_device_count: int
    
    @property
    def primary_device(self) -> str:
        """返回主要设备类型"""
        if self.has_ascend:
            return "ascend"
        elif self.has_cuda:
            return "cuda"
        else:
            return "cpu"


@dataclass(frozen=True)
class ModelSpec:
    """记录单个逻辑模型对应的物理加载信息。"""

    model_path: str
    adapter_path: Optional[str] = None
    engine: Optional[str] = None      # 可为单个模型指定引擎
    use_4bit: Optional[bool] = None
    max_model_len: Optional[int] = None
    tensor_parallel_size: Optional[int] = None
    # KV Cache 优化配置
    enable_prefix_caching: Optional[bool] = None  # 启用前缀缓存
    kv_cache_dtype: Optional[str] = None          # KV Cache 数据类型 (auto, fp8)
    gpu_memory_utilization: Optional[float] = None  # GPU 显存利用率 (0.0-1.0)
    # Prompt 缓存配置
    enable_prompt_cache: Optional[bool] = None    # 启用 Prompt 结果缓存
    prompt_cache_ttl: Optional[int] = None        # Prompt 缓存 TTL (秒)


@dataclass(frozen=True)
class WorkerConfig:
    """Worker 启动时需要的聚合配置。"""

    preferred_backend: str            # 引擎类型
    hardware: HardwareProfile
    model_registry: Dict[str, ModelSpec] = field(default_factory=dict)
    
    def get_engine_type(self) -> str:
        """获取规范化的引擎类型"""
        # 优先级：
        #   1. CY_LLM_ENGINE 或 CY_LLM_BACKEND 环境变量
        #   2. 根据硬件自动选择（CUDA -> cuda-vllm, Ascend -> ascend-vllm）
# ============================================================================
# 硬件检测
# ============================================================================

def detect_hardware() -> HardwareProfile:
    """检测 CUDA / Ascend 设备可用性，用于后续自动选择后端。"""
    
    if not _ensure_torch_imported():
        return HardwareProfile(
            has_cuda=False,
            cuda_device_count=0,
            has_ascend=False,
            ascend_device_count=0,
        )

    has_cuda = _torch.cuda.is_available() if _torch else False
    cuda_device_count = _torch.cuda.device_count() if has_cuda else 0

    ascend_available = False
    ascend_device_count = 0
    
    if _torch_npu is not None:
        try:
            ascend_available = _torch_npu.npu.is_available()
            ascend_device_count = _torch_npu.npu.device_count() if ascend_available else 0
        except Exception:
            pass
    elif _torch and hasattr(_torch, "npu"):
        try:
            ascend_available = _torch.npu.is_available()
            ascend_device_count = _torch.npu.device_count() if ascend_available else 0
        except Exception:
            pass

    profile = HardwareProfile(
        has_cuda=has_cuda,
        cuda_device_count=cuda_device_count,
        has_ascend=ascend_available,
        ascend_device_count=ascend_device_count,
    )
    
    LOGGER.info(
        "硬件检测: CUDA=%s(%d), Ascend=%s(%d)",
        has_cuda, cuda_device_count, ascend_available, ascend_device_count
    )
    
    return profile


# ============================================================================
# 后端选择
# ============================================================================

def determine_backend(hardware: HardwareProfile, preferred: Optional[str] = None) -> str:
    """
    结合硬件状况与显式配置，决策本 Worker 使用的后端。
    
    Args:
        hardware: 硬件配置
        preferred: 用户指定的引擎类型（可选）
        
    Returns:
        引擎类型字符串
    """
    # 如果用户明确指定了引擎，优先使用
    if preferred:
        preferred_normalized = preferred.lower().strip()
        if preferred_normalized in SUPPORTED_ENGINES:
            LOGGER.info("使用用户指定的引擎: %s", preferred_normalized)
            return preferred_normalized
        else:
            LOGGER.warning(
                "未知的引擎类型 '%s'，将自动选择。支持的类型: %s",
                preferred, ", ".join(SUPPORTED_ENGINES)
            )
    
    # 根据硬件自动选择
    device = hardware.primary_device
    engine = DEFAULT_ENGINE_BY_HARDWARE.get(device, "nvidia")
    LOGGER.info("自动选择引擎: %s (基于 %s 硬件)", engine, device)
    return engine


def validate_engine_type(engine_type: str) -> bool:
    """验证引擎类型是否有效"""
    return engine_type.lower().strip() in SUPPORTED_ENGINES


# ============================================================================
# 模型注册表
# ============================================================================

def _parse_registry_json(payload: str) -> Dict[str, ModelSpec]:
    """解析 JSON 格式的模型注册表"""
    data = json.loads(payload)
    registry: Dict[str, ModelSpec] = {}
    
    for logical_name, spec in data.items():
        if not isinstance(spec, dict) or "model_path" not in spec:
            raise ValueError(f"Invalid registry item: {logical_name}")
            
        registry[logical_name] = ModelSpec(
            model_path=spec["model_path"],
            adapter_path=spec.get("adapter_path"),
            engine=spec.get("engine"),
            use_4bit=spec.get("use_4bit"),
            max_model_len=spec.get("max_model_len"),
            tensor_parallel_size=spec.get("tensor_parallel_size"),
            # KV Cache 配置
            enable_prefix_caching=spec.get("enable_prefix_caching"),
            kv_cache_dtype=spec.get("kv_cache_dtype"),
            gpu_memory_utilization=spec.get("gpu_memory_utilization"),
            # Prompt 缓存配置
            enable_prompt_cache=spec.get("enable_prompt_cache"),
            prompt_cache_ttl=spec.get("prompt_cache_ttl"),
        )
        
    return registry


def load_model_registry(path: Optional[str] = None) -> Dict[str, ModelSpec]:
    """从环境变量或 JSON 文件加载模型映射。"""

    # 优先从环境变量读取 JSON
    if ENV_MODEL_REGISTRY in os.environ:
        LOGGER.info("从环境变量加载模型注册表")
        return _parse_registry_json(os.environ[ENV_MODEL_REGISTRY])

    # 其次从文件读取
    registry_path = path or os.environ.get(ENV_MODEL_REGISTRY_PATH)
    if registry_path:
        file_path = Path(registry_path).expanduser()
        if not file_path.exists():
            raise FileNotFoundError(f"Model registry file not found: {file_path}")
        LOGGER.info("从文件加载模型注册表: %s", file_path)
        return _parse_registry_json(file_path.read_text(encoding="utf-8"))

    # 默认注册表
    default_model = os.environ.get(ENV_DEFAULT_MODEL, "deepseek-ai/deepseek-llm-7b-chat")
    adapter = os.environ.get(ENV_DEFAULT_ADAPTER)
    LOGGER.info("使用默认模型注册表: model=%s, adapter=%s", default_model, adapter)
    
    return {
        "default": ModelSpec(
            model_path=default_model,
            adapter_path=adapter,
            engine=None,
            use_4bit=True
        ),
    }


# ============================================================================
# 主配置加载
# ============================================================================

def load_worker_config(registry_path: Optional[str] = None) -> WorkerConfig:
    """
    供 Worker 启动入口调用的统一配置装载函数。
    
    优先级：
      1. EW_ENGINE 或 EW_BACKEND 环境变量
      2. 根据硬件自动选择（CUDA -> cuda-vllm, Ascend -> ascend-vllm）
      
    Returns:
        WorkerConfig 实例
    """
    hardware = detect_hardware()
    
    # 支持两个环境变量名（EW_ENGINE 或 EW_BACKEND）
    preferred = os.environ.get(ENV_ENGINE_TYPE) or os.environ.get(ENV_PREFERRED_BACKEND)
    backend = determine_backend(hardware, preferred)
    
    registry = load_model_registry(registry_path)
    
    config = WorkerConfig(
        preferred_backend=backend,
        hardware=hardware,
        model_registry=registry
    )
    
    LOGGER.info(
        "Worker 配置加载完成: engine=%s, models=%s",
        backend, list(registry.keys())
    )
    
    return config


def print_config_help():
    """打印配置帮助信息"""
    help_text = """
EW AI Worker 配置说明
=====================

环境变量：
  EW_ENGINE / EW_BACKEND    引擎类型（见下方列表）
  EW_MODEL_REGISTRY         JSON 格式的模型注册表
  EW_MODEL_REGISTRY_PATH    模型注册表文件路径
  EW_DEFAULT_MODEL          默认模型（HuggingFace ID 或本地路径）
  EW_DEFAULT_ADAPTER        默认 LoRA 适配器路径

支持的引擎类型：
  cuda-vllm       NVIDIA GPU + vLLM（通用，高性能，推荐）
  cuda-trt        NVIDIA GPU + TensorRT-LLM（极致性能，需预编译）
  ascend-vllm     华为 NPU + vLLM-Ascend（通用，CANN 环境）
  ascend-mindie   华为 NPU + MindIE Turbo（极致性能，华为原生）
  nvidia          旧版兼容（transformers，支持 CPU）
  ascend          旧版兼容（transformers）

示例：
  export EW_ENGINE=cuda-vllm
  export EW_DEFAULT_MODEL=deepseek-ai/deepseek-llm-7b-chat
  python -m worker.main
"""
    print(help_text)


