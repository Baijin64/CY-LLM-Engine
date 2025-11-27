"""Runtime configuration helpers for the worker process."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import torch

try:  # Ascend 设备可选依赖
	import torch_npu  # type: ignore  # noqa: F401
except ImportError:  # pragma: no cover - 在无 Ascend 的环境里是预期行为
	torch_npu = None  # type: ignore


ENV_PREFERRED_BACKEND = "EW_BACKEND"
ENV_MODEL_REGISTRY = "EW_MODEL_REGISTRY"
ENV_MODEL_REGISTRY_PATH = "EW_MODEL_REGISTRY_PATH"
ENV_DEFAULT_MODEL = "EW_DEFAULT_MODEL"
ENV_DEFAULT_ADAPTER = "EW_DEFAULT_ADAPTER"


@dataclass(frozen=True)
class HardwareProfile:
	"""描述当前节点的硬件能力。"""

	has_cuda: bool
	cuda_device_count: int
	has_ascend: bool
	ascend_device_count: int


@dataclass(frozen=True)
class ModelSpec:
	"""记录单个逻辑模型对应的物理加载信息。"""

	model_path: str
	adapter_path: Optional[str] = None
	engine: Optional[str] = None
	use_4bit: Optional[bool] = None


@dataclass(frozen=True)
class WorkerConfig:
	"""Worker 启动时需要的聚合配置。"""

	preferred_backend: str
	hardware: HardwareProfile
	model_registry: Dict[str, ModelSpec] = field(default_factory=dict)


def detect_hardware() -> HardwareProfile:
	"""检测 CUDA / Ascend 设备可用性，用于后续自动选择后端。"""

	has_cuda = torch.cuda.is_available()
	cuda_device_count = torch.cuda.device_count() if has_cuda else 0

	ascend_available = bool(getattr(torch, "npu", None)) and getattr(torch.npu, "is_available", lambda: False)()
	ascend_device_count = getattr(torch.npu, "device_count", lambda: 0)() if ascend_available else 0

	return HardwareProfile(
		has_cuda=has_cuda,
		cuda_device_count=cuda_device_count,
		has_ascend=ascend_available,
		ascend_device_count=ascend_device_count,
	)


def determine_backend(hardware: HardwareProfile, preferred: Optional[str] = None) -> str:
	"""结合硬件状况与显式配置，决策本 Worker 使用的后端。"""

	if preferred:
		return preferred.lower()
	if hardware.has_ascend:
		return "ascend"
	if hardware.has_cuda:
		return "nvidia"
	return "nvidia"  # 默认回落到 NvidiaEngine（可运行在 CPU 上）


def _parse_registry_json(payload: str) -> Dict[str, ModelSpec]:
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
		)
	return registry


def load_model_registry(path: Optional[str] = None) -> Dict[str, ModelSpec]:
	"""从环境变量或 JSON 文件加载模型映射。"""

	if ENV_MODEL_REGISTRY in os.environ:
		return _parse_registry_json(os.environ[ENV_MODEL_REGISTRY])

	registry_path = path or os.environ.get(ENV_MODEL_REGISTRY_PATH)
	if registry_path:
		file_path = Path(registry_path).expanduser()
		if not file_path.exists():
			raise FileNotFoundError(f"Model registry file not found: {file_path}")
		return _parse_registry_json(file_path.read_text(encoding="utf-8"))

	# 最简默认映射：允许通过环境变量覆盖
	default_model = os.environ.get(ENV_DEFAULT_MODEL, "deepseek-ai/deepseek-llm-7b-chat")
	adapter = os.environ.get(ENV_DEFAULT_ADAPTER)
	return {
		"default": ModelSpec(model_path=default_model, adapter_path=adapter, engine=None, use_4bit=True),
	}


def load_worker_config(registry_path: Optional[str] = None) -> WorkerConfig:
	"""供 Worker 启动入口调用的统一配置装载函数。"""

	hardware = detect_hardware()
	preferred = determine_backend(hardware, os.environ.get(ENV_PREFERRED_BACKEND))
	registry = load_model_registry(registry_path)
	return WorkerConfig(preferred_backend=preferred, hardware=hardware, model_registry=registry)

