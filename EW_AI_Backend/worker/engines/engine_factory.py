"""Factory utilities for creating inference engine instances."""

from __future__ import annotations

from typing import Dict, Optional, Type

from ..config.config_loader import HardwareProfile, WorkerConfig, detect_hardware, load_worker_config
from .abstract_engine import BaseEngine
from .ascend_engine import AscendEngine
from .hybrid_engine import HybridEngine
from .nvidia_engine import NvidiaEngine


ENGINE_REGISTRY: Dict[str, Type[BaseEngine]] = {
	"nvidia": NvidiaEngine,
	"cuda": NvidiaEngine,
	"ascend": AscendEngine,
	"npu": AscendEngine,
	"hybrid": HybridEngine,
}

	# 说明：登记不同后端类型对应的引擎类，create_engine 将根据该表实例化适当引擎


def create_engine(engine_type: str, **kwargs) -> BaseEngine:
	"""Instantiate an engine by type identifier."""

	engine_type_normalized = engine_type.lower()
	if engine_type_normalized not in ENGINE_REGISTRY:
		raise ValueError(f"Unknown engine type: {engine_type}")

	engine_cls = ENGINE_REGISTRY[engine_type_normalized]
	return engine_cls(**kwargs)

	# 说明：create_engine 通过 ENGINE_REGISTRY 返回实例，kwargs 会传递给引擎构造函数


def detect_default_engine(hardware: Optional[HardwareProfile] = None) -> BaseEngine:
	"""创建一个与当前硬件最匹配的引擎实例。"""

	hardware = hardware or detect_hardware()
	if hardware.has_ascend:
		preferred = "ascend"
	elif hardware.has_cuda:
		preferred = "nvidia"
	else:
		preferred = "nvidia"

	# 根据探测到的硬件选择默认后端，便于自动部署场景
	return create_engine(preferred)


def create_engine_from_config(config: Optional[WorkerConfig] = None) -> BaseEngine:
	"""利用 WorkerConfig（或自动加载）生成对应的引擎。"""

	config = config or load_worker_config()
	# 从配置中读取 preferred_backend，并创建对应引擎实例
	return create_engine(config.preferred_backend)

