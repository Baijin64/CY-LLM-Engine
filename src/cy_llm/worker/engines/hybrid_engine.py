"""
hybrid_engine.py
# [可选] 混合策略引擎占位：支持跨设备拆分模型或流水线推理
# 说明：用于实现复杂场景下的设备协同推理（可选模块）。
"""

from __future__ import annotations

from typing import Dict, Generator, Iterable, Optional

from .abstract_engine import BaseEngine
from .ascend_engine import AscendEngine
from .nvidia_engine import NvidiaEngine


ENGINE_ALIAS = {
	"nvidia": NvidiaEngine,
	"cuda": NvidiaEngine,
	"ascend": AscendEngine,
	"npu": AscendEngine,
}


class HybridEngine(BaseEngine):
	"""Attempt to run on preferred backends, falling back when unavailable."""

	# 说明：混合引擎根据优先级尝试在多个后端上加载模型（例如先尝试 Nvidia，再尝试 Ascend）
	# - 适用于需要在不同硬件间回退或动态选择后端的部署场景
	# - 当前实现很轻量：尝试顺序加载，成功后将该后端设为活跃并在其上推理

	def __init__(self, preference: Optional[Iterable[str]] = None) -> None:
		self._preference = tuple(preference or ("nvidia", "ascend"))
		self._engines: Dict[str, BaseEngine] = {}
		self._active_backend: Optional[str] = None

	def _get_or_create(self, backend: str) -> BaseEngine:
		backend = backend.lower()
		if backend in self._engines:
			return self._engines[backend]

		engine_cls = ENGINE_ALIAS.get(backend)
		if engine_cls is None:
			raise ValueError(f"Unsupported backend in hybrid preference: {backend}")

		# 创建并缓存对应后端的引擎实例
		engine = engine_cls()
		self._engines[backend] = engine
		return engine

	def load_model(self, model_path: str, adapter_path: Optional[str] = None, **kwargs) -> None:
		last_error: Optional[Exception] = None

		# 按偏好顺序逐个尝试加载，遇到第一个成功的后端即作为活动后端
		for backend in self._preference:
			try:
				engine = self._get_or_create(backend)
				engine.load_model(model_path, adapter_path=adapter_path, **kwargs)
			except Exception as exc:  # noqa: BLE001
				last_error = exc
				continue

			# 成功加载，标记活跃后端并返回
			self._active_backend = backend
			return

		raise RuntimeError("No backend in HybridEngine succeeded.") from last_error

	def infer(self, prompt: str, **kwargs) -> Generator[str, None, None]:
		engine = self._active_engine()
		yield from engine.infer(prompt, **kwargs)

	# 说明：infer 简单代理到当前活跃后端的 infer 接口

	def unload_model(self) -> None:
		if self._active_backend is None:
			return
		engine = self._engines.get(self._active_backend)
		if engine:
			engine.unload_model()
		self._active_backend = None

	# 说明：卸载时只清理活跃后端的模型，其他后端实例保留以便重复使用（可调整）

	def get_memory_usage(self):  # type: ignore[override]
		if self._active_backend is None:
			return {"used": 0.0, "total": 0.0}
		engine = self._engines.get(self._active_backend)
		if not engine:
			return {"used": 0.0, "total": 0.0}
		return engine.get_memory_usage()

	def _active_engine(self) -> BaseEngine:
		if self._active_backend is None:
			raise RuntimeError("No backend is currently active. Call load_model first.")
		engine = self._engines.get(self._active_backend)
		if engine is None:
			raise RuntimeError("Active backend is missing its engine instance.")
		return engine

