"""""
server.py
# [gRPC 服务] 实现 Protobuf 中定义的服务：StreamPredict 与控制信道
# 说明：负责接受网关的双向流、调度推理任务、并与内存管理器协调。
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Callable, Dict, Generator, Optional

from ..engines.abstract_engine import BaseEngine
from ..engines.engine_factory import create_engine
from .memory_manager import GPUMemoryManager
from .task_scheduler import SchedulerBusy, TaskScheduler
from .telemetry import Telemetry


InferenceEngineFactory = Callable[[str], BaseEngine]


class InferenceServer:

	# 说明：该类作为推理服务的协调层，负责：
	# - 确保模型按需加载到 GPU（通过 GPUMemoryManager）
	# - 将推理请求提交到本地调度器以限制并发
	# - 在推理过程中收集 Telemetry 指标并以队列方式流式返回生成文本

	def __init__(
		self,
		engine_factory: Optional[InferenceEngineFactory] = None,
		scheduler: Optional[TaskScheduler] = None,
		telemetry: Optional[Telemetry] = None,
	) -> None:
		self._engine_factory = engine_factory or self._default_engine_factory
		self._scheduler = scheduler or TaskScheduler()
		self._telemetry = telemetry or Telemetry()
		self._memory = GPUMemoryManager()
		self._model_lock: Dict[str, threading.Lock] = {}

	def _default_engine_factory(self, engine_type: str) -> BaseEngine:
		return create_engine(engine_type)

	def ensure_model(
		self,
		model_id: str,
		*,
		model_path: str,
		adapter_path: Optional[str] = None,
		engine_type: str = "nvidia",
		engine_kwargs: Optional[Dict] = None,
	) -> BaseEngine:
		# 如果内存管理器中已有加载的模型，直接返回
		engine = self._memory.get_loaded_model(model_id)
		if engine is not None:
			return engine

		# 使用 per-model 锁防止并发重复加载同一模型
		lock = self._model_lock.setdefault(model_id, threading.Lock())
		with lock:
			# 再次检查，避免竞争条件下重复加载
			engine = self._memory.get_loaded_model(model_id)
			if engine is not None:
				return engine

			# 通过工厂创建引擎并加载模型，然后交由内存管理器注册
			engine = self._engine_factory(engine_type)
			engine.load_model(model_path, adapter_path=adapter_path, **(engine_kwargs or {}))
			self._memory.register_model(model_id, engine)
			return engine

	def stream_predict(
		self,
		*,
		model_id: str,
		prompt: str,
		model_path: str,
		adapter_path: Optional[str] = None,
		engine_type: str = "nvidia",
		generation_kwargs: Optional[Dict] = None,
		engine_kwargs: Optional[Dict] = None,
		priority: int = 0,
	) -> Generator[str, None, None]:
		# 参数检查：prompt 不能为空
		if not prompt.strip():
			raise ValueError("Prompt must not be empty.")

		response_queue: "queue.Queue[object]" = queue.Queue()
		sentinel = object()
		gen_kwargs = dict(generation_kwargs or {})

		# 工作函数：由调度器在后台线程调用，负责实际的推理并把结果放入队列
		def _task() -> None:
			start = time.time()
			self._telemetry.track_request_start()
			try:
				# 确保模型已加载（可能会触发加载）
				engine = self.ensure_model(
					model_id,
					model_path=model_path,
					adapter_path=adapter_path,
					engine_type=engine_type,
					engine_kwargs=engine_kwargs,
				)
				# 调用引擎的 infer，流式读取生成数据并入队
				for chunk in engine.infer(prompt, **gen_kwargs):
					response_queue.put(chunk)
				# 成功完成一次请求
				self._telemetry.track_request_end(time.time() - start, success=True)
				response_queue.put(sentinel)
			except Exception as exc:  # noqa: BLE001
				# 发生异常时记录并将异常对象放入队列，外层会重新抛出
				self._telemetry.track_request_end(time.time() - start, success=False)
				response_queue.put(exc)
			finally:
				# 标记该模型最近访问时间，防止被误回收
				self._memory.access_model(model_id)

		try:
			self._scheduler.submit(_task, priority=priority)
		except SchedulerBusy as exc:  # noqa: PERF203
			raise RuntimeError("Server is overloaded, please retry later.") from exc

		while True:
			chunk = response_queue.get()
			if chunk is sentinel:
				break
			if isinstance(chunk, Exception):
				raise chunk
			yield chunk

	def unload_model(self, model_id: str) -> None:
		engine = self._memory.get_loaded_model(model_id)
		if engine is None:
			return
		lock = self._model_lock.setdefault(model_id, threading.Lock())
		with lock:
			engine = self._memory.get_loaded_model(model_id)
			if engine is None:
				return
			engine.unload_model()

	def shutdown(self) -> None:
		self._scheduler.shutdown()
