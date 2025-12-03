"""""
server.py
# [gRPC 服务] 实现 Protobuf 中定义的服务：StreamPredict 与控制信道
# 说明：负责接受网关的双向流、调度推理任务、并与内存管理器协调。
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Callable, Dict, Generator, List, Optional, TYPE_CHECKING

from ..engines.abstract_engine import BaseEngine
from ..engines.engine_factory import create_engine
from .memory_manager import GPUMemoryManager
from .task_scheduler import SchedulerBusy, TaskScheduler
from .telemetry import Telemetry

# 延迟导入 PromptCache 以避免循环导入
if TYPE_CHECKING:
    from ..cache.prompt_cache import PromptCache


InferenceEngineFactory = Callable[[str], BaseEngine]


def _get_prompt_cache_lazy():
    """延迟获取 PromptCache 实例"""
    from ..cache.prompt_cache import get_prompt_cache
    return get_prompt_cache()


class InferenceServer:

	# 说明：该类作为推理服务的协调层，负责：
	# - 确保模型按需加载到 GPU（通过 GPUMemoryManager）
	# - 将推理请求提交到本地调度器以限制并发
	# - 在推理过程中收集 Telemetry 指标并以队列方式流式返回生成文本
	# - 支持 Prompt 缓存以加速重复请求

	def __init__(
		self,
		engine_factory: Optional[InferenceEngineFactory] = None,
		scheduler: Optional[TaskScheduler] = None,
		telemetry: Optional[Telemetry] = None,
		prompt_cache: Optional["PromptCache"] = None,
		enable_cache: bool = False,  # 兼容旧测试
	) -> None:
		self._engine_factory = engine_factory or self._default_engine_factory
		self._scheduler = scheduler or TaskScheduler()
		self._telemetry = telemetry or Telemetry()
		self._memory = GPUMemoryManager()
		self._model_lock: Dict[str, threading.Lock] = {}
		# Prompt 缓存：用于缓存完整的推理结果
		self._prompt_cache = prompt_cache
		self._enable_cache = enable_cache
		# 已加载模型记录
		self._loaded_models: Dict[str, dict] = {}

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
		enable_prompt_cache: bool = False,
		prompt_cache_ttl: Optional[int] = None,
	) -> Generator[str, None, None]:
		# 参数检查：prompt 不能为空
		if not prompt.strip():
			raise ValueError("Prompt must not be empty.")

		gen_kwargs = dict(generation_kwargs or {})

		# ========== Prompt 缓存检查 ==========
		if enable_prompt_cache:
			cache = self._prompt_cache or _get_prompt_cache_lazy()
			# 直接使用 PromptCache 的 get 方法（基于 prompt + model_id + params）
			cached_result = cache.get(prompt, model_id, **gen_kwargs)
			if cached_result is not None:
				# 缓存命中，直接返回缓存的结果
				self._telemetry.track_cache_hit()
				yield cached_result
				return

		response_queue: "queue.Queue[object]" = queue.Queue()
		sentinel = object()
		collected_chunks: List[str] = []  # 用于收集响应以便缓存

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
					if enable_prompt_cache:
						collected_chunks.append(chunk)
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

		# ========== 缓存完整响应 ==========
		if enable_prompt_cache and collected_chunks:
			cache = self._prompt_cache or _get_prompt_cache_lazy()
			full_response = "".join(collected_chunks)
			ttl = prompt_cache_ttl or 3600  # 默认 1 小时
			cache.set(prompt, model_id, full_response, ttl=ttl, **gen_kwargs)

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

	async def async_unload_model(self, model_id: str) -> None:
		"""异步卸载模型（在线程池中执行）"""
		import asyncio
		loop = asyncio.get_running_loop()
		await loop.run_in_executor(None, self.unload_model, model_id)

	async def async_stream_predict(
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
		enable_prompt_cache: bool = False,
		prompt_cache_ttl: Optional[int] = None,
	):
		"""
		异步流式推理。
		
		将同步的 stream_predict 包装为异步生成器，
		允许在异步上下文中使用。
		"""
		import asyncio
		
		loop = asyncio.get_running_loop()
		response_queue: "queue.Queue[object]" = queue.Queue()
		sentinel = object()
		
		def _sync_generate():
			try:
				for chunk in self.stream_predict(
					model_id=model_id,
					prompt=prompt,
					model_path=model_path,
					adapter_path=adapter_path,
					engine_type=engine_type,
					generation_kwargs=generation_kwargs,
					engine_kwargs=engine_kwargs,
					priority=priority,
					enable_prompt_cache=enable_prompt_cache,
					prompt_cache_ttl=prompt_cache_ttl,
				):
					response_queue.put(chunk)
			except Exception as e:
				response_queue.put(e)
			finally:
				response_queue.put(sentinel)
		
		# 在线程池中运行同步生成
		loop.run_in_executor(None, _sync_generate)
		
		# 异步轮询队列
		while True:
			# 非阻塞获取，配合 sleep 让出控制权
			try:
				chunk = response_queue.get_nowait()
				if chunk is sentinel:
					break
				if isinstance(chunk, Exception):
					raise chunk
				yield chunk
			except queue.Empty:
				await asyncio.sleep(0.001)  # 1ms 轮询间隔

	def shutdown(self) -> None:
		self._scheduler.shutdown()

	# ====== 兼容旧测试 API ======
	
	def load_model(self, model_id: str, model_path: str, **kwargs) -> bool:
		"""加载模型 (兼容旧测试 API)"""
		try:
			self.ensure_model(
				model_id,
				model_path=model_path,
				**kwargs
			)
			self._loaded_models[model_id] = {"path": model_path, **kwargs}
			return True
		except Exception:
			return False
	
	def get_loaded_models(self) -> List[str]:
		"""获取已加载模型列表 (兼容旧测试 API)"""
		return list(self._loaded_models.keys())
	
	def get_model_info(self, model_id: str) -> Optional[dict]:
		"""获取模型信息 (兼容旧测试 API)"""
		return self._loaded_models.get(model_id)
	
	def health_check(self) -> bool:
		"""健康检查 (兼容旧测试 API)"""
		return True
	
	def get_memory_usage(self) -> Dict:
		"""获取显存使用情况 (兼容旧测试 API)"""
		return self._memory.get_memory_info()
