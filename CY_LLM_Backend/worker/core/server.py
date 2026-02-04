"""""
server.py
# [gRPC 服务] 实现 Protobuf 中定义的服务：StreamPredict 与控制信道
# 说明：负责接受网关的双向流、调度推理任务、并与内存管理器协调。
"""

from __future__ import annotations

import os
import queue
import threading
import time
from typing import Callable, Dict, Generator, List, Optional, TYPE_CHECKING, cast

import logging

from ..engines.abstract_engine import BaseEngine
from ..engines.engine_factory import create_engine
from ..config.config_loader import ModelSpec, WorkerConfig
from ..exceptions import GPUMemoryError
from ..utils.diagnostic import check_vram_for_model, format_vram_report
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


LOGGER = logging.getLogger("cy_llm.worker.server")


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
		worker_config: Optional[WorkerConfig] = None,
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
		self._worker_config = worker_config

	def _default_engine_factory(self, engine_type: str) -> BaseEngine:
		return create_engine(engine_type)

	def _load_model_with_retry(
		self,
		model_id: str,
		engine: BaseEngine,
		model_path: str,
		adapter_path: Optional[str],
		base_kwargs: Optional[dict],
		*,
		preflight_report=None,
		max_retries: int = 4,
		progress_callback: Optional[Callable[[str], None]] = None,
	) -> None:
		"""带自动降级策略的模型加载。"""
		import gc
		import torch

		base_kwargs = dict(base_kwargs or {})
		retry_plan = self._build_retry_plan(base_kwargs, max_retries)
		attempts = len(retry_plan)
		last_error: Optional[Exception] = None

		for attempt, overrides in enumerate(retry_plan, 1):
			effective_kwargs = {**base_kwargs, **overrides}
			try:
				load_msg = f"正在加载模型 {model_id}"
				if attempts > 1:
					load_msg += f" (尝试 {attempt}/{attempts})"
				load_msg += "..."
				LOGGER.info(load_msg)
				if progress_callback:
					progress_callback(load_msg)
				engine.load_model(
					model_path,
					adapter_path,
					**effective_kwargs,
				)
				if overrides:
					warn_msg = f"模型 {model_id} 加载成功（尝试 {attempt}/{attempts}，降级配置: {overrides}）"
					LOGGER.warning(warn_msg)
					if progress_callback:
						progress_callback(warn_msg)
				return
			except RuntimeError as exc:
				if not self._is_cuda_oom(exc):
					raise
				last_error = exc
				oom_msg = f"模型 {model_id} 加载 OOM（尝试 {attempt}/{attempts}），正在重试..."
				LOGGER.warning(oom_msg)
				if progress_callback:
					progress_callback(oom_msg)
				gc.collect()
				if torch.cuda.is_available():
					torch.cuda.empty_cache()

		if last_error is not None:
			free_mb = 0.0
			if torch.cuda.is_available():
				free, _total = torch.cuda.mem_get_info()
				free_mb = free / (1024 ** 2)
			suggestions = []
			if preflight_report and preflight_report.suggestions:
				suggestions.extend(preflight_report.suggestions)
			suggestions.append("尝试启用 AWQ/GPTQ 量化或降低 batch/max_model_len")
			raise GPUMemoryError(
				required_mb=(preflight_report.required_vram_gb * 1024) if preflight_report else 0.0,
				available_mb=free_mb,
				suggestions=list(dict.fromkeys(suggestions)),
			) from last_error

	def _build_retry_plan(self, base_kwargs: dict, max_retries: int) -> List[dict]:
		plan: List[dict] = [{}]
		base_util = float(base_kwargs.get("gpu_memory_utilization") or 0.75)
		base_max_len = int(base_kwargs.get("max_model_len") or 8192)

		plan.append({"gpu_memory_utilization": min(base_util, 0.70)})
		plan.append({
			"gpu_memory_utilization": min(base_util, 0.60),
			"max_model_len": min(base_max_len, 4096),
		})
		plan.append({
			"gpu_memory_utilization": 0.55,
			"max_model_len": min(base_max_len, 2048),
		})

		return plan[:max_retries]

	@staticmethod
	def _is_cuda_oom(exc: Exception) -> bool:
		message = str(exc).lower()
		return "out of memory" in message or "cuda error" in message

	def _apply_vram_headroom_adjustments(
		self,
		report,
		engine_kwargs: Optional[Dict],
		model_spec: Optional[ModelSpec],
	) -> Optional[Dict]:
		if report is None:
			return engine_kwargs
		kwargs = dict(engine_kwargs or {})
		if report.required_vram_gb <= 0 or report.available_vram_gb <= 0:
			return kwargs or None

		available_ratio = report.available_vram_gb / max(report.required_vram_gb, 1e-6)
		adjusted = False

		if available_ratio < 1.15:
			current_util = float(kwargs.get("gpu_memory_utilization") or getattr(model_spec, "gpu_memory_utilization", 0.75) or 0.75)
			recommended_util = round(max(0.55, min(current_util, available_ratio * 0.85)), 2)
			if recommended_util < current_util:
				kwargs["gpu_memory_utilization"] = recommended_util
				adjusted = True

		if available_ratio < 1.05:
			base_max_len = int(kwargs.get("max_model_len") or getattr(model_spec, "max_model_len", 8192) or 8192)
			new_len = max(2048, int(base_max_len * available_ratio * 0.95))
			if new_len < base_max_len:
				kwargs["max_model_len"] = new_len
				adjusted = True

		if adjusted:
			LOGGER.warning(
				"VRAM 余量不足 (%.2fx)，自动调整推理配置: %s",
				available_ratio,
				kwargs,
			)

		return kwargs or None

	def ensure_model(
		self,
		model_id: str,
		*,
		model_path: str,
		adapter_path: Optional[str] = None,
		engine_type: str = "nvidia",
		engine_kwargs: Optional[Dict] = None,
		progress_callback: Optional[Callable[[str], None]] = None,
	) -> BaseEngine:
		# 如果内存管理器中已有加载的模型，直接返回
		engine = self._memory.get_loaded_model(model_id)
		if engine is not None:
			return engine

		if model_path == "/path/to/model" and os.getenv("CY_LLM_ALLOW_PLACEHOLDER_MODEL") != "true":
			raise ValueError("Placeholder model path")

        # 使用 per-model 锁防止并发重复加载同一模型
		lock = self._model_lock.setdefault(model_id, threading.Lock())
		with lock:
			# 再次检查，避免竞争条件下重复加载
			engine = self._memory.get_loaded_model(model_id)
			if engine is not None:
				return engine

			preflight_report = None
			model_spec = self._resolve_model_spec(model_id)
			if model_spec is not None:
				if progress_callback:
					progress_callback("正在检查显存资源...")
				preflight_report = check_vram_for_model(model_id, model_spec)
				if not preflight_report.success:
					LOGGER.error(format_vram_report(preflight_report))
					raise GPUMemoryError(
						required_mb=preflight_report.required_vram_gb * 1024,
						available_mb=preflight_report.available_vram_gb * 1024,
						suggestions=preflight_report.suggestions,
					)
				engine_kwargs = self._apply_vram_headroom_adjustments(
					preflight_report,
					engine_kwargs,
					model_spec,
				)

			# 通过工厂创建引擎并加载模型，然后交由内存管理器注册
			if progress_callback:
				progress_callback(f"正在初始化 {engine_type} 引擎...")
			engine = self._engine_factory(engine_type)
			
			# 使用带重试的模型加载
			self._load_model_with_retry(
				model_id,
				engine,
				model_path,
				adapter_path,
				engine_kwargs or {},
				preflight_report=preflight_report,
				max_retries=4,
				progress_callback=progress_callback,
			)
			
			if progress_callback:
				progress_callback("模型加载完成，准备开始推理...")
			self._memory.register_model(model_id, engine)
			return engine

	def _resolve_model_spec(self, model_id: str) -> Optional[ModelSpec]:
		if not self._worker_config:
			return None
		return self._worker_config.model_registry.get(model_id)
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
		if len(prompt) > 50000:
			raise ValueError("Prompt too long (max 50000 chars).")

		gen_kwargs = dict(generation_kwargs or {})

		# ========== Prompt 缓存检查 ==========
		if enable_prompt_cache:
			cache = self._prompt_cache or _get_prompt_cache_lazy()
			# 直接使用 PromptCache 的 get 方法（基于 prompt + model_id + params）
			cached_result = cache.get(prompt, model_id, **gen_kwargs)
			if cached_result is not None:
				# 缓存命中，直接返回缓存的结果
				self._telemetry.track_cache_hit()
				yield cast(str, cached_result)
				return

		response_queue: queue.Queue[str] = queue.Queue()
		sentinel = "__CY_LLM_STREAM_END__"
		error_holder: List[Exception] = []
		collected_chunks: List[str] = []  # 用于收集响应以便缓存
		
		# 进度队列：用于模型加载过程中的进度信息
		progress_queue: queue.Queue[str] = queue.Queue()
		progress_sentinel = "__CY_LLM_PROGRESS_END__"
		
		# 检查模型是否已加载
		model_loaded = self._memory.get_loaded_model(model_id) is not None
		
		# 如果模型未加载，先发送加载开始消息
		if not model_loaded:
			progress_queue.put("__CY_LLM_LOADING_START__")
		
		# 进度回调函数：将加载进度信息放入进度队列
		def progress_callback(message: str) -> None:
			progress_queue.put(f"__CY_LLM_LOADING__{message}__")

		# 工作函数：由调度器在后台线程调用，负责实际的推理并把结果放入队列
		def _task() -> None:
			start = time.time()
			self._telemetry.track_request_start()
			total_tokens = 0
			try:
				# 确保模型已加载（可能会触发加载）
				engine = self.ensure_model(
					model_id,
					model_path=model_path,
					adapter_path=adapter_path,
					engine_type=engine_type,
					engine_kwargs=engine_kwargs,
					progress_callback=progress_callback if not model_loaded else None,
				)
				# 模型加载完成，发送加载结束消息
				if not model_loaded:
					progress_queue.put(progress_sentinel)
				# 调用引擎的 infer，流式读取生成数据并入队
				for chunk in engine.infer(prompt, **gen_kwargs):
					response_queue.put(str(chunk))
					total_tokens += 1
					if enable_prompt_cache:
						collected_chunks.append(str(chunk))
				# 成功完成一次请求
				self._telemetry.track_request_end(time.time() - start, success=True)
				self._telemetry.track_token_generated(total_tokens)
				response_queue.put(sentinel)
			except Exception as exc:  # noqa: BLE001
				# 发生异常时记录并通知外层处理
				self._telemetry.track_request_end(time.time() - start, success=False)
				error_holder.append(exc)
				if not model_loaded:
					progress_queue.put(progress_sentinel)
				response_queue.put(sentinel)
			finally:
				# 标记该模型最近访问时间，防止被误回收
				self._memory.access_model(model_id)

		try:
			self._scheduler.submit(_task, priority=priority)
		except SchedulerBusy as exc:  # noqa: PERF203
			raise RuntimeError("Server is overloaded, please retry later.") from exc

		# 先处理加载进度信息
		progress_done = model_loaded  # 如果模型已加载，跳过进度处理
		if not model_loaded:
			yield "[模型加载] 开始加载模型，这可能需要几分钟，请耐心等待...\n"
		
		while not progress_done:
			try:
				# 使用超时避免阻塞
				progress_item = progress_queue.get(timeout=0.1)
				if progress_item == progress_sentinel:
					progress_done = True
				elif progress_item.startswith("__CY_LLM_LOADING__"):
					# 提取进度消息并输出
					message = progress_item.replace("__CY_LLM_LOADING__", "").rstrip("__")
					yield f"[模型加载] {message}\n"
			except queue.Empty:
				# 检查是否有错误
				if error_holder:
					break
				# 继续等待进度信息
				pass
			# 检查是否有错误
			if error_holder:
				# 确保处理完所有进度消息
				while True:
					try:
						progress_item = progress_queue.get_nowait()
						if progress_item == progress_sentinel:
							progress_done = True
							break
						elif progress_item.startswith("__CY_LLM_LOADING__"):
							message = progress_item.replace("__CY_LLM_LOADING__", "").rstrip("__")
							yield f"[模型加载] {message}\n"
					except queue.Empty:
						break
				break

		# 然后处理推理结果
		while True:
			item: str = response_queue.get()
			if item == sentinel:
				break
			yield item

		if error_holder:
			raise RuntimeError("Inference failed") from error_holder[0]

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
		response_queue: queue.Queue[str] = queue.Queue()
		sentinel = "__CY_LLM_STREAM_END__"
		error_holder: List[Exception] = []
		
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
				error_holder.append(e)
			finally:
				response_queue.put(sentinel)
		
		# 在线程池中运行同步生成
		loop.run_in_executor(None, _sync_generate)
		
		# 异步轮询队列
		while True:
			# 非阻塞获取，配合 sleep 让出控制权
			try:
				item: str = response_queue.get_nowait()
				if item == sentinel:
					break
				yield item
			except queue.Empty:
				await asyncio.sleep(0.001)  # 1ms 轮询间隔

		if error_holder:
			raise RuntimeError("Inference failed") from error_holder[0]

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
