"""
task_scheduler.py
# [调度] 轻量任务队列与请求排队，配合背压与优先级
# 说明：控制并发推理任务数量，支持优先级与队列回压策略。
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Callable, Tuple


class SchedulerBusy(RuntimeError):
	"""Raised when the queue is full and new tasks cannot be accepted."""

	# 说明：当调度队列已满时抛出该异常，用于上层进行背压 / 返回 503 相应。


@dataclass(order=True)
class ScheduledTask:
	priority: int
	created_at: float
	sequence: int = field(compare=False)
	func: Callable[..., Any] = field(compare=False)
	args: Tuple[Any, ...] = field(compare=False)
	kwargs: dict = field(compare=False)
	future: Future = field(compare=False)
	task_id: str = field(default_factory=lambda: uuid.uuid4().hex, compare=False)


class TaskScheduler:
	"""Executes callables on worker threads with simple QoS semantics."""

	def __init__(self, max_workers: int = 2, queue_size: int = 128) -> None:
		self._queue: "queue.PriorityQueue[ScheduledTask]" = queue.PriorityQueue(queue_size)
		self._max_workers = max_workers
		self._sequence = 0
		self._workers = []
		self._stop = threading.Event()
		self._lock = threading.Lock()

		for _ in range(max_workers):
			worker = threading.Thread(target=self._worker_loop, daemon=True)
			worker.start()
			self._workers.append(worker)

	def submit(self, func: Callable[..., Any], *args: Any, priority: int = 0, **kwargs: Any) -> Future:
		# 如果已停止调度器，拒绝提交
		if self._stop.is_set():
			raise RuntimeError("Scheduler already shutdown.")

		# 创建 Future 以便调用者能等待或查询执行结果
		future: Future = Future()
		with self._lock:
			self._sequence += 1
			seq = self._sequence

		task = ScheduledTask(priority, time.time(), seq, func, args, kwargs, future)

		# 非阻塞地把任务放入队列，队列已满则触发背压
		# 尝试将任务放入队列：为了在高并发短暂峰值下更稳定，使用短超时的阻塞 put。
		# 如果在超时时间内仍无法入队，则视为队列已饱和，触发背压异常。
		try:
			self._queue.put(task, block=True, timeout=0.2)
		except queue.Full as exc:  # noqa: PERF203
			raise SchedulerBusy("Scheduler queue is full.") from exc

		return future

	def _worker_loop(self) -> None:
		while not self._stop.is_set():
			try:
				task = self._queue.get(timeout=0.1)
			except queue.Empty:
				continue

			# 执行任务并将结果或异常写入 Future
			if task.future.set_running_or_notify_cancel():
				try:
					result = task.func(*task.args, **task.kwargs)
					task.future.set_result(result)
				except Exception as exc:  # noqa: BLE001
					task.future.set_exception(exc)

			self._queue.task_done()

		# Drain remaining tasks with cancellation to avoid dangling futures.
		while not self._queue.empty():
			task = self._queue.get()
			if task.future.set_running_or_notify_cancel():
				task.future.set_exception(RuntimeError("Scheduler stopped before execution."))
			self._queue.task_done()

	def shutdown(self, wait: bool = True) -> None:
		self._stop.set()
		if wait:
			for worker in self._workers:
				worker.join(timeout=1)
