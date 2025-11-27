"""
stream_buffer.py
# [流控] 处理 token/帧流缓冲、背压与调试信息清理
# 说明：在 gRPC 双向流中充当中间缓冲层，避免直接阻塞推理逻辑。
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, Dict, Generic, Iterable, List, Optional, TypeVar


T = TypeVar("T")


class BufferClosed(RuntimeError):
	"""在对已关闭缓冲区进行读写时抛出。"""


class StreamBuffer(Generic[T]):

	def __init__(self, max_size: int = 512, warn_threshold: float = 0.8) -> None:
		if max_size <= 0:
			raise ValueError("max_size must be positive.")
		if not 0 < warn_threshold <= 1:
			raise ValueError("warn_threshold must be within (0, 1].")

		self._queue: Deque[T] = deque()
		self._max_size = max_size
		self._warn_threshold = warn_threshold
		self._cond = threading.Condition()
		self._closed = False
		self._pushed = 0
		self._popped = 0
		self._dropped = 0
		# 说明：StreamBuffer 用于解耦生产者(推理线程)与消费者(gRPC 发送线程)，
		# 在高并发场景下通过阻塞或丢弃策略实现背压。

	def push(self, item: T, *, block: bool = True, timeout: Optional[float] = None) -> bool:

		with self._cond:
			# 如果缓冲区已关闭，拒绝写入
			if self._closed:
				raise BufferClosed("Cannot push to a closed buffer.")

			start = time.monotonic()
			while len(self._queue) >= self._max_size:
				# 非阻塞模式下直接丢弃并计数，供上层采用回退逻辑
				if not block:
					self._dropped += 1
					return False
				remaining = None if timeout is None else timeout - (time.monotonic() - start)
				if remaining is not None and remaining <= 0:
					self._dropped += 1
					return False
				self._cond.wait(timeout=remaining)

			self._queue.append(item)
			self._pushed += 1
			self._cond.notify_all()
			return True

	def pop(self, *, block: bool = True, timeout: Optional[float] = None) -> T:
		"""Remove and return the next item, blocking if configured."""

		with self._cond:
			# 等待直到有可用元素或超时/已关闭
			start = time.monotonic()
			while not self._queue:
				if self._closed:
					# 已关闭且无数据，通知上层结束流
					raise BufferClosed("Buffer closed and drained.")
				if not block:
					# 非阻塞且为空时立即返回错误，调用方可捕获并处理
					raise BufferClosed("Buffer empty and non-blocking pop requested.")
				remaining = None if timeout is None else timeout - (time.monotonic() - start)
				if remaining is not None and remaining <= 0:
					raise BufferClosed("Timed out waiting for buffer items.")
				self._cond.wait(timeout=remaining)

			item = self._queue.popleft()
			self._popped += 1
			self._cond.notify_all()
			return item

	def close(self) -> None:

		with self._cond:
			self._closed = True
			self._cond.notify_all()
		# 说明：调用 close 后，所有等待的 pop 将收到 BufferClosed，生产者不能再 push

	def drain(self, max_items: Optional[int] = None) -> List[T]:

		with self._cond:
			count = len(self._queue) if max_items is None else min(len(self._queue), max_items)
			items = [self._queue.popleft() for _ in range(count)]
			self._popped += len(items)
			self._cond.notify_all()
			return items

	# 说明：drain 可用于在关闭前或调试时一次性清空缓冲区并获取残留项

	def __len__(self) -> int:  # pragma: no cover - trivial wrapper
		return len(self._queue)

	def stats(self) -> Dict[str, float]:

		with self._cond:
			usage = len(self._queue) / self._max_size
			return {
				"size": len(self._queue),
				"max_size": float(self._max_size),
				"usage": usage,
				"near_capacity": float(usage >= self._warn_threshold),
				"pushed": float(self._pushed),
				"popped": float(self._popped),
				"dropped": float(self._dropped),
				"closed": float(self._closed),
			}

	# 说明：stats 提供当前缓冲区指标，便于与 Telemetry 集成或触发报警

	def is_closed(self) -> bool:
		return self._closed

	def extend(self, items: Iterable[T], *, block: bool = True) -> int:
		"""Push multiple items, returning how many were accepted."""

		accepted = 0
		for item in items:
			if not self.push(item, block=block):
				break
			accepted += 1
		return accepted
