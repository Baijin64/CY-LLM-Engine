"""
telemetry.py
# [监控] 导出指标与日志（GPU 利用率、内存使用、请求延迟）
# 说明：用于 Prometheus/Logging 集成，帮助 Gateway 与运维进行故障分析。
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, Dict


class Telemetry:
	"""Thread-safe singleton for collecting worker health metrics."""

	_instance = None
	_instance_lock = threading.Lock()

	def __new__(cls):
		if cls._instance is None:
			with cls._instance_lock:
				if cls._instance is None:
					cls._instance = super().__new__(cls)
					cls._instance._initialized = False
		return cls._instance

	def __init__(self) -> None:
		if getattr(self, "_initialized", False):
			return

		self._lock = threading.Lock()
		self._inflight = 0
		self._success = 0
		self._failed = 0
		self._latencies: Deque[float] = deque(maxlen=512)
		self._latest_gpu = {"used_mb": 0.0, "total_mb": 0.0, "ts": 0.0}
		self._initialized = True

	def track_request_start(self) -> None:
		# 标记请求开始，用于统计并发/排队信息
		with self._lock:
			self._inflight += 1

	def track_request_end(self, latency: float, *, success: bool) -> None:
		# 请求结束时记录延迟与成功/失败计数
		with self._lock:
			self._inflight = max(0, self._inflight - 1)
			if success:
				self._success += 1
			else:
				self._failed += 1
			self._latencies.append(latency)

	def record_gpu_usage(self, used_mb: float, total_mb: float) -> None:
		# 手动或定时调用以记录最新的 GPU 使用情况
		with self._lock:
			self._latest_gpu = {"used_mb": used_mb, "total_mb": total_mb, "ts": time.time()}

	def snapshot(self) -> Dict[str, float]:
		# 返回一份指标快照，便于导出或报警
		with self._lock:
			avg_latency = sum(self._latencies) / len(self._latencies) if self._latencies else 0.0
			return {
				"inflight": float(self._inflight),
				"success": float(self._success),
				"failed": float(self._failed),
				"avg_latency_ms": avg_latency * 1000.0,
				"gpu_used_mb": self._latest_gpu["used_mb"],
				"gpu_total_mb": self._latest_gpu["total_mb"],
			}

	def export_prometheus(self) -> str:
		# 把快照序列化为 Prometheus 文本数据格式（最小化实现）
		data = self.snapshot()
		lines = [
			"worker_inflight_requests %.0f" % data["inflight"],
			"worker_success_total %.0f" % data["success"],
			"worker_failed_total %.0f" % data["failed"],
			"worker_avg_latency_ms %.2f" % data["avg_latency_ms"],
			"worker_gpu_used_mb %.2f" % data["gpu_used_mb"],
			"worker_gpu_total_mb %.2f" % data["gpu_total_mb"],
		]
		return "\n".join(lines)
