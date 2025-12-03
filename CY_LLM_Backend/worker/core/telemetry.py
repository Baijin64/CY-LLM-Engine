"""
telemetry.py
# [监控] 导出指标与日志（GPU 利用率、内存使用、请求延迟）
# 说明：用于 Prometheus/Logging 集成，帮助 Gateway 与运维进行故障分析。
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, Dict, List


def _percentile(sorted_data: List[float], p: float) -> float:
    """计算百分位数"""
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * p / 100.0
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


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
		self._cache_hits = 0
		self._cache_misses = 0
		self._latencies: Deque[float] = deque(maxlen=512)
		self._latest_gpu = {"used_mb": 0.0, "total_mb": 0.0, "ts": 0.0}
		self._initialized = True

	def track_request_start(self) -> None:
		# 标记请求开始，用于统计并发/排队信息
		with self._lock:
			self._inflight += 1
			self._cache_misses += 1  # 进入推理流程即为缓存未命中

	# 兼容旧 API
	def track_request(self) -> None:
		"""兼容方法：标记一次请求开始（用于旧测试/调用）。"""
		self.track_request_start()

	def track_cache_hit(self) -> None:
		# 标记缓存命中
		with self._lock:
			self._cache_hits += 1

	def track_request_end(self, latency: float, *, success: bool) -> None:
		# 请求结束时记录延迟与成功/失败计数
		with self._lock:
			self._inflight = max(0, self._inflight - 1)
			if success:
				self._success += 1
			else:
				self._failed += 1
			self._latencies.append(latency)

	def track_error(self) -> None:
		"""兼容旧 API：记录一次错误"""
		with self._lock:
			self._failed += 1

	def track_latency(self, latency_ms: float) -> None:
		"""兼容旧 API：记录延迟（毫秒）"""
		# 内部以秒为单位存储
		with self._lock:
			self._latencies.append(latency_ms / 1000.0)

	def record_gpu_usage(self, used_mb: float, total_mb: float) -> None:
		# 手动或定时调用以记录最新的 GPU 使用情况
		with self._lock:
			self._latest_gpu = {"used_mb": used_mb, "total_mb": total_mb, "ts": time.time()}

	def snapshot(self) -> Dict[str, float]:
		# 返回一份指标快照，便于导出或报警
		with self._lock:
			# 基础统计
			avg_latency = sum(self._latencies) / len(self._latencies) if self._latencies else 0.0
			total_requests = self._cache_hits + self._cache_misses
			cache_hit_rate = (self._cache_hits / total_requests * 100.0) if total_requests > 0 else 0.0
			
			# 百分位延迟统计
			sorted_latencies = sorted(self._latencies) if self._latencies else []
			p50 = _percentile(sorted_latencies, 50)
			p95 = _percentile(sorted_latencies, 95)
			p99 = _percentile(sorted_latencies, 99)
			
			return {
				"inflight": float(self._inflight),
				"success": float(self._success),
				"failed": float(self._failed),
				"cache_hits": float(self._cache_hits),
				"cache_misses": float(self._cache_misses),
				"cache_hit_rate_percent": cache_hit_rate,
				"avg_latency_ms": avg_latency * 1000.0,
				"p50_latency_ms": p50 * 1000.0,
				"p95_latency_ms": p95 * 1000.0,
				"p99_latency_ms": p99 * 1000.0,
				"gpu_used_mb": self._latest_gpu["used_mb"],
				"gpu_total_mb": self._latest_gpu["total_mb"],
				# 兼容旧测试期望的字段
				"total_requests": int(total_requests),
				"total_errors": int(self._failed),
				"latency_p50_ms": p50 * 1000.0,
				"latency_p95_ms": p95 * 1000.0,
				"latency_p99_ms": p99 * 1000.0,
				"total_tokens": getattr(self, "_total_tokens", 0),
			}

	def percentile(self, p: float) -> float:
		"""兼容旧 API: 返回给定百分位的延迟（毫秒）"""
		with self._lock:
			sorted_latencies = sorted(self._latencies) if self._latencies else []
			return _percentile(sorted_latencies, p) * 1000.0

	def track_token_generated(self, tokens: int) -> None:
		"""兼容旧 API：计数生成的 token"""
		with self._lock:
			if not hasattr(self, "_total_tokens"):
				self._total_tokens = 0
			self._total_tokens += int(tokens)

	def reset(self) -> None:
		"""复位所有统计数据"""
		with self._lock:
			self._inflight = 0
			self._success = 0
			self._failed = 0
			self._cache_hits = 0
			self._cache_misses = 0
			self._latencies.clear()
			self._latest_gpu = {"used_mb": 0.0, "total_mb": 0.0, "ts": 0.0}

	def export_prometheus(self) -> str:
		# 把快照序列化为 Prometheus 文本数据格式（最小化实现）
		data = self.snapshot()
		lines = [
			"# HELP worker_inflight_requests Current number of inflight requests",
			"# TYPE worker_inflight_requests gauge",
			"worker_inflight_requests %.0f" % data["inflight"],
			"# HELP worker_success_total Total successful requests",
			"# TYPE worker_success_total counter",
			"worker_success_total %.0f" % data["success"],
			"# HELP worker_failed_total Total failed requests",
			"# TYPE worker_failed_total counter",
			"worker_failed_total %.0f" % data["failed"],
			"# HELP worker_cache_hits_total Cache hits",
			"# TYPE worker_cache_hits_total counter",
			"worker_cache_hits_total %.0f" % data["cache_hits"],
			"# HELP worker_cache_misses_total Cache misses",
			"# TYPE worker_cache_misses_total counter",
			"worker_cache_misses_total %.0f" % data["cache_misses"],
			"# HELP worker_cache_hit_rate_percent Cache hit rate",
			"# TYPE worker_cache_hit_rate_percent gauge",
			"worker_cache_hit_rate_percent %.2f" % data["cache_hit_rate_percent"],
			"# HELP worker_latency_ms Request latency in milliseconds",
			"# TYPE worker_latency_ms summary",
			'worker_latency_ms{quantile="0.5"} %.2f' % data["p50_latency_ms"],
			'worker_latency_ms{quantile="0.95"} %.2f' % data["p95_latency_ms"],
			'worker_latency_ms{quantile="0.99"} %.2f' % data["p99_latency_ms"],
			"worker_latency_ms_avg %.2f" % data["avg_latency_ms"],
			"# HELP worker_gpu_used_mb GPU memory used in MB",
			"# TYPE worker_gpu_used_mb gauge",
			"worker_gpu_used_mb %.2f" % data["gpu_used_mb"],
			"# HELP worker_gpu_total_mb GPU memory total in MB",
			"# TYPE worker_gpu_total_mb gauge",
			"worker_gpu_total_mb %.2f" % data["gpu_total_mb"],
		]
		return "\n".join(lines)
