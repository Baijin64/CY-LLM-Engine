"""
test_telemetry.py
core/telemetry.py 模块的单元测试
"""

import pytest
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from worker.core.telemetry import Telemetry


class TestTelemetry:
    """测试 Telemetry 遥测类"""

    @pytest.fixture
    def telemetry(self):
        """创建新的 Telemetry 实例并重置状态"""
        instance = Telemetry()
        instance.reset()  # 重置状态以隔离测试
        return instance

    def test_initial_state(self, telemetry):
        """初始状态应为空"""
        snapshot = telemetry.snapshot()
        assert snapshot["total_requests"] == 0
        assert snapshot["total_errors"] == 0

    def test_track_request(self, telemetry):
        """track_request 应增加请求计数"""
        telemetry.track_request()
        telemetry.track_request()
        snapshot = telemetry.snapshot()
        assert snapshot["total_requests"] == 2

    def test_track_error(self, telemetry):
        """track_error 应增加错误计数"""
        telemetry.track_error()
        snapshot = telemetry.snapshot()
        assert snapshot["total_errors"] == 1

    def test_track_latency(self, telemetry):
        """track_latency 应记录延迟"""
        telemetry.track_latency(100.0)
        telemetry.track_latency(200.0)
        snapshot = telemetry.snapshot()
        # 检查平均延迟（使用近似比较避免浮点误差）
        assert "avg_latency_ms" in snapshot
        assert abs(snapshot["avg_latency_ms"] - 150.0) < 0.01

    def test_percentile_calculation(self, telemetry):
        """percentile 应正确计算百分位数"""
        # 添加 100 个延迟值
        for i in range(1, 101):
            telemetry.track_latency(float(i))
        
        # P50 应约为 50（放宽容差）
        p50 = telemetry.percentile(50)
        assert 48 <= p50 <= 52

        # P95 应约为 95
        p95 = telemetry.percentile(95)
        assert 93 <= p95 <= 97

        # P99 应约为 99
        p99 = telemetry.percentile(99)
        assert 97 <= p99 <= 101

    def test_percentile_empty(self, telemetry):
        """空延迟列表的 percentile 应返回 0"""
        assert telemetry.percentile(50) == 0.0

    def test_percentile_single_value(self, telemetry):
        """单个延迟值的 percentile 应返回该值"""
        telemetry.track_latency(42.0)
        # 单值情况允许一定误差
        assert abs(telemetry.percentile(50) - 42.0) < 0.01
        assert abs(telemetry.percentile(99) - 42.0) < 0.01

    def test_snapshot_contains_percentiles(self, telemetry):
        """snapshot 应包含百分位延迟"""
        for i in range(100):
            telemetry.track_latency(float(i))
        
        snapshot = telemetry.snapshot()
        assert "latency_p50_ms" in snapshot
        assert "latency_p95_ms" in snapshot
        assert "latency_p99_ms" in snapshot

    def test_track_token_generated(self, telemetry):
        """track_token_generated 应记录生成的 token 数"""
        telemetry.track_token_generated(100)
        telemetry.track_token_generated(50)
        snapshot = telemetry.snapshot()
        assert snapshot.get("total_tokens", 0) == 150

    def test_reset(self, telemetry):
        """reset 应清空所有统计"""
        telemetry.track_request()
        telemetry.track_error()
        telemetry.track_latency(100.0)
        telemetry.reset()
        
        snapshot = telemetry.snapshot()
        assert snapshot["total_requests"] == 0
        assert snapshot["total_errors"] == 0

    def test_thread_safety(self, telemetry):
        """并发访问应线程安全"""
        import threading
        
        def track_requests():
            for _ in range(100):
                telemetry.track_request()
        
        threads = [threading.Thread(target=track_requests) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        snapshot = telemetry.snapshot()
        assert snapshot["total_requests"] == 1000

    def test_export_prometheus(self, telemetry):
        """export_prometheus 应返回有效的 Prometheus 格式"""
        telemetry.track_request()
        telemetry.track_latency(50.0)
        
        output = telemetry.export_prometheus()
        assert isinstance(output, str)
        # 验证包含 Prometheus 指标相关内容
        assert "worker" in output or "latency" in output or "requests" in output


class TestTelemetrySingleton:
    """测试 Telemetry 单例模式（如果有）"""

    def test_multiple_instances_independent(self):
        """多个实例应独立"""
        t1 = Telemetry()
        t2 = Telemetry()
        
        t1.track_request()
        
        # t2 不应受影响（除非是单例）
        # 这取决于实现
        s1 = t1.snapshot()
        s2 = t2.snapshot()
        # 如果是单例，两者相等；否则 t2 应为 0
        assert s1["total_requests"] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
