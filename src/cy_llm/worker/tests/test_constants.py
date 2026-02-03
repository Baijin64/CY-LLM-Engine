"""
test_constants.py
constants.py 模块的单元测试
"""

import pytest
import sys
import os

# 添加 worker 目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from worker.constants import (
    GRPCDefaults,
    MemoryDefaults,
    CacheDefaults,
    TrainingDefaults,
    UnitConversions,
)


class TestGRPCDefaults:
    """测试 gRPC 默认配置常量"""

    def test_port_is_valid(self):
        """端口应在有效范围内"""
        assert 1 <= GRPCDefaults.PORT <= 65535
        assert GRPCDefaults.PORT == 50051

    def test_max_workers_positive(self):
        """最大 worker 数应为正整数"""
        assert GRPCDefaults.MAX_WORKERS > 0

    def test_max_message_size_reasonable(self):
        """消息大小应在合理范围"""
        # 100MB
        assert GRPCDefaults.MAX_MESSAGE_SIZE == 100 * 1024 * 1024

    def test_timeout_positive(self):
        """超时时间应为正数"""
        assert GRPCDefaults.TIMEOUT_SECONDS > 0


class TestMemoryDefaults:
    """测试内存管理默认配置"""

    def test_gpu_threshold_valid_range(self):
        """GPU 阈值应在 0-1 之间"""
        assert 0 < MemoryDefaults.GPU_THRESHOLD <= 1.0
        assert MemoryDefaults.GPU_THRESHOLD == 0.90

    def test_cleanup_interval_positive(self):
        """清理间隔应为正整数"""
        assert MemoryDefaults.CLEANUP_INTERVAL > 0

    def test_lock_cleanup_interval_positive(self):
        """锁清理间隔应为正整数"""
        assert MemoryDefaults.LOCK_CLEANUP_INTERVAL > 0


class TestCacheDefaults:
    """测试缓存默认配置"""

    def test_max_size_positive(self):
        """最大缓存大小应为正整数"""
        assert CacheDefaults.MAX_SIZE > 0
        assert CacheDefaults.MAX_SIZE == 1000

    def test_ttl_positive(self):
        """TTL 应为正整数"""
        assert CacheDefaults.TTL_SECONDS > 0
        assert CacheDefaults.TTL_SECONDS == 3600

    def test_enabled_is_bool(self):
        """enabled 应为布尔值"""
        assert isinstance(CacheDefaults.ENABLED, bool)


class TestTrainingDefaults:
    """测试训练默认配置"""

    def test_batch_size_positive(self):
        """批量大小应为正整数"""
        assert TrainingDefaults.BATCH_SIZE > 0

    def test_learning_rate_valid(self):
        """学习率应在合理范围"""
        assert 0 < TrainingDefaults.LEARNING_RATE < 1

    def test_epochs_positive(self):
        """训练轮数应为正整数"""
        assert TrainingDefaults.EPOCHS > 0

    def test_lora_rank_positive(self):
        """LoRA rank 应为正整数"""
        assert TrainingDefaults.LORA_RANK > 0

    def test_lora_alpha_positive(self):
        """LoRA alpha 应为正数"""
        assert TrainingDefaults.LORA_ALPHA > 0


class TestUnitConversions:
    """测试单位转换常量"""

    def test_bytes_conversions(self):
        """字节转换应正确"""
        assert UnitConversions.KB == 1024
        assert UnitConversions.MB == 1024 * 1024
        assert UnitConversions.GB == 1024 * 1024 * 1024

    def test_time_conversions(self):
        """时间转换应正确"""
        assert UnitConversions.MS_PER_SECOND == 1000
        assert UnitConversions.NS_PER_MS == 1_000_000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
