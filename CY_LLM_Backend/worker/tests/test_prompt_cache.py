"""
test_prompt_cache.py
cache/prompt_cache.py 模块的单元测试
"""

import pytest
import sys
import os
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from worker.cache.prompt_cache import PromptCache, CacheEntry, CacheStats, get_prompt_cache


class TestPromptCache:
    """测试 Prompt 缓存"""

    @pytest.fixture
    def cache(self):
        """创建缓存实例"""
        return PromptCache(max_size=100, ttl_seconds=3600, enabled=True)

    def test_set_and_get(self, cache):
        """set 和 get 应正确工作"""
        cache.set("Hello", "model-1", "World")
        result = cache.get("Hello", "model-1")
        assert result == "World"

    def test_cache_miss(self, cache):
        """缓存未命中应返回 None"""
        result = cache.get("nonexistent", "model-1")
        assert result is None

    def test_different_models_isolated(self, cache):
        """不同模型的缓存应隔离"""
        cache.set("Hello", "model-1", "Result1")
        cache.set("Hello", "model-2", "Result2")
        
        assert cache.get("Hello", "model-1") == "Result1"
        assert cache.get("Hello", "model-2") == "Result2"

    def test_different_params_isolated(self, cache):
        """不同参数的缓存应隔离"""
        cache.set("Hello", "model-1", "Result1", temperature=0.5)
        cache.set("Hello", "model-1", "Result2", temperature=0.9)
        
        assert cache.get("Hello", "model-1", temperature=0.5) == "Result1"
        assert cache.get("Hello", "model-1", temperature=0.9) == "Result2"

    def test_cache_expiry(self, cache):
        """过期条目应返回 None"""
        # 使用短 TTL
        short_cache = PromptCache(max_size=100, ttl_seconds=1, enabled=True)
        short_cache.set("Hello", "model-1", "World")
        
        # 等待过期
        time.sleep(1.5)
        
        result = short_cache.get("Hello", "model-1")
        assert result is None

    def test_lru_eviction(self):
        """超出容量应驱逐最旧条目"""
        small_cache = PromptCache(max_size=3, ttl_seconds=3600, enabled=True)
        
        small_cache.set("p1", "m", "r1")
        small_cache.set("p2", "m", "r2")
        small_cache.set("p3", "m", "r3")
        
        # 访问 p1 使其成为最近使用
        small_cache.get("p1", "m")
        
        # 添加新条目，应驱逐 p2（最旧未访问）
        small_cache.set("p4", "m", "r4")
        
        assert small_cache.get("p1", "m") == "r1"  # 应该还在
        assert small_cache.get("p4", "m") == "r4"  # 新添加的

    def test_invalidate(self, cache):
        """invalidate 应移除指定缓存"""
        cache.set("Hello", "model-1", "World")
        result = cache.invalidate("Hello", "model-1")
        
        assert result is True
        assert cache.get("Hello", "model-1") is None

    def test_invalidate_nonexistent(self, cache):
        """invalidate 不存在的条目应返回 False"""
        result = cache.invalidate("nonexistent", "model-1")
        assert result is False

    def test_clear(self, cache):
        """clear 应清空所有缓存"""
        cache.set("p1", "m", "r1")
        cache.set("p2", "m", "r2")
        
        count = cache.clear()
        
        assert count == 2
        assert cache.get("p1", "m") is None
        assert cache.get("p2", "m") is None

    def test_cleanup_expired(self, cache):
        """cleanup_expired 应清理过期条目"""
        # 设置短 TTL 的条目
        cache.set("expire", "m", "result", ttl=1)
        cache.set("keep", "m", "result", ttl=3600)
        
        time.sleep(1.5)
        
        count = cache.cleanup_expired()
        
        assert count >= 1
        assert cache.get("expire", "m") is None
        assert cache.get("keep", "m") == "result"

    def test_stats_tracking(self, cache):
        """应正确跟踪统计信息"""
        cache.set("Hello", "model-1", "World")
        
        # 命中
        cache.get("Hello", "model-1")
        cache.get("Hello", "model-1")
        
        # 未命中
        cache.get("missing", "model-1")
        
        stats = cache.get_stats()
        
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] > 0

    def test_disabled_cache(self):
        """禁用的缓存应不存储任何内容"""
        disabled_cache = PromptCache(max_size=100, ttl_seconds=3600, enabled=False)
        
        disabled_cache.set("Hello", "model-1", "World")
        result = disabled_cache.get("Hello", "model-1")
        
        assert result is None

    def test_set_enabled(self, cache):
        """set_enabled 应控制缓存启用状态"""
        cache.set("Hello", "model-1", "World")
        
        cache.set_enabled(False)
        assert cache.get("Hello", "model-1") is None
        
        cache.set_enabled(True)
        cache.set("Hello2", "model-1", "World2")
        assert cache.get("Hello2", "model-1") == "World2"

    def test_thread_safety(self, cache):
        """并发访问应线程安全"""
        errors = []
        
        def writer():
            try:
                for i in range(100):
                    cache.set(f"prompt-{i}", "model", f"result-{i}")
            except Exception as e:
                errors.append(e)
        
        def reader():
            try:
                for i in range(100):
                    cache.get(f"prompt-{i}", "model")
            except Exception as e:
                errors.append(e)
        
        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=writer),
            threading.Thread(target=reader),
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0


class TestCacheEntry:
    """测试 CacheEntry 数据类"""

    def test_creation(self):
        """应正确创建条目"""
        now = time.time()
        entry = CacheEntry(
            value="test",
            created_at=now,
            expires_at=now + 3600,
        )
        
        assert entry.value == "test"
        assert entry.hit_count == 0

    def test_hit_count(self):
        """hit_count 应可更新"""
        entry = CacheEntry(
            value="test",
            created_at=time.time(),
            expires_at=time.time() + 3600,
        )
        
        entry.hit_count += 1
        assert entry.hit_count == 1


class TestCacheStats:
    """测试 CacheStats 数据类"""

    def test_hit_rate_calculation(self):
        """hit_rate 应正确计算"""
        stats = CacheStats(hits=80, misses=20)
        assert stats.hit_rate == 0.8

    def test_hit_rate_zero_total(self):
        """零总数时 hit_rate 应为 0"""
        stats = CacheStats(hits=0, misses=0)
        assert stats.hit_rate == 0.0


class TestGetPromptCache:
    """测试全局缓存获取函数"""

    def test_returns_cache(self):
        """应返回 PromptCache 实例"""
        cache = get_prompt_cache()
        assert isinstance(cache, PromptCache)

    def test_singleton_behavior(self):
        """应返回相同实例（单例）"""
        cache1 = get_prompt_cache()
        cache2 = get_prompt_cache()
        assert cache1 is cache2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
