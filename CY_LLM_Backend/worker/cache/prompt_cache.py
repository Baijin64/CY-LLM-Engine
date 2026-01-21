"""
cache/prompt_cache.py
[缓存] Prompt 缓存机制，用于减少重复计算

特点：
  - 支持内存缓存（LRU）
  - 支持 Redis 分布式缓存（可选）
  - 自动过期和清理
  - 缓存命中统计
  - 使用 xxhash 高性能哈希（可选回退到 SHA256）

使用示例：
    >>> cache = PromptCache(max_size=1000, ttl_seconds=3600)
    >>> cache.set("Hello", "model-1", "World")
    >>> cache.get("Hello", "model-1")  # -> "World"
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Callable, List

LOGGER = logging.getLogger("cy_llm.worker.cache.prompt")

# ============================================================================
# 高性能哈希：优先使用 xxhash，回退到 hashlib
# ============================================================================
_hash_func: Callable[[bytes], str]

try:
    import xxhash
    
    def _xxhash_digest(data: bytes) -> str:
        """使用 xxhash（~10x faster than SHA256）"""
        return xxhash.xxh64(data).hexdigest()
    
    _hash_func = _xxhash_digest
    LOGGER.info("使用 xxhash 作为缓存键哈希算法")
    
except ImportError:
    import hashlib
    
    def _sha256_digest(data: bytes) -> str:
        """回退到 SHA256"""
        return hashlib.sha256(data).hexdigest()
    
    _hash_func = _sha256_digest
    LOGGER.warning("xxhash 未安装，回退到 SHA256（建议 pip install xxhash）")


@dataclass
class CacheEntry:
    """缓存条目"""
    value: str
    created_at: float
    expires_at: float
    hit_count: int = 0
    last_access_at: float = field(default_factory=time.time)


@dataclass
class CacheStats:
    """缓存统计"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class PromptCache:
    """
    Prompt 缓存（内存 LRU 实现）
    
    用于缓存相同 prompt + model 的推理结果，避免重复计算。
    适用于：
      - 相同问题的多次查询
      - 系统提示词的缓存
      - 测试/调试场景
      
    注意：
      - 缓存仅适用于确定性输出（temperature=0）或可接受近似结果的场景
      - 对于创意生成任务，不建议使用缓存
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: int = 3600,
        enabled: bool = True,
        auto_cleanup_interval: int = 300,
    ):
        """
        初始化缓存。
        
        Args:
            max_size: 最大缓存条目数
            ttl_seconds: 缓存过期时间（秒）
            enabled: 是否启用缓存
            auto_cleanup_interval: 自动清理间隔（秒），0 表示禁用
        """
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._enabled = enabled
        self._stats = CacheStats()
        self._auto_cleanup_interval = auto_cleanup_interval
        self._cleanup_stop = threading.Event()
        self._cleanup_thread: Optional[threading.Thread] = None
        
        LOGGER.info(
            "PromptCache 初始化: max_size=%d, ttl=%ds, enabled=%s, auto_cleanup=%ds",
            max_size, ttl_seconds, enabled, auto_cleanup_interval,
        )
        
        if auto_cleanup_interval > 0 and enabled:
            self._start_cleanup_thread()


    def _make_key(self, prompt: str, model_id: str, **params) -> str:
        """
        生成缓存键。
        
        使用 prompt + model_id + 关键参数的哈希值作为键。
        """
        # 只包含影响输出的关键参数
        key_params = {
            "temperature": params.get("temperature", 0.7),
            "top_p": params.get("top_p", 0.9),
            "max_tokens": params.get("max_tokens", 512),
        }
        
        key_str = f"{model_id}:{prompt}:{sorted(key_params.items())}"
        return _hash_func(key_str.encode())

    def get(
        self,
        prompt: str,
        model_id: str,
        **params
    ) -> Optional[str]:
        """
        获取缓存的结果。
        
        Args:
            prompt: 输入提示
            model_id: 模型 ID
            **params: 生成参数
            
        Returns:
            缓存的结果，如果未命中返回 None
        """
        if not self._enabled:
            return None
            
        key = self._make_key(prompt, model_id, **params)
        
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._stats.misses += 1
                return None
                
            # 检查是否过期
            if time.time() > entry.expires_at:
                del self._cache[key]
                self._stats.misses += 1
                self._stats.evictions += 1
                return None
            
            # 命中，更新统计和 LRU 顺序
            entry.hit_count += 1
            entry.last_access_at = time.time()
            self._stats.hits += 1
            self._cache.move_to_end(key)
            
            LOGGER.debug("缓存命中: key=%s..., hits=%d", key[:16], entry.hit_count)
            return entry.value

    def set(
        self,
        prompt: str,
        model_id: str,
        result: str,
        ttl: Optional[int] = None,
        **params
    ) -> None:
        """
        设置缓存。
        
        Args:
            prompt: 输入提示
            model_id: 模型 ID
            result: 推理结果
            ttl: 过期时间（秒），None 使用默认值
            **params: 生成参数
        """
        if not self._enabled:
            return
            
        key = self._make_key(prompt, model_id, **params)
        ttl = ttl or self._ttl_seconds
        now = time.time()
        
        entry = CacheEntry(
            value=result,
            created_at=now,
            expires_at=now + ttl,
        )
        
        with self._lock:
            # 如果已存在，更新并移到末尾
            if key in self._cache:
                del self._cache[key]
            
            # 检查容量，基于 LRU 驱逐最旧的
            while len(self._cache) >= self._max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._stats.evictions += 1
                LOGGER.debug("LRU 驱逐: key=%s...", oldest_key[:16])
            
            self._cache[key] = entry
            self._stats.size = len(self._cache)
            
        LOGGER.debug("缓存设置: key=%s..., ttl=%ds", key[:16], ttl)

    def invalidate(self, prompt: str, model_id: str, **params) -> bool:
        """
        使指定缓存失效。
        
        Returns:
            是否成功删除
        """
        key = self._make_key(prompt, model_id, **params)
        
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats.size = len(self._cache)
                return True
        return False

    def invalidate_model(self, model_id: str) -> int:
        """
        使指定模型的所有缓存失效。
        
        Returns:
            删除的条目数
        """
        # 由于 key 是哈希值，无法直接按 model_id 过滤
        # 这里清空所有缓存（简化实现）
        return self.clear()

    def clear(self) -> int:
        """
        清空所有缓存。
        
        Returns:
            清除的条目数
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._stats.size = 0
            self._stats.evictions += count
        LOGGER.info("缓存已清空: %d 条", count)
        return count

    def cleanup_expired(self) -> int:
        """
        清理过期条目。
        
        Returns:
            清理的条目数
        """
        now = time.time()
        expired_keys = []
        
        with self._lock:
            for key, entry in self._cache.items():
                if now > entry.expires_at:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._cache[key]
            
            self._stats.size = len(self._cache)
            self._stats.evictions += len(expired_keys)
        
        if expired_keys:
            LOGGER.debug("清理过期缓存: %d 条", len(expired_keys))
        
        return len(expired_keys)
    
    def _start_cleanup_thread(self) -> None:
        """启动后台清理线程"""
        def _cleanup_loop():
            while not self._cleanup_stop.wait(timeout=self._auto_cleanup_interval):
                try:
                    cleaned = self.cleanup_expired()
                    if cleaned > 0:
                        LOGGER.info("自动清理过期缓存: %d 条", cleaned)
                except Exception as e:
                    LOGGER.error("缓存清理线程异常: %s", e)
        
        self._cleanup_thread = threading.Thread(target=_cleanup_loop, daemon=True, name="PromptCache-Cleanup")
        self._cleanup_thread.start()
        LOGGER.info("缓存自动清理线程已启动，间隔=%ds", self._auto_cleanup_interval)
    
    def _stop_cleanup_thread(self) -> None:
        """停止后台清理线程"""
        if self._cleanup_thread:
            self._cleanup_stop.set()
            self._cleanup_thread.join(timeout=5)
            LOGGER.info("缓存自动清理线程已停止")
    
    def shutdown(self) -> None:
        """关闭缓存，释放资源"""
        self._stop_cleanup_thread()
        self.clear()
        LOGGER.info("PromptCache 已关闭")


    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息。"""
        with self._lock:
            return {
                "enabled": self._enabled,
                "max_size": self._max_size,
                "current_size": self._stats.size,
                "ttl_seconds": self._ttl_seconds,
                "hits": self._stats.hits,
                "misses": self._stats.misses,
                "evictions": self._stats.evictions,
                "hit_rate": round(self._stats.hit_rate, 4),
            }

    def set_enabled(self, enabled: bool) -> None:
        """启用或禁用缓存。"""
        was_enabled = self._enabled
        self._enabled = enabled
        
        if enabled and not was_enabled:
            if self._auto_cleanup_interval > 0:
                self._start_cleanup_thread()
        elif not enabled and was_enabled:
            self._stop_cleanup_thread()
        
        LOGGER.info("缓存%s", "已启用" if enabled else "已禁用")


class RedisPromptCache:
    """
    基于 Redis 的分布式 Prompt 缓存。
    
    适用于多 Worker 部署场景，支持跨进程缓存共享。
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        prefix: str = "cy_llm:prompt:",
        ttl_seconds: int = 3600,
        enabled: bool = True,
    ):
        """
        初始化 Redis 缓存。
        
        Args:
            redis_url: Redis 连接 URL
            prefix: 缓存键前缀
            ttl_seconds: 默认过期时间
            enabled: 是否启用
        """
        self._prefix = prefix
        self._ttl_seconds = ttl_seconds
        self._enabled = enabled
        self._redis = None
        self._stats = CacheStats()
        
        if enabled:
            try:
                import redis
                self._redis = redis.from_url(redis_url)
                self._redis.ping()
                LOGGER.info("Redis 缓存连接成功: %s", redis_url)
            except Exception as e:
                LOGGER.warning("Redis 缓存连接失败，将禁用: %s", e)
                self._enabled = False

    def _make_key(self, prompt: str, model_id: str, **params) -> str:
        """生成 Redis 键。"""
        key_params = {
            "temperature": params.get("temperature", 0.7),
            "top_p": params.get("top_p", 0.9),
            "max_tokens": params.get("max_tokens", 512),
        }
        key_str = f"{model_id}:{prompt}:{sorted(key_params.items())}"
        hash_val = _hash_func(key_str.encode())
        return f"{self._prefix}{model_id}:{hash_val}"

    def get(self, prompt: str, model_id: str, **params) -> Optional[str]:
        """获取缓存。"""
        if not self._enabled or self._redis is None:
            return None
            
        key = self._make_key(prompt, model_id, **params)
        
        try:
            value = self._redis.get(key)
            if value:
                self._stats.hits += 1
                return value.decode("utf-8")
            else:
                self._stats.misses += 1
                return None
        except Exception as e:
            LOGGER.warning("Redis 读取失败: %s", e)
            self._stats.misses += 1
            return None

    def set(
        self,
        prompt: str,
        model_id: str,
        result: str,
        ttl: Optional[int] = None,
        **params
    ) -> None:
        """设置缓存。"""
        if not self._enabled or self._redis is None:
            return
            
        key = self._make_key(prompt, model_id, **params)
        ttl = ttl or self._ttl_seconds
        
        try:
            self._redis.setex(key, ttl, result.encode("utf-8"))
        except Exception as e:
            LOGGER.warning("Redis 写入失败: %s", e)

    def invalidate(self, prompt: str, model_id: str, **params) -> bool:
        """使缓存失效。"""
        if not self._enabled or self._redis is None:
            return False
            
        key = self._make_key(prompt, model_id, **params)
        
        try:
            return self._redis.delete(key) > 0
        except Exception as e:
            LOGGER.warning("Redis 删除失败: %s", e)
            return False

    def invalidate_model(self, model_id: str) -> int:
        """使指定模型的所有缓存失效。"""
        if not self._enabled or self._redis is None:
            return 0
            
        pattern = f"{self._prefix}{model_id}:*"
        
        try:
            keys = list(self._redis.scan_iter(match=pattern, count=1000))
            if keys:
                return self._redis.delete(*keys)
            return 0
        except Exception as e:
            LOGGER.warning("Redis 批量删除失败: %s", e)
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息。"""
        return {
            "enabled": self._enabled,
            "backend": "redis",
            "prefix": self._prefix,
            "ttl_seconds": self._ttl_seconds,
            "hits": self._stats.hits,
            "misses": self._stats.misses,
            "hit_rate": round(self._stats.hit_rate, 4),
        }


# 全局缓存实例
_prompt_cache: Optional[PromptCache] = None


def get_prompt_cache(
    max_size: int = 1000,
    ttl_seconds: int = 3600,
    enabled: bool = True,
) -> PromptCache:
    """获取全局 Prompt 缓存实例（单例模式）。"""
    global _prompt_cache
    if _prompt_cache is None:
        _prompt_cache = PromptCache(
            max_size=max_size,
            ttl_seconds=ttl_seconds,
            enabled=enabled,
        )
    return _prompt_cache
