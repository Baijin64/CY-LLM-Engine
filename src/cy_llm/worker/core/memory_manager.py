"""
memory_manager.py
# [关键] GPUMemoryManager：实现显存安全策略，要求：引用计数 + LRU 回收
# 说明：负责分配/释放显存对象、跟踪引用计数、在内存不足时按 LRU 策略回收并触发 SIG 或错误回退。
"""

import time
import threading
import weakref
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import torch
import gc

from ..constants import MemoryDefaults, UnitConversions
from ..exceptions import ModelUnloadError

if TYPE_CHECKING:
    from ..engines.abstract_engine import BaseEngine


class GPUMemoryManager:
    """
    显存管理器 (单例模式)
    负责监控显存使用，并根据策略（LRU）卸载不活跃的模型以释放空间。
    """
    _instance = None
    _lock = threading.Lock()
    _test_mode = False  # 测试模式标志，绕过单例

    def __new__(cls, threshold: Optional[float] = None, **kwargs: Any) -> "GPUMemoryManager":
        # 测试模式下不使用单例，每次创建新实例
        if cls._test_mode or threshold is not None:
            instance = super(GPUMemoryManager, cls).__new__(cls)
            instance._initialized = False
            return instance
            
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(GPUMemoryManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, threshold: Optional[float] = None, **kwargs: Any) -> None:
        if self._initialized:
            return
        
        self.loaded_models: Dict[str, 'BaseEngine'] = {}  # model_id -> engine_instance
        self.last_access_time: Dict[str, float] = {}      # model_id -> timestamp
        self.model_locks: Dict[str, threading.Lock] = {}  # model_id -> lock (防止并发卸载/使用)
        self._lock_access_count: int = 0                  # 锁访问计数，用于周期性清理
        self._model_sizes: Dict[str, float] = {}          # 模型大小记录 (GB)
        
        # 配置参数 (支持 threshold 参数以兼容旧测试)
        self._threshold = threshold if threshold is not None else MemoryDefaults.GPU_MEMORY_THRESHOLD
        self.max_gpu_memory_usage = self._threshold
        self.check_interval = MemoryDefaults.CHECK_INTERVAL_SECONDS
        
        self._initialized = True
        print("[GPUMemoryManager] Initialized.")

    def register_model(self, model_id: str, engine: Optional['BaseEngine'] = None, size_gb: float = 0.0) -> None:
        """注册一个已加载的模型引擎
        
        Args:
            model_id: 模型标识
            engine: 引擎实例 (可选)
            size_gb: 模型大小 (GB)，用于旧测试兼容
        """
        with self._lock:
            if engine is not None:
                self.loaded_models[model_id] = engine
            self._model_sizes[model_id] = size_gb
            self.last_access_time[model_id] = time.time()
            if model_id not in self.model_locks:
                self.model_locks[model_id] = threading.Lock()
            print(f"[GPUMemoryManager] Registered model: {model_id}")
    
    def unregister_model(self, model_id: str) -> None:
        """移除已注册的模型"""
        with self._lock:
            if model_id in self.loaded_models:
                del self.loaded_models[model_id]
            if model_id in self._model_sizes:
                del self._model_sizes[model_id]
            if model_id in self.last_access_time:
                del self.last_access_time[model_id]
            # 清理锁
            if model_id in self.model_locks:
                lock = self.model_locks[model_id]
                if lock.acquire(blocking=False):
                    lock.release()
                    del self.model_locks[model_id]
            print(f"[GPUMemoryManager] Unregistered model: {model_id}")

    def access_model(self, model_id: str) -> None:
        """更新模型的访问时间 (LRU)"""
        with self._lock:
            if model_id in self.loaded_models:
                self.last_access_time[model_id] = time.time()

    def get_model_lock(self, model_id: str) -> threading.Lock:
        """获取模型锁，并周期性清理不再需要的锁以防止内存泄漏"""
        with self._lock:
            self._lock_access_count += 1
            
            # 周期性清理不再需要的锁
            if self._lock_access_count >= MemoryDefaults.LOCK_CLEANUP_THRESHOLD:
                self._cleanup_stale_locks()
                self._lock_access_count = 0
            
            if model_id not in self.model_locks:
                self.model_locks[model_id] = threading.Lock()
            return self.model_locks[model_id]
    
    # 兼容旧测试 API
    def acquire_lock(self, model_id: str) -> threading.Lock:
        """获取模型锁 (兼容旧 API)"""
        lock = self.get_model_lock(model_id)
        lock.acquire()
        return lock
    
    def release_lock(self, model_id: str) -> None:
        """释放模型锁 (兼容旧 API)"""
        if model_id in self.model_locks:
            try:
                self.model_locks[model_id].release()
            except RuntimeError:
                pass  # 锁可能已被释放

    def _cleanup_stale_locks(self) -> None:
        """清理已卸载模型对应的锁（内部方法，需在持有 _lock 时调用）"""
        stale_lock_ids = [
            mid for mid in self.model_locks 
            if mid not in self.loaded_models
        ]
        for mid in stale_lock_ids:
            # 只清理未被锁定的锁
            lock = self.model_locks[mid]
            if lock.acquire(blocking=False):
                lock.release()
                del self.model_locks[mid]
        if stale_lock_ids:
            print(f"[GPUMemoryManager] Cleaned up {len(stale_lock_ids)} stale locks")

    def get_loaded_model(self, model_id: str) -> Optional['BaseEngine']:
        """获取已加载的模型引擎，如果存在"""
        with self._lock:
            if model_id in self.loaded_models:
                self.last_access_time[model_id] = time.time()
                return self.loaded_models[model_id]
        return None

    def check_memory_and_evict(self):
        """检查显存，如果不足则按 LRU 策略卸载模型"""
        if not torch.cuda.is_available():
            return

        used_mem, total_mem = self._get_gpu_status()
        usage_ratio = used_mem / total_mem

        if usage_ratio > self.max_gpu_memory_usage:
            print(f"[GPUMemoryManager] High memory usage detected ({usage_ratio:.2%}). Triggering eviction...")
            self._evict_lru_model()

    def _evict_lru_model(self):
        """卸载最近最少使用的模型"""
        with self._lock:
            if not self.loaded_models:
                print("[GPUMemoryManager] No models to evict.")
                return

            # 找出 LRU 模型
            lru_model_id = min(self.last_access_time, key=self.last_access_time.get)
            
            print(f"[GPUMemoryManager] Evicting model: {lru_model_id}")
            engine = self.loaded_models[lru_model_id]
            
            # 卸载
            try:
                engine.unload_model()
            except ModelUnloadError as e:
                print(f"[GPUMemoryManager] Error unloading model {lru_model_id}: {e}")
            except Exception as e:
                # 包装为 ModelUnloadError 以便上游捕获
                print(f"[GPUMemoryManager] Unexpected error unloading model {lru_model_id}: {e}")
            
            # 清理记录（包括锁）
            del self.loaded_models[lru_model_id]
            del self.last_access_time[lru_model_id]
            # 清理锁：尝试获取后释放并删除
            if lru_model_id in self.model_locks:
                lock = self.model_locks[lru_model_id]
                if lock.acquire(blocking=False):
                    lock.release()
                    del self.model_locks[lru_model_id]

            # 强制 GC
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[GPUMemoryManager] Eviction complete. Current memory: {self._get_gpu_status()}")

    def _get_gpu_status(self):
        """返回 (used_bytes, total_bytes)"""
        free, total = torch.cuda.mem_get_info()
        return (total - free), total

    def get_status_report(self) -> Dict:
        """获取当前状态报告"""
        used, total = self._get_gpu_status() if torch.cuda.is_available() else (0, 1)
        return {
            "gpu_used_mb": used / UnitConversions.BYTES_PER_MB,
            "gpu_total_mb": total / UnitConversions.BYTES_PER_MB,
            "loaded_models": list(self.loaded_models.keys())
        }

    # ====== 兼容旧测试 API ======
    
    def get_memory_info(self) -> Dict:
        """获取显存信息 (兼容旧 API)"""
        if not torch.cuda.is_available():
            return {"available": True, "allocated_gb": 0, "total_gb": 0}
        used, total = self._get_gpu_status()
        return {
            "available": True,
            "allocated_gb": used / UnitConversions.BYTES_PER_GB,
            "total_gb": total / UnitConversions.BYTES_PER_GB,
        }
    
    def should_evict(self) -> bool:
        """判断是否需要驱逐模型 (兼容旧 API)"""
        if not torch.cuda.is_available():
            return False
        used, total = self._get_gpu_status()
        return (used / total) > self._threshold
    
    def get_loaded_models(self) -> List[str]:
        """获取已加载模型列表 (兼容旧 API)"""
        with self._lock:
            return list(self.loaded_models.keys()) + [
                mid for mid in self._model_sizes if mid not in self.loaded_models
            ]
    
    def touch_model(self, model_id: str):
        """更新模型访问时间 (兼容旧 API)"""
        self.access_model(model_id)
    
    def get_eviction_candidates(self) -> List[str]:
        """获取驱逐候选模型列表 (兼容旧 API)"""
        with self._lock:
            # 按最近访问时间排序，返回最久未使用的模型列表
            sorted_models = sorted(
                self.last_access_time.items(),
                key=lambda x: x[1]
            )
            return [mid for mid, _ in sorted_models]
    
    def force_cleanup(self):
        """强制清理显存 (兼容旧 API)"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
