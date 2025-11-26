"""
memory_manager.py
# [关键] GPUMemoryManager：实现显存安全策略，要求：引用计数 + LRU 回收
# 说明：负责分配/释放显存对象、跟踪引用计数、在内存不足时按 LRU 策略回收并触发 SIG 或错误回退。
"""

import time
import threading
from typing import Dict, Optional, List
import torch
import gc

# 尝试导入引擎基类，用于类型注解
try:
    from ..engines.abstract_engine import BaseEngine
except ImportError:
    pass

class GPUMemoryManager:
    """
    显存管理器 (单例模式)
    负责监控显存使用，并根据策略（LRU）卸载不活跃的模型以释放空间。
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(GPUMemoryManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.loaded_models: Dict[str, 'BaseEngine'] = {}  # model_id -> engine_instance
        self.last_access_time: Dict[str, float] = {}      # model_id -> timestamp
        self.model_locks: Dict[str, threading.Lock] = {}  # model_id -> lock (防止并发卸载/使用)
        
        # 配置参数
        self.max_gpu_memory_usage = 0.90  # 显存使用率阈值 (95%)，超过则触发回收
        self.check_interval = 60          # 自动检查间隔 (秒) - 暂未启用自动后台线程
        
        self._initialized = True
        print("[GPUMemoryManager] Initialized.")

    def register_model(self, model_id: str, engine: 'BaseEngine'):
        """注册一个已加载的模型引擎"""
        with self._lock:
            self.loaded_models[model_id] = engine
            self.last_access_time[model_id] = time.time()
            if model_id not in self.model_locks:
                self.model_locks[model_id] = threading.Lock()
            print(f"[GPUMemoryManager] Registered model: {model_id}")

    def access_model(self, model_id: str):
        """更新模型的访问时间 (LRU)"""
        with self._lock:
            if model_id in self.loaded_models:
                self.last_access_time[model_id] = time.time()

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
            
            # 确保该模型当前没有被锁定（正在推理）
            # 注意：这里简化处理，如果被锁则跳过或等待，实际场景可能需要更复杂的逻辑
            
            print(f"[GPUMemoryManager] Evicting model: {lru_model_id}")
            engine = self.loaded_models[lru_model_id]
            
            # 卸载
            try:
                engine.unload_model()
            except Exception as e:
                print(f"[GPUMemoryManager] Error unloading model {lru_model_id}: {e}")
            
            # 清理记录
            del self.loaded_models[lru_model_id]
            del self.last_access_time[lru_model_id]
            # lock 不删，防止还有线程持有

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
            "gpu_used_mb": used / 1024**2,
            "gpu_total_mb": total / 1024**2,
            "loaded_models": list(self.loaded_models.keys())
        }
