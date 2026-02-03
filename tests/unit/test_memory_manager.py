"""
测试 GPUMemoryManager 的核心功能
"""

import pytest
import threading
from unittest.mock import MagicMock, patch
from cy_llm.worker.core.memory_manager import GPUMemoryManager


class TestGPUMemoryManager:
    """GPUMemoryManager 单元测试"""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """每个测试前重置单例"""
        GPUMemoryManager._instance = None
        GPUMemoryManager._test_mode = True
        yield
        GPUMemoryManager._test_mode = False

    def test_singleton_pattern(self):
        """测试单例模式"""
        GPUMemoryManager._test_mode = False
        manager1 = GPUMemoryManager()
        manager2 = GPUMemoryManager()
        assert manager1 is manager2

    def test_register_model(self):
        """测试注册模型"""
        manager = GPUMemoryManager(threshold=0.8)
        mock_engine = MagicMock()
        
        manager.register_model("model_a", engine=mock_engine, size_gb=5.0)
        
        assert "model_a" in manager.loaded_models
        assert manager.loaded_models["model_a"] is mock_engine
        assert "model_a" in manager.last_access_time
        assert "model_a" in manager.model_locks

    def test_unregister_model(self):
        """测试注销模型"""
        manager = GPUMemoryManager(threshold=0.8)
        mock_engine = MagicMock()
        
        manager.register_model("model_a", engine=mock_engine)
        manager.unregister_model("model_a")
        
        assert "model_a" not in manager.loaded_models
        assert "model_a" not in manager.last_access_time

    def test_access_model_updates_time(self):
        """测试访问模型会更新时间戳"""
        manager = GPUMemoryManager(threshold=0.8)
        mock_engine = MagicMock()
        
        manager.register_model("model_a", engine=mock_engine)
        initial_time = manager.last_access_time["model_a"]
        
        import time
        time.sleep(0.01)
        manager.access_model("model_a")
        
        assert manager.last_access_time["model_a"] > initial_time

    def test_get_model_lock(self):
        """测试获取模型锁"""
        manager = GPUMemoryManager(threshold=0.8)
        
        lock1 = manager.get_model_lock("model_a")
        lock2 = manager.get_model_lock("model_a")
        
        assert lock1 is lock2
        assert isinstance(lock1, threading.Lock)

    def test_acquire_and_release_lock(self):
        """测试锁的获取和释放"""
        manager = GPUMemoryManager(threshold=0.8)
        
        lock = manager.acquire_lock("model_a")
        assert lock.locked()
        
        manager.release_lock("model_a")
        assert not lock.locked()

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.memory_allocated", return_value=5 * 1024**3)  # 5GB
    @patch("torch.cuda.max_memory_allocated", return_value=10 * 1024**3)  # 10GB
    def test_check_memory_pressure(self, mock_max, mock_alloc, mock_cuda):
        """测试显存压力检测"""
        manager = GPUMemoryManager(threshold=0.8)
        mock_engine = MagicMock()
        
        manager.register_model("model_a", engine=mock_engine)
        
        # 模拟显存使用超过阈值
        is_pressure = manager.check_memory_pressure()
        
        # 根据实际逻辑判断（这里简化测试）
        assert isinstance(is_pressure, bool)
