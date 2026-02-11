"""
test_abstract_engine.py
engines/abstract_engine.py 模块的单元测试
"""

import pytest
import sys
import os
from typing import Generator, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from worker.engines.abstract_engine import BaseEngine


class ConcreteEngine(BaseEngine):
    """用于测试的具体引擎实现"""
    
    def __init__(self):
        self._loaded = False
        self._model_path = None
    
    def load_model(self, model_path: str, adapter_path=None, **kwargs) -> None:
        self._loaded = True
        self._model_path = model_path
    
    def infer(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        if not self._loaded:
            raise RuntimeError("Model not loaded")
        for word in prompt.split():
            yield word
    
    def unload_model(self) -> None:
        self._loaded = False
        self._model_path = None
    
    def get_memory_usage(self) -> Dict[str, float]:
        return {
            "allocated_gb": 1.0,
            "reserved_gb": 2.0,
            "total_gb": 24.0,
            "utilization": 0.04,
        }


class TestBaseEngine:
    """测试 BaseEngine 抽象类"""

    def test_cannot_instantiate_abstract(self):
        """不能直接实例化抽象类"""
        with pytest.raises(TypeError):
            BaseEngine()

    def test_concrete_implementation(self):
        """具体实现应能实例化"""
        engine = ConcreteEngine()
        assert engine is not None

    def test_load_model(self):
        """load_model 应加载模型"""
        engine = ConcreteEngine()
        engine.load_model("/path/to/model")
        
        assert engine._loaded is True
        assert engine._model_path == "/path/to/model"

    def test_infer_returns_generator(self):
        """infer 应返回生成器"""
        engine = ConcreteEngine()
        engine.load_model("/path/to/model")
        
        result = engine.infer("Hello World")
        
        assert hasattr(result, '__iter__')
        tokens = list(result)
        assert tokens == ["Hello", "World"]

    def test_infer_without_load_raises(self):
        """未加载模型时 infer 应抛出异常"""
        engine = ConcreteEngine()
        
        with pytest.raises(RuntimeError):
            list(engine.infer("test"))

    def test_unload_model(self):
        """unload_model 应卸载模型"""
        engine = ConcreteEngine()
        engine.load_model("/path/to/model")
        engine.unload_model()
        
        assert engine._loaded is False

    def test_get_memory_usage(self):
        """get_memory_usage 应返回内存信息"""
        engine = ConcreteEngine()
        
        usage = engine.get_memory_usage()
        
        assert "allocated_gb" in usage
        assert "total_gb" in usage
        assert usage["utilization"] >= 0


class TestBaseEngineOptionalMethods:
    """测试 BaseEngine 可选方法"""

    def test_get_model_info_default(self):
        """get_model_info 应有默认实现"""
        engine = ConcreteEngine()
        
        info = engine.get_model_info()
        
        assert isinstance(info, dict)
        assert "engine" in info

    def test_health_check_default(self):
        """health_check 应有默认实现"""
        engine = ConcreteEngine()
        
        healthy = engine.health_check()
        
        assert isinstance(healthy, bool)

    def test_is_async_property(self):
        """is_async 应返回 False（同步引擎）"""
        engine = ConcreteEngine()
        
        assert engine.is_async is False


class TestBaseEngineAsyncMethods:
    """测试 BaseEngine 异步方法"""

    @pytest.mark.asyncio
    async def test_async_infer_wrapper(self):
        """async_infer 应包装同步 infer"""
        engine = ConcreteEngine()
        engine.load_model("/path/to/model")
        
        tokens = []
        async for token in engine.async_infer("Hello World"):
            tokens.append(token)
        
        assert tokens == ["Hello", "World"]


class TestEngineLifecycle:
    """测试引擎生命周期"""

    def test_full_lifecycle(self):
        """完整生命周期应正常工作"""
        engine = ConcreteEngine()
        
        # 1. 加载
        engine.load_model("/model")
        assert engine._loaded
        
        # 2. 推理
        result = list(engine.infer("test"))
        assert result == ["test"]
        
        # 3. 卸载
        engine.unload_model()
        assert not engine._loaded

    def test_reload_model(self):
        """重新加载模型应工作"""
        engine = ConcreteEngine()
        
        engine.load_model("/model1")
        engine.unload_model()
        engine.load_model("/model2")
        
        assert engine._model_path == "/model2"

    def test_multiple_unload_safe(self):
        """多次卸载应安全"""
        engine = ConcreteEngine()
        engine.load_model("/model")
        
        engine.unload_model()
        engine.unload_model()  # 不应崩溃


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
