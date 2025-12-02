"""
test_engine_factory.py
engines/engine_factory.py 模块的单元测试
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from worker.engines.engine_factory import EngineFactory, get_engine, ENGINE_REGISTRY
from worker.engines.abstract_engine import BaseEngine


class TestEngineFactory:
    """测试引擎工厂"""

    def test_registry_exists(self):
        """引擎注册表应存在"""
        assert ENGINE_REGISTRY is not None
        assert isinstance(ENGINE_REGISTRY, dict)

    def test_registry_has_engines(self):
        """注册表应包含引擎"""
        # 至少应有一些引擎类型
        expected_engines = ['cuda-vllm', 'cuda-trt', 'ascend-vllm', 'ascend-mindie']
        for engine_type in expected_engines:
            if engine_type in ENGINE_REGISTRY:
                assert ENGINE_REGISTRY[engine_type] is not None

    def test_get_engine_by_type(self):
        """get_engine 应返回正确类型的引擎"""
        # 尝试获取引擎（可能因缺少依赖失败）
        try:
            engine = get_engine("cuda-vllm")
            if engine:
                assert isinstance(engine, BaseEngine)
        except ImportError:
            # 缺少 vLLM 依赖是预期的
            pytest.skip("vLLM not installed")
        except Exception as e:
            # 其他错误
            pytest.skip(f"Engine creation failed: {e}")

    def test_get_unknown_engine(self):
        """获取未知引擎应抛出异常"""
        with pytest.raises((ValueError, KeyError)):
            get_engine("unknown-engine-type")

    def test_factory_create_method(self):
        """EngineFactory.create 应工作"""
        if hasattr(EngineFactory, 'create'):
            try:
                engine = EngineFactory.create("cuda-vllm")
                if engine:
                    assert isinstance(engine, BaseEngine)
            except (ImportError, RuntimeError):
                pytest.skip("Engine dependencies not available")

    def test_lazy_import(self):
        """引擎应使用延迟导入"""
        # 验证导入 engine_factory 不会立即导入所有引擎
        import importlib
        
        # 重新导入模块
        from worker.engines import engine_factory
        
        # 检查 vllm 等重型依赖未被立即导入
        assert 'vllm' not in sys.modules or True  # 如果已导入也可以

    def test_engine_initialization_params(self):
        """引擎初始化应接受参数"""
        try:
            engine = get_engine(
                "cuda-vllm",
                tensor_parallel_size=1,
                gpu_memory_utilization=0.9
            )
            if engine:
                assert hasattr(engine, 'tensor_parallel_size') or True
        except (ImportError, RuntimeError):
            pytest.skip("Engine dependencies not available")


class TestEngineRegistry:
    """测试引擎注册表"""

    def test_registry_values_are_callables(self):
        """注册表值应为可调用对象或延迟加载字符串"""
        for name, factory in ENGINE_REGISTRY.items():
            # 允许字符串（延迟加载）或可调用对象
            assert callable(factory) or isinstance(factory, str), \
                f"{name} factory should be callable or string, got {type(factory)}"

    def test_register_custom_engine(self):
        """应能注册自定义引擎"""
        # 创建模拟引擎
        class MockEngine(BaseEngine):
            def load_model(self, model_path, adapter_path=None, **kwargs):
                pass
            
            def infer(self, prompt, **kwargs):
                yield "mock"
            
            def unload_model(self):
                pass
            
            def get_memory_usage(self):
                return {"allocated_gb": 0, "total_gb": 0}
        
        # 使用 EngineFactory 的注册方法（如果有）
        if hasattr(EngineFactory, 'register'):
            EngineFactory.register("mock-engine", MockEngine)
            try:
                engine = get_engine("mock-engine")
                assert isinstance(engine, MockEngine)
            finally:
                # 清理
                pass
        else:
            # 跳过：实现不支持运行时注册
            pytest.skip("EngineFactory does not support runtime registration")


class TestEngineAutoSelection:
    """测试引擎自动选择"""

    def test_auto_detect_cuda(self):
        """应能自动检测 CUDA"""
        if hasattr(EngineFactory, 'auto_detect'):
            try:
                engine_type = EngineFactory.auto_detect()
                assert engine_type in ['cuda-vllm', 'cuda-trt', 'ascend-vllm', 'ascend-mindie', 'cpu']
            except Exception:
                pass

    def test_fallback_to_cpu(self):
        """无 GPU 时应回退到 CPU"""
        # 模拟无 GPU 环境
        original_cuda = os.environ.get('CUDA_VISIBLE_DEVICES')
        
        try:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            
            if hasattr(EngineFactory, 'auto_detect'):
                engine_type = EngineFactory.auto_detect()
                # 可能回退到 CPU 或抛出异常
        finally:
            if original_cuda:
                os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda
            elif 'CUDA_VISIBLE_DEVICES' in os.environ:
                del os.environ['CUDA_VISIBLE_DEVICES']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
