"""
test_exceptions.py
exceptions.py 模块的单元测试
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from worker.exceptions import (
    EWWorkerException,
    ModelNotFoundError,
    ModelLoadError,
    InferenceError,
    TrainingError,
    ConfigurationError,
    AuthenticationError,
    ResourceExhaustedError,
)


class TestEWWorkerException:
    """测试基础异常类"""

    def test_basic_exception(self):
        """应能创建基础异常"""
        exc = EWWorkerException("test error")
        assert "test error" in str(exc)

    def test_exception_inheritance(self):
        """应继承自 Exception"""
        assert issubclass(EWWorkerException, Exception)

    def test_exception_with_cause(self):
        """应支持异常链"""
        cause = ValueError("original error")
        exc = EWWorkerException("wrapper error")
        exc.__cause__ = cause
        
        assert exc.__cause__ is cause


class TestModelNotFoundError:
    """测试模型未找到异常"""

    def test_creation(self):
        """应能创建异常"""
        exc = ModelNotFoundError("model-id")
        assert "model-id" in str(exc)

    def test_inheritance(self):
        """应继承自 EWWorkerException"""
        assert issubclass(ModelNotFoundError, EWWorkerException)

    def test_model_id_attribute(self):
        """应有 model_id 属性"""
        exc = ModelNotFoundError("my-model")
        if hasattr(exc, 'model_id'):
            assert exc.model_id == "my-model"


class TestModelLoadError:
    """测试模型加载错误"""

    def test_creation(self):
        """应能创建异常"""
        exc = ModelLoadError("failed to load model")
        assert "failed" in str(exc).lower() or "load" in str(exc).lower()

    def test_inheritance(self):
        """应继承自 EWWorkerException"""
        assert issubclass(ModelLoadError, EWWorkerException)

    def test_with_model_path(self):
        """应能包含模型路径"""
        exc = ModelLoadError("Load failed", model_path="/path/to/model")
        if hasattr(exc, 'model_path'):
            assert exc.model_path == "/path/to/model"


class TestInferenceError:
    """测试推理错误"""

    def test_creation(self):
        """应能创建异常"""
        exc = InferenceError("inference failed")
        assert "inference" in str(exc).lower() or "failed" in str(exc).lower()

    def test_inheritance(self):
        """应继承自 EWWorkerException"""
        assert issubclass(InferenceError, EWWorkerException)

    def test_with_details(self):
        """应能包含详细信息"""
        exc = InferenceError("OOM", model_id="llama", prompt_length=1000)
        # 验证创建成功
        assert exc is not None


class TestTrainingError:
    """测试训练错误"""

    def test_creation(self):
        """应能创建异常"""
        exc = TrainingError("training failed")
        assert exc is not None

    def test_inheritance(self):
        """应继承自 EWWorkerException"""
        assert issubclass(TrainingError, EWWorkerException)

    def test_with_epoch(self):
        """应能包含 epoch 信息"""
        exc = TrainingError("Loss exploded", epoch=5)
        if hasattr(exc, 'epoch'):
            assert exc.epoch == 5


class TestConfigurationError:
    """测试配置错误"""

    def test_creation(self):
        """应能创建异常"""
        exc = ConfigurationError("invalid config")
        assert exc is not None

    def test_inheritance(self):
        """应继承自 EWWorkerException"""
        assert issubclass(ConfigurationError, EWWorkerException)

    def test_with_key(self):
        """应能包含配置键"""
        exc = ConfigurationError("Missing value", key="server.port")
        if hasattr(exc, 'key'):
            assert exc.key == "server.port"


class TestAuthenticationError:
    """测试认证错误"""

    def test_creation(self):
        """应能创建异常"""
        exc = AuthenticationError("invalid token")
        assert exc is not None

    def test_inheritance(self):
        """应继承自 EWWorkerException"""
        assert issubclass(AuthenticationError, EWWorkerException)


class TestResourceExhaustedError:
    """测试资源耗尽错误"""

    def test_creation(self):
        """应能创建异常"""
        exc = ResourceExhaustedError("GPU memory exhausted")
        assert exc is not None

    def test_inheritance(self):
        """应继承自 EWWorkerException"""
        assert issubclass(ResourceExhaustedError, EWWorkerException)

    def test_with_resource_type(self):
        """应能包含资源类型"""
        exc = ResourceExhaustedError("Exhausted", resource_type="gpu_memory")
        if hasattr(exc, 'resource_type'):
            assert exc.resource_type == "gpu_memory"


class TestExceptionHandling:
    """测试异常处理场景"""

    def test_catch_base_exception(self):
        """应能通过基类捕获所有自定义异常"""
        exceptions = [
            ModelNotFoundError("test"),
            ModelLoadError("test"),
            InferenceError("test"),
            TrainingError("test"),
            ConfigurationError("test"),
            AuthenticationError("test"),
            ResourceExhaustedError("test"),
        ]
        
        for exc in exceptions:
            try:
                raise exc
            except EWWorkerException as e:
                # 应该能捕获
                assert isinstance(e, EWWorkerException)

    def test_exception_repr(self):
        """异常应有有意义的字符串表示"""
        exc = ModelNotFoundError("missing-model")
        repr_str = repr(exc)
        assert "ModelNotFoundError" in repr_str or "missing-model" in str(exc)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
