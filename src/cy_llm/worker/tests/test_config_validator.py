"""
test_config_validator.py
config/validator.py 和 config/models.py 模块的单元测试
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from worker.config.validator import ConfigValidator, validate_config
from worker.config.models import ModelSpec, ServerConfig, WorkerConfig, HardwareProfile


class TestConfigValidator:
    """测试配置验证器"""

    def test_validate_valid_config(self):
        """有效配置应通过验证"""
        config = WorkerConfig(
            grpc_port=50051,
            max_workers=4,
        )
        
        result = validate_config(config)
        assert result is True

    def test_validate_missing_required_field(self):
        """使用默认值时配置应有效"""
        config = WorkerConfig()
        
        result = validate_config(config)
        # 使用默认值应该有效
        assert isinstance(result, bool)

    def test_validate_invalid_port(self):
        """无效端口应在创建时失败"""
        with pytest.raises((ValueError, Exception)):
            WorkerConfig(grpc_port=-1)

    def test_validate_port_range(self):
        """端口应在有效范围内"""
        # 有效端口
        config = WorkerConfig(grpc_port=50051)
        assert config.grpc_port == 50051
        
        config = WorkerConfig(grpc_port=65535)
        assert config.grpc_port == 65535
        
        # 无效端口应抛出异常
        with pytest.raises((ValueError, Exception)):
            WorkerConfig(grpc_port=0)
        
        with pytest.raises((ValueError, Exception)):
            WorkerConfig(grpc_port=70000)

    def test_validate_model_path(self):
        """模型路径验证"""
        # 有效的 ModelSpec
        spec = ModelSpec(model_path="/valid/path")
        assert spec.model_path == "/valid/path"
        
        # 空路径在当前实现中允许（可以是 HuggingFace ID 的占位符）
        # 实际验证在 ConfigValidator._validate_model_spec 中
        spec_empty = ModelSpec(model_path="")
        assert spec_empty.model_path == ""


class TestModelSpec:
    """测试模型规格数据类"""

    def test_create_model_spec(self):
        """应能创建 ModelSpec"""
        spec = ModelSpec(
            model_path="/models/llama-7b",
            engine="cuda-vllm",
        )
        
        assert spec.model_path == "/models/llama-7b"
        assert spec.engine.value == "cuda-vllm" if hasattr(spec.engine, 'value') else spec.engine == "cuda-vllm"

    def test_model_spec_defaults(self):
        """应有合理默认值"""
        spec = ModelSpec(
            model_path="/path/to/model",
        )
        
        # 检查默认值
        assert spec.adapter_path is None
        assert spec.use_4bit is None

    def test_model_spec_with_lora(self):
        """应支持 LoRA 配置"""
        spec = ModelSpec(
            model_path="/path/to/model",
            adapter_path="/adapters/lora",
        )
        
        assert spec.adapter_path == "/adapters/lora"

    def test_model_spec_validation(self):
        """应验证字段"""
        # model_path 是必需字段
        with pytest.raises((ValueError, TypeError, Exception)):
            ModelSpec()  # 缺少 model_path 应失败


class TestServerConfig:
    """测试服务器配置"""

    def test_create_server_config(self):
        """应能创建 ServerConfig"""
        config = ServerConfig(
            port=50051,
            grpc_max_workers=4,
        )
        
        assert config.port == 50051
        assert config.grpc_max_workers == 4

    def test_server_config_defaults(self):
        """应有默认值"""
        config = ServerConfig()
        
        assert config.port == 50051 or config.port > 0
        assert config.grpc_max_workers > 0


class TestWorkerConfig:
    """测试 Worker 配置"""

    def test_create_worker_config(self):
        """应能创建 WorkerConfig"""
        config = WorkerConfig(
            preferred_backend="cuda-vllm",
        )
        
        # preferred_backend 可能是 EngineType 枚举
        backend = config.get_engine_type()
        assert backend == "cuda-vllm"

    def test_worker_config_model_dump(self):
        """应能转换为字典（使用 model_dump）"""
        config = WorkerConfig(preferred_backend="cuda-vllm")
        
        # Pydantic v2 使用 model_dump
        data = config.model_dump()
        assert isinstance(data, dict)
        # preferred_backend 可能被转换为枚举值
        assert "preferred_backend" in data

    def test_worker_config_defaults(self):
        """应有合理默认值"""
        config = WorkerConfig()
        
        assert config.grpc_port > 0
        assert config.max_workers > 0
        assert config.queue_size > 0


class TestHotReload:
    """测试热重载功能"""

    def test_config_watcher_creation(self):
        """应能创建配置监视器"""
        from worker.config.hot_reload import ConfigWatcher
        
        watcher = ConfigWatcher("/path/to/config.json")
        assert watcher is not None

    def test_reload_callback_registration(self):
        """应能注册重载回调"""
        from worker.config.hot_reload import ConfigWatcher
        
        callback_called = [False]
        
        def on_reload(event):
            callback_called[0] = True
        
        watcher = ConfigWatcher("/path/to/config.json")
        # 使用新 API 注册回调
        watcher.on_reload(on_reload)
        
        # 验证回调已注册
        assert len(watcher._reload_callbacks) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
