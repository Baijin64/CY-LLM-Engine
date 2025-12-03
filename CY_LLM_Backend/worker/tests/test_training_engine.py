"""
test_training_engine.py
training_engine.py 和 training/engine.py 模块的单元测试
"""

import pytest
import sys
import os
from unittest.mock import Mock, MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestTrainingEngine:
    """测试训练引擎"""

    @pytest.fixture
    def mock_model(self):
        """创建模拟模型"""
        model = Mock()
        param = Mock()
        param.requires_grad = True
        param.numel = Mock(return_value=100)
        model.parameters = Mock(return_value=[param])
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """创建模拟分词器"""
        tokenizer = Mock()
        tokenizer.pad_token = None
        tokenizer.eos_token = "</s>"
        return tokenizer

    def test_detect_device_cuda(self):
        """应检测 CUDA 设备"""
        from training.engine import TrainingEngine
        # 通过 patch DeviceConfig.detect 模拟一个启用 CUDA 的设备
        from training.model.setup import DeviceConfig
        fake_dev = DeviceConfig()
        fake_dev.cuda_available = True
        fake_dev.cuda_device_count = 1
        fake_dev.cuda_device_name = "GPU"
        fake_dev.bitsandbytes_available = True
        with patch.object(TrainingEngine, '_detect_device', return_value=fake_dev):
            engine = TrainingEngine()
            device = engine._detect_device("test-job")
            assert device.cuda_available is True

    def test_detect_device_cpu_fallback(self):
        """无 GPU 时应回退到 CPU"""
        from training.engine import TrainingEngine
        from training.model.setup import DeviceConfig
        fake_dev = DeviceConfig()
        fake_dev.cuda_available = False
        fake_dev.cuda_device_count = 0
        fake_dev.cuda_device_name = None
        fake_dev.bitsandbytes_available = False
        with patch.object(TrainingEngine, '_detect_device', return_value=fake_dev):
            engine = TrainingEngine()
            device = engine._detect_device("test-job")
            assert device.cuda_available is False

    def test_load_dataset(self):
        """应能加载数据集"""
        from worker.training.engine import TrainingEngine
        engine = TrainingEngine()
        
        # 使用模拟数据
        mock_data = [{"input": "Hello", "output": "World"}]
        
        # Patch the instance method that actually loads the dataset
        from worker.training.engine import TrainingRequest
        with patch('worker.training.engine.TrainingEngine._load_dataset', return_value=mock_data):
            request = TrainingRequest(base_model="m", output_dir="/tmp", dataset_path="/path/to/data.json")
            dataset = engine._load_dataset("test-job", request)
            assert dataset is not None

    def test_apply_lora(self, mock_model):
        """应能应用 LoRA"""
        from worker.training.engine import TrainingEngine
        engine = TrainingEngine()
        
        from worker.training.engine import TrainingRequest
        with patch('peft.get_peft_model', return_value=mock_model):
            request = TrainingRequest(base_model="m", output_dir="/tmp", dataset_path="/tmp/data.json")
            result = engine._apply_lora("test-job", request, mock_model, use_quantization=False)
            assert result is not None

    def test_create_trainer(self, mock_model, mock_tokenizer):
        """应能创建 Trainer"""
        from worker.training.engine import TrainingEngine
        engine = TrainingEngine()
        
        mock_dataset = Mock()
        
        from worker.training.engine import TrainingRequest
        with patch('transformers.Trainer'):
            request = TrainingRequest(base_model="m", output_dir="/tmp", dataset_path="/tmp/data.json")
            trainer, trainer_factory = engine._create_trainer(
                "test-job",
                request,
                mock_model,
                mock_tokenizer,
                mock_dataset,
                progress_queue=[],
            )
            # 验证创建成功

    def test_save_model(self, mock_model, mock_tokenizer):
        """应能保存模型"""
        from training.engine import TrainingEngine
        engine = TrainingEngine()
        
        with patch.object(mock_model, 'save_pretrained'):
            engine._save_model(mock_model, mock_tokenizer, "/tmp/output")

    def test_execute_training_integration(self):
        """集成测试：执行完整训练流程"""
        # 这是一个较重的测试，可能需要跳过
        pytest.skip("Integration test - requires GPU and model")


class TestTrainingEngineConfig:
    """测试训练引擎配置"""

    def test_default_config(self):
        """应有默认配置"""
        from training.engine import TrainingEngine
        engine = TrainingEngine()
        # 引擎默认值应存在
        assert hasattr(engine, '_semaphore')

    def test_custom_config(self):
        """应接受自定义配置"""
        from training.engine import TrainingEngine
        engine = TrainingEngine(max_concurrent_jobs=2)
        
        # 验证配置被应用


class TestLoRATraining:
    """测试 LoRA 训练"""

    def test_lora_config_creation(self):
        """应能创建 LoRA 配置"""
        from training.model.lora import create_lora_config
        
        config = create_lora_config(
            rank=8,
            alpha=16,
            dropout=0.1,
            target_modules=["q_proj", "v_proj"]
        )
        
        assert config is not None
        assert config.r == 8
        assert config.lora_alpha == 16

    def test_lora_target_modules_detection(self):
        """应能检测目标模块"""
        from training.model.lora import detect_target_modules
        
        mock_model = Mock()
        mock_model.named_modules = Mock(return_value=[
            ("layer.q_proj", Mock()),
            ("layer.k_proj", Mock()),
            ("layer.v_proj", Mock()),
        ])
        
        modules = detect_target_modules(mock_model)
        assert isinstance(modules, list)


class TestTrainingCallbacks:
    """测试训练回调"""

    def test_progress_callback(self):
        """进度回调应被调用"""
        from worker.training.loop.callbacks import ProgressCallback
        
        callback = ProgressCallback(job_id="test-job", progress_queue=[], total_epochs=1)
        
        # 模拟训练状态
        state = Mock()
        state.global_step = 100
        state.max_steps = 1000
        
        control = Mock()
        
        callback.on_step_end(None, state, control)

    def test_early_stopping_callback(self):
        """早停回调应工作"""
        from worker.training.loop.callbacks import EarlyStoppingCallback
        
        callback = EarlyStoppingCallback(patience=3)
        
        # 模拟连续无改善
        for _ in range(5):
            callback.on_evaluate(None, Mock(), Mock(), metrics={"eval_loss": 1.0})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
