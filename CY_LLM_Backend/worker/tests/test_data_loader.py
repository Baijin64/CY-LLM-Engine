"""
test_data_loader.py
training/data/loader.py 和 training/data/formatter.py 模块的单元测试
"""

import pytest
import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from worker.training.data.loader import DatasetLoader, load_dataset
from worker.training.data.formatter import DataFormatter, format_conversation


class TestDatasetLoader:
    """测试数据集加载器"""

    @pytest.fixture
    def sample_jsonl_file(self):
        """创建示例 JSONL 文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for i in range(5):
                f.write(json.dumps({
                    "instruction": f"Question {i}",
                    "input": "",
                    "output": f"Answer {i}"
                }) + "\n")
            return f.name

    @pytest.fixture
    def sample_json_file(self):
        """创建示例 JSON 文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([
                {"instruction": "Q1", "output": "A1"},
                {"instruction": "Q2", "output": "A2"},
            ], f)
            return f.name

    def test_load_jsonl(self, sample_jsonl_file):
        """应能加载 JSONL 文件"""
        try:
            data = load_dataset(sample_jsonl_file)
            assert len(data) == 5
        finally:
            os.unlink(sample_jsonl_file)

    def test_load_json(self, sample_json_file):
        """应能加载 JSON 文件"""
        try:
            data = load_dataset(sample_json_file)
            assert len(data) == 2
        finally:
            os.unlink(sample_json_file)

    def test_load_nonexistent_file(self):
        """加载不存在的文件应抛出异常"""
        with pytest.raises(FileNotFoundError):
            load_dataset("/nonexistent/path.json")

    def test_load_invalid_format(self):
        """无效格式应抛出异常"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("not json")
            path = f.name
        
        try:
            with pytest.raises((ValueError, json.JSONDecodeError)):
                load_dataset(path)
        finally:
            os.unlink(path)

    def test_loader_with_split(self):
        """应支持训练/验证拆分"""
        loader = DatasetLoader()
        
        data = [{"text": f"sample {i}"} for i in range(100)]
        
        train, val = loader.split(data, val_ratio=0.1)
        
        assert len(train) == 90
        assert len(val) == 10

    def test_loader_shuffle(self):
        """应支持数据打乱"""
        loader = DatasetLoader(shuffle=True, seed=42)
        
        data = [{"text": f"sample {i}"} for i in range(10)]
        
        shuffled = loader.shuffle_data(data)
        
        # 验证已打乱（顺序不同）
        assert shuffled != data or len(data) < 2


class TestDataFormatter:
    """测试数据格式化器"""

    def test_format_alpaca_style(self):
        """应能格式化 Alpaca 风格数据"""
        formatter = DataFormatter(template="alpaca")
        
        sample = {
            "instruction": "What is AI?",
            "input": "",
            "output": "AI is artificial intelligence."
        }
        
        formatted = formatter.format(sample)
        
        assert "What is AI?" in formatted
        assert "artificial intelligence" in formatted

    def test_format_conversation_style(self):
        """应能格式化对话风格数据"""
        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        
        formatted = format_conversation(conversation)
        
        assert "Hello" in formatted
        assert "Hi there!" in formatted

    def test_format_with_system_prompt(self):
        """应支持系统提示"""
        formatter = DataFormatter(
            template="chatml",
            system_prompt="You are a helpful assistant."
        )
        
        sample = {
            "instruction": "Help me",
            "output": "Sure!"
        }
        
        formatted = formatter.format(sample)
        
        assert "helpful assistant" in formatted

    def test_custom_template(self):
        """应支持自定义模板"""
        formatter = DataFormatter(
            template="custom",
            template_str="### Input:\n{instruction}\n### Output:\n{output}"
        )
        
        sample = {"instruction": "Q", "output": "A"}
        
        formatted = formatter.format(sample)
        
        assert "### Input:" in formatted
        assert "### Output:" in formatted

    def test_batch_format(self):
        """应支持批量格式化"""
        formatter = DataFormatter()
        
        samples = [
            {"instruction": f"Q{i}", "output": f"A{i}"}
            for i in range(5)
        ]
        
        formatted = formatter.format_batch(samples)
        
        assert len(formatted) == 5


class TestDataValidation:
    """测试数据验证"""

    def test_validate_required_fields(self):
        """应验证必需字段"""
        from worker.training.data.loader import validate_sample
        
        valid = {"instruction": "Q", "output": "A"}
        assert validate_sample(valid) is True
        
        invalid = {"instruction": "Q"}  # 缺少 output
        assert validate_sample(invalid) is False

    def test_filter_invalid_samples(self):
        """应过滤无效样本"""
        from worker.training.data.loader import filter_valid_samples
        
        samples = [
            {"instruction": "Q1", "output": "A1"},
            {"instruction": "Q2"},  # 无效
            {"instruction": "Q3", "output": "A3"},
        ]
        
        valid = filter_valid_samples(samples)
        
        assert len(valid) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
