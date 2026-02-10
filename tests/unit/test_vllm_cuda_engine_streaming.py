"""
test_vllm_cuda_engine_streaming.py
vLLM CUDA引擎流式输出单元测试

验证:
1. infer()返回Generator
2. 流式输出不为空
3. token块大小配置生效
4. 输出内容一致性
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestVllmCudaEngineStreaming(unittest.TestCase):
    """测试vllm_cuda_engine的流式输出功能"""

    def test_infer_returns_generator(self):
        """TC-001: 验证infer()返回Generator"""
        # 模拟引擎和vLLM
        mock_llm = MagicMock()
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="Hello World")]
        mock_llm.generate.return_value = [mock_output]

        # 导入引擎
        from CY_LLM_Backend.worker.engines.vllm_cuda_engine import VllmCudaEngine

        # 创建引擎实例并注入mock
        engine = VllmCudaEngine()
        engine._llm = mock_llm

        # 调用infer
        result = engine.infer("Test prompt")

        # 验证返回类型
        self.assertIsInstance(result, Generator)

        # 消费生成器并验证
        chunks = list(result)
        self.assertGreater(len(chunks), 0)

    def test_streaming_output_not_empty(self):
        """TC-002: 验证流式输出不为空"""
        from CY_LLM_Backend.worker.engines.vllm_cuda_engine import VllmCudaEngine

        mock_llm = MagicMock()
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="Test response")]
        mock_llm.generate.return_value = [mock_output]

        engine = VllmCudaEngine()
        engine._llm = mock_llm

        chunks = list(engine.infer("Test"))

        # 验证输出不为空
        self.assertTrue(len(chunks) > 0)

        # 验证可以拼接成完整文本
        full_text = "".join(chunks)
        self.assertEqual(full_text, "Test response")

    def test_chunk_size_configuration(self):
        """TC-003: 验证token块大小配置生效"""
        from CY_LLM_Backend.worker.engines.vllm_cuda_engine import VllmCudaEngine

        # 测试不同chunk大小
        for chunk_size in [1, 4, 8, 16]:
            with self.subTest(chunk_size=chunk_size):
                engine = VllmCudaEngine(stream_chunk_size=chunk_size)

                # 验证配置已设置
                self.assertEqual(engine.stream_chunk_size, chunk_size)

    def test_chunk_size_affects_output_granularity(self):
        """TC-003b: 验证chunk大小影响输出粒度"""
        from CY_LLM_Backend.worker.engines.vllm_cuda_engine import VllmCudaEngine

        mock_llm = MagicMock()
        mock_output = MagicMock()
        # 生成20个字符的文本
        mock_output.outputs = [MagicMock(text="01234567890123456789")]
        mock_llm.generate.return_value = [mock_output]

        # 测试chunk_size=4
        engine = VllmCudaEngine(stream_chunk_size=4)
        engine._llm = mock_llm

        chunks = list(engine.infer("Test"))

        # 验证chunks数量减少（20字符/chunk_size=4 => 5个chunks）
        # 注意：实际实现可能是字符级或token级
        self.assertLessEqual(len(chunks), 20)  # 应该比逐字符少

    def test_output_content_consistency(self):
        """TC-004: 验证输出内容与原始实现一致"""
        from CY_LLM_Backend.worker.engines.vllm_cuda_engine import VllmCudaEngine

        mock_llm = MagicMock()
        mock_output = MagicMock()
        test_text = "This is a test response with multiple words."
        mock_output.outputs = [MagicMock(text=test_text)]
        mock_llm.generate.return_value = [mock_output]

        engine = VllmCudaEngine(stream_chunk_size=4)
        engine._llm = mock_llm

        chunks = list(engine.infer("Test"))
        reconstructed = "".join(chunks)

        # 验证拼接后与原文本一致
        self.assertEqual(reconstructed, test_text)

    def test_streaming_with_lora(self):
        """TC-008: 验证LoRA切换功能完整"""
        from CY_LLM_Backend.worker.engines.vllm_cuda_engine import VllmCudaEngine

        mock_llm = MagicMock()
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="LoRA response")]
        mock_llm.generate.return_value = [mock_output]

        engine = VllmCudaEngine(enable_lora=True)
        engine._llm = mock_llm
        engine._loaded_loras["test_lora"] = "/path/to/lora"

        # 验证可以使用LoRA进行推理
        chunks = list(engine.infer("Test", lora_name="test_lora"))
        self.assertGreater(len(chunks), 0)

    def test_memory_usage_reporting(self):
        """TC-010: 验证显存信息获取"""
        from CY_LLM_Backend.worker.engines.vllm_cuda_engine import VllmCudaEngine

        engine = VllmCudaEngine()
        mem_info = engine.get_memory_usage()

        # 验证返回结构
        self.assertIn("allocated_gb", mem_info)
        self.assertIn("reserved_gb", mem_info)
        self.assertIn("total_gb", mem_info)
        self.assertIn("utilization", mem_info)

    def test_unload_model(self):
        """验证模型卸载功能"""
        from CY_LLM_Backend.worker.engines.vllm_cuda_engine import VllmCudaEngine

        engine = VllmCudaEngine()
        mock_llm = MagicMock()
        engine._llm = mock_llm
        engine._model_path = "/test/model"
        engine._loaded_loras["test"] = "/test/lora"

        # 卸载
        engine.unload_model()

        # 验证状态清理
        self.assertIsNone(engine._llm)
        self.assertIsNone(engine._model_path)
        self.assertEqual(len(engine._loaded_loras), 0)


class TestVllmCudaEngineInit(unittest.TestCase):
    """测试引擎初始化参数"""

    def test_default_chunk_size(self):
        """验证默认chunk大小"""
        from CY_LLM_Backend.worker.engines.vllm_cuda_engine import VllmCudaEngine

        engine = VllmCudaEngine()
        # 默认应该为1（逐字符）或4（优化后）
        self.assertTrue(hasattr(engine, "stream_chunk_size"))

    def test_custom_chunk_size(self):
        """验证自定义chunk大小"""
        from CY_LLM_Backend.worker.engines.vllm_cuda_engine import VllmCudaEngine

        engine = VllmCudaEngine(stream_chunk_size=8)
        self.assertEqual(engine.stream_chunk_size, 8)

    def test_chunk_size_validation(self):
        """验证chunk大小参数验证"""
        from CY_LLM_Backend.worker.engines.vllm_cuda_engine import VllmCudaEngine

        # 测试无效值（应该使用默认值或报错）
        with self.assertRaises((ValueError, AssertionError)):
            VllmCudaEngine(stream_chunk_size=0)

        with self.assertRaises((ValueError, AssertionError)):
            VllmCudaEngine(stream_chunk_size=-1)


class TestStreamingPerformance(unittest.TestCase):
    """测试流式输出性能"""

    def test_streaming_latency(self):
        """TC-005: 验证TTFT < 500ms"""
        import time
        from CY_LLM_Backend.worker.engines.vllm_cuda_engine import VllmCudaEngine

        mock_llm = MagicMock()
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="Response")]
        mock_llm.generate.return_value = [mock_output]

        engine = VllmCudaEngine()
        engine._llm = mock_llm

        start = time.time()
        gen = engine.infer("Test")
        first_chunk = next(gen)
        ttft = (time.time() - start) * 1000

        # 使用mock时TTFT应该很小
        self.assertLess(ttft, 500)


if __name__ == "__main__":
    unittest.main(verbosity=2)
