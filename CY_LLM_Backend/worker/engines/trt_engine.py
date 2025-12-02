"""
trt_engine.py
[推理引擎] NVIDIA CUDA + TensorRT-LLM 实现
说明：基于 TensorRT-LLM 的极致性能推理引擎，需要预编译模型。

特点：
  - 极致性能：TensorRT 优化，延迟最低
  - 需预编译：模型需要提前转换为 TensorRT 格式
  - 显存效率：Inflight Batching + KV Cache 优化

依赖：
  pip install tensorrt-llm
  # 或从 NVIDIA 官方获取预编译包
"""

from __future__ import annotations

import gc
import logging
from typing import Any, Dict, Generator, List, Optional

from .abstract_engine import BaseEngine

LOGGER = logging.getLogger("cy_llm.worker.engines.trt")

# TensorRT-LLM 导入标记
_trt_imported = False
_ModelRunner = None
_ModelRunnerCpp = None


def _ensure_trt_imported():
    """确保 TensorRT-LLM 已导入"""
    global _trt_imported, _ModelRunner, _ModelRunnerCpp
    if not _trt_imported:
        try:
            # TensorRT-LLM 的导入路径可能因版本不同而变化
            from tensorrt_llm.runtime import ModelRunner
            _ModelRunner = ModelRunner
            _trt_imported = True
            LOGGER.info("TensorRT-LLM 模块加载成功")
        except ImportError as e:
            raise ImportError(
                "TensorRT-LLM 未安装。请参考 NVIDIA 官方文档安装：\n"
                "https://github.com/NVIDIA/TensorRT-LLM\n"
                f"原始错误: {e}"
            ) from e


class TensorRTEngine(BaseEngine):
    """
    基于 TensorRT-LLM 的 NVIDIA CUDA 推理引擎（极致性能）。
    
    特点：
      - 最高性能：经过 TensorRT 优化的推理
      - 需要预编译：模型必须先转换为 TensorRT 引擎格式
      - 适合生产部署：固定模型场景下的最佳选择
      
    使用流程：
        1. 使用 TensorRT-LLM 工具将 HF 模型转换为 TRT 引擎
        2. 加载转换后的引擎文件
        
    示例：
        >>> engine = TensorRTEngine()
        >>> engine.load_model("/path/to/trt_engine_dir")
        >>> for token in engine.infer("你好"):
        ...     print(token, end="", flush=True)
    """

    def __init__(
        self,
        max_batch_size: int = 8,
        max_input_len: int = 2048,
        max_output_len: int = 512,
        **kwargs
    ) -> None:
        """
        初始化 TensorRT-LLM 引擎。
        
        Args:
            max_batch_size: 最大批处理大小
            max_input_len: 最大输入长度
            max_output_len: 最大输出长度
        """
        _ensure_trt_imported()
        
        self.max_batch_size = max_batch_size
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.extra_kwargs = kwargs
        
        self._runner: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._model_path: Optional[str] = None
        
        LOGGER.info(
            "TensorRTEngine 初始化: batch=%d, input=%d, output=%d",
            max_batch_size, max_input_len, max_output_len
        )

    def load_model(
        self,
        model_path: str,
        adapter_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        加载 TensorRT 引擎。
        
        Args:
            model_path: TensorRT 引擎目录路径（包含 config.json 和 engine 文件）
            adapter_path: 暂不支持 LoRA（TRT 需要将 LoRA 融合后重新编译）
            tokenizer_path: Tokenizer 路径，默认与 model_path 相同
            **kwargs: 额外参数
        """
        if adapter_path:
            LOGGER.warning(
                "TensorRT-LLM 不支持动态 LoRA 加载。"
                "请将 LoRA 权重融合到基座模型后重新编译 TRT 引擎。"
            )
        
        LOGGER.info("正在加载 TensorRT 引擎: %s", model_path)
        
        try:
            # 加载 Tokenizer
            from transformers import AutoTokenizer
            tokenizer_path = tokenizer_path or model_path
            self._tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, 
                trust_remote_code=True
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            # 加载 TensorRT-LLM Runner
            self._runner = _ModelRunner.from_dir(
                engine_dir=model_path,
                rank=0,  # 单卡场景
            )
            
            self._model_path = model_path
            LOGGER.info("TensorRT 引擎加载成功: %s", model_path)
            
        except Exception as e:
            LOGGER.error("TensorRT 引擎加载失败: %s", e)
            raise

    def infer(
        self,
        prompt: str,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        流式推理。
        
        Args:
            prompt: 输入提示
            **kwargs: 生成参数
            
        Yields:
            生成的文本 token
        """
        if self._runner is None or self._tokenizer is None:
            raise RuntimeError("引擎未加载，请先调用 load_model()")
        
        # Tokenize 输入
        input_ids = self._tokenizer.encode(prompt, return_tensors="pt")
        
        # 生成参数
        max_new_tokens = kwargs.get("max_new_tokens", kwargs.get("max_tokens", 512))
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.9)
        
        try:
            # TensorRT-LLM 生成
            outputs = self._runner.generate(
                batch_input_ids=[input_ids.squeeze().tolist()],
                max_new_tokens=max_new_tokens,
                end_id=self._tokenizer.eos_token_id,
                pad_id=self._tokenizer.pad_token_id,
                temperature=temperature,
                top_p=top_p,
                streaming=False,  # TRT-LLM 的流式需要特殊处理
            )
            
            # 解码输出
            if outputs is not None and len(outputs) > 0:
                output_ids = outputs[0]
                # 移除输入部分
                new_tokens = output_ids[len(input_ids.squeeze()):]
                generated_text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
                
                # 模拟流式输出
                for char in generated_text:
                    yield char
                    
        except Exception as e:
            LOGGER.error("TensorRT 推理失败: %s", e)
            raise

    def infer_batch(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[str]:
        """
        批量推理（TensorRT-LLM 的强项）。
        
        Args:
            prompts: 输入提示列表
            **kwargs: 生成参数
            
        Returns:
            生成的文本列表
        """
        if self._runner is None or self._tokenizer is None:
            raise RuntimeError("引擎未加载，请先调用 load_model()")
        
        # 批量 Tokenize
        batch_input_ids = [
            self._tokenizer.encode(p, return_tensors="pt").squeeze().tolist()
            for p in prompts
        ]
        
        max_new_tokens = kwargs.get("max_new_tokens", kwargs.get("max_tokens", 512))
        
        outputs = self._runner.generate(
            batch_input_ids=batch_input_ids,
            max_new_tokens=max_new_tokens,
            end_id=self._tokenizer.eos_token_id,
            pad_id=self._tokenizer.pad_token_id,
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
        )
        
        results = []
        for i, output_ids in enumerate(outputs):
            input_len = len(batch_input_ids[i])
            new_tokens = output_ids[input_len:]
            text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
            results.append(text)
            
        return results

    def unload_model(self) -> None:
        """卸载引擎并释放显存。"""
        if self._runner is not None:
            LOGGER.info("正在卸载 TensorRT 引擎...")
            del self._runner
            self._runner = None
            
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
            
        self._model_path = None
        
        gc.collect()
        
        try:
            import torch
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            pass
            
        LOGGER.info("TensorRT 引擎已卸载")

    def get_memory_usage(self) -> Dict[str, float]:
        """返回当前显存使用情况。"""
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                return {
                    "allocated_gb": round(allocated, 2),
                    "reserved_gb": round(reserved, 2),
                    "total_gb": round(total, 2),
                    "utilization": round(allocated / total, 4) if total > 0 else 0,
                }
        except Exception as e:
            LOGGER.warning("获取显存信息失败: %s", e)
            
        return {"allocated_gb": 0, "reserved_gb": 0, "total_gb": 0, "utilization": 0}

    def get_model_info(self) -> Dict[str, Any]:
        """获取当前加载的引擎信息。"""
        return {
            "model_path": self._model_path,
            "engine": "tensorrt-llm",
            "max_batch_size": self.max_batch_size,
            "max_input_len": self.max_input_len,
            "max_output_len": self.max_output_len,
            "is_loaded": self._runner is not None,
            "note": "TensorRT-LLM 不支持动态 LoRA，需要融合后重新编译",
        }

    @staticmethod
    def convert_model_guide() -> str:
        """返回模型转换指南。"""
        return """
TensorRT-LLM 模型转换指南
========================

1. 安装 TensorRT-LLM:
   pip install tensorrt-llm

2. 转换 HuggingFace 模型到 TRT 引擎:
   
   # 以 DeepSeek 为例
   python -m tensorrt_llm.commands.convert_checkpoint \\
       --model_type deepseek \\
       --model_dir /path/to/hf_model \\
       --output_dir /path/to/trt_checkpoint \\
       --dtype float16

3. 构建 TRT 引擎:
   
   trtllm-build \\
       --checkpoint_dir /path/to/trt_checkpoint \\
       --output_dir /path/to/trt_engine \\
       --gemm_plugin float16 \\
       --max_batch_size 8 \\
       --max_input_len 2048 \\
       --max_output_len 512

4. 使用转换后的引擎:
   
   engine = TensorRTEngine()
   engine.load_model("/path/to/trt_engine")

详细文档: https://github.com/NVIDIA/TensorRT-LLM
"""
