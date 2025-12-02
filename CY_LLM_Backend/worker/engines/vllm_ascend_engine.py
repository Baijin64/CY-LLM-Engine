"""
vllm_ascend_engine.py
[推理引擎] 华为 Ascend NPU + vLLM-Ascend 实现
说明：基于 vLLM-Ascend（华为官方 fork）的高性能推理引擎。

特点：
  - 华为 NPU 原生支持：基于 CANN 和 Ascend 硬件
  - 与 vLLM 接口兼容：迁移成本低
  - PagedAttention：显存效率高

依赖：
  - CANN 环境（华为 Ascend 驱动和运行时）
  - pip install vllm-ascend  # 或从华为官方获取
"""

from __future__ import annotations

import gc
import logging
from typing import Any, Dict, Generator, List, Optional

from .abstract_engine import BaseEngine

LOGGER = logging.getLogger("ew.worker.engines.vllm_ascend")

# 延迟导入标记
_vllm_ascend_imported = False
_LLM = None
_SamplingParams = None


def _ensure_vllm_ascend_imported():
    """确保 vLLM-Ascend 已导入"""
    global _vllm_ascend_imported, _LLM, _SamplingParams
    if not _vllm_ascend_imported:
        try:
            # vLLM-Ascend 的导入路径
            # 注意：华为的 fork 可能使用不同的包名
            from vllm import LLM, SamplingParams
            _LLM = LLM
            _SamplingParams = SamplingParams
            _vllm_ascend_imported = True
            LOGGER.info("vLLM-Ascend 模块加载成功")
        except ImportError as e:
            raise ImportError(
                "vLLM-Ascend 未安装。请参考华为官方文档安装：\n"
                "https://gitee.com/ascend/vllm-ascend\n"
                "确保已正确配置 CANN 环境。\n"
                f"原始错误: {e}"
            ) from e


class VllmAscendEngine(BaseEngine):
    """
    基于 vLLM-Ascend 的华为 NPU 推理引擎。
    
    特点：
      - 华为昇腾 NPU 原生支持
      - PagedAttention 显存优化
      - 与标准 vLLM 接口兼容
      
    前置条件：
      - 安装 CANN 驱动和运行时
      - 安装 torch_npu
      - 安装 vllm-ascend
      
    使用示例：
        >>> engine = VllmAscendEngine()
        >>> engine.load_model("deepseek-ai/deepseek-llm-7b-chat")
        >>> for token in engine.infer("你好"):
        ...     print(token, end="", flush=True)
    """

    def __init__(
        self,
        tensor_parallel_size: int = 1,
        npu_memory_utilization: float = 0.90,
        max_model_len: Optional[int] = None,
        enable_lora: bool = True,
        max_loras: int = 4,
        **kwargs
    ) -> None:
        """
        初始化 vLLM-Ascend 引擎。
        
        Args:
            tensor_parallel_size: 张量并行数（多卡时使用）
            npu_memory_utilization: NPU 显存使用率上限
            max_model_len: 最大序列长度，None 则自动检测
            enable_lora: 是否启用 LoRA 支持
            max_loras: 最大同时加载的 LoRA 数量
        """
        _ensure_vllm_ascend_imported()
        
        self.tensor_parallel_size = tensor_parallel_size
        self.npu_memory_utilization = npu_memory_utilization
        self.max_model_len = max_model_len
        self.enable_lora = enable_lora
        self.max_loras = max_loras
        self.extra_kwargs = kwargs
        
        self._llm: Optional[Any] = None
        self._model_path: Optional[str] = None
        self._loaded_loras: Dict[str, str] = {}
        
        LOGGER.info(
            "VllmAscendEngine 初始化: tp=%d, mem=%.2f, lora=%s",
            tensor_parallel_size, npu_memory_utilization, enable_lora
        )

    def load_model(
        self,
        model_path: str,
        adapter_path: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        加载模型到 NPU。
        
        Args:
            model_path: HuggingFace 模型 ID 或本地路径
            adapter_path: LoRA 适配器路径（可选）
            **kwargs: 额外参数
        """
        LOGGER.info("正在加载模型到 Ascend NPU: %s", model_path)
        
        llm_kwargs = {
            "model": model_path,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.npu_memory_utilization,  # vLLM-Ascend 使用相同参数名
            "trust_remote_code": True,
            "dtype": "auto",
            "device": "npu",  # 指定使用 NPU
        }
        
        if self.max_model_len:
            llm_kwargs["max_model_len"] = self.max_model_len
            
        if self.enable_lora:
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_loras"] = self.max_loras
            llm_kwargs["max_lora_rank"] = kwargs.get("max_lora_rank", 64)
        
        llm_kwargs.update(self.extra_kwargs)
        llm_kwargs.update(kwargs)
        
        try:
            self._llm = _LLM(**llm_kwargs)
            self._model_path = model_path
            LOGGER.info("模型加载成功: %s", model_path)
            
            if adapter_path:
                self.load_lora(adapter_path, "default")
                
        except Exception as e:
            LOGGER.error("模型加载失败: %s", e)
            raise

    def load_lora(self, adapter_path: str, lora_name: str = "default") -> None:
        """加载 LoRA 适配器。"""
        if not self.enable_lora:
            raise RuntimeError("LoRA 未启用")
            
        if self._llm is None:
            raise RuntimeError("请先加载基座模型")
            
        LOGGER.info("加载 LoRA: %s as '%s'", adapter_path, lora_name)
        self._loaded_loras[lora_name] = adapter_path

    def infer(
        self,
        prompt: str,
        lora_name: Optional[str] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """流式推理。"""
        if self._llm is None:
            raise RuntimeError("模型未加载")
            
        sampling_params = _SamplingParams(
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            max_tokens=kwargs.get("max_new_tokens", kwargs.get("max_tokens", 512)),
            repetition_penalty=kwargs.get("repetition_penalty", 1.1),
            stop=kwargs.get("stop", None),
        )
        
        request_kwargs = {"sampling_params": sampling_params}
        
        if lora_name and lora_name in self._loaded_loras:
            try:
                from vllm.lora.request import LoRARequest
                lora_path = self._loaded_loras[lora_name]
                request_kwargs["lora_request"] = LoRARequest(
                    lora_name=lora_name,
                    lora_int_id=hash(lora_name) % (2**31),
                    lora_local_path=lora_path,
                )
            except ImportError:
                LOGGER.warning("LoRA 请求模块不可用，将忽略 LoRA 设置")
        
        outputs = self._llm.generate([prompt], **request_kwargs)
        
        if outputs and len(outputs) > 0:
            generated_text = outputs[0].outputs[0].text
            for char in generated_text:
                yield char

    def infer_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """批量推理。"""
        if self._llm is None:
            raise RuntimeError("模型未加载")
            
        sampling_params = _SamplingParams(
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            max_tokens=kwargs.get("max_new_tokens", kwargs.get("max_tokens", 512)),
            repetition_penalty=kwargs.get("repetition_penalty", 1.1),
        )
        
        outputs = self._llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]

    def unload_model(self) -> None:
        """卸载模型。"""
        if self._llm is not None:
            LOGGER.info("正在卸载模型...")
            del self._llm
            self._llm = None
            self._model_path = None
            self._loaded_loras.clear()
            
            gc.collect()
            
            try:
                import torch_npu
                torch_npu.npu.empty_cache()
            except Exception:
                pass
                
            LOGGER.info("模型已卸载")

    def get_memory_usage(self) -> Dict[str, float]:
        """返回 NPU 显存使用情况。"""
        try:
            import torch_npu
            allocated = torch_npu.npu.memory_allocated() / (1024 ** 3)
            reserved = torch_npu.npu.memory_reserved() / (1024 ** 3)
            # NPU 获取总显存的方式可能不同
            total = reserved * 1.2 if reserved > 0 else 32  # 估算值
            return {
                "allocated_gb": round(allocated, 2),
                "reserved_gb": round(reserved, 2),
                "total_gb": round(total, 2),
                "utilization": round(allocated / total, 4) if total > 0 else 0,
                "device": "ascend_npu",
            }
        except Exception as e:
            LOGGER.warning("获取 NPU 显存信息失败: %s", e)
            
        return {"allocated_gb": 0, "reserved_gb": 0, "total_gb": 0, "utilization": 0, "device": "ascend_npu"}

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息。"""
        return {
            "model_path": self._model_path,
            "engine": "vllm-ascend",
            "tensor_parallel_size": self.tensor_parallel_size,
            "loaded_loras": list(self._loaded_loras.keys()),
            "is_loaded": self._llm is not None,
            "device": "ascend_npu",
        }
