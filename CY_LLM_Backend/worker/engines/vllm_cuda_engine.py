"""
vllm_cuda_engine.py
[推理引擎] NVIDIA CUDA + vLLM 实现
说明：基于 vLLM 的高性能推理引擎，支持 PagedAttention、Continuous Batching 等特性。

特点：
  - PagedAttention：显存利用率高，支持更长上下文
  - Continuous Batching：动态批处理，高并发吞吐
  - 多 LoRA 热切换：支持同一基座模型挂载多个适配器
  - OpenAI 兼容：可直接使用 vLLM 的 API Server

依赖：
  pip install vllm
"""

from __future__ import annotations

import gc
import logging
from typing import Any, Dict, Generator, List, Optional

from .abstract_engine import BaseEngine

LOGGER = logging.getLogger("cy_llm.worker.engines.vllm_cuda")

# 延迟导入 vLLM，仅在实际使用时加载
_vllm_imported = False
_LLM = None
_SamplingParams = None


def _ensure_vllm_imported():
    """确保 vLLM 已导入"""
    global _vllm_imported, _LLM, _SamplingParams
    if not _vllm_imported:
        try:
            from vllm import LLM, SamplingParams
            _LLM = LLM
            _SamplingParams = SamplingParams
            _vllm_imported = True
            LOGGER.info("vLLM 模块加载成功")
        except ImportError as e:
            raise ImportError(
                "vLLM 未安装。请运行: pip install vllm\n"
                f"原始错误: {e}"
            ) from e


class VllmCudaEngine(BaseEngine):
    """
    基于 vLLM 的 NVIDIA CUDA 推理引擎。
    
    特点：
      - 高性能：PagedAttention + Continuous Batching
      - 多 LoRA：支持动态切换多个 LoRA 适配器
      - 量化支持：AWQ, GPTQ, FP8 等
      
    使用示例：
        >>> engine = VllmCudaEngine()
        >>> engine.load_model("deepseek-ai/deepseek-llm-7b-chat")
        >>> for token in engine.infer("你好"):
        ...     print(token, end="", flush=True)
    """

    def __init__(
        self,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.75,  # 修改: 0.90 -> 0.75
        max_model_len: Optional[int] = None,
        quantization: Optional[str] = None,
        enable_lora: bool = True,
        max_loras: int = 4,
        enable_prefix_caching: bool = False,
        kv_cache_dtype: Optional[str] = None,
        allow_auto_tuning: bool = True,
        **kwargs
    ) -> None:
        """
        初始化 vLLM CUDA 引擎。

        Args:
            tensor_parallel_size: 张量并行数（多卡时使用）
            gpu_memory_utilization: GPU 显存使用率上限（默认 0.75，推荐 0.70-0.85）
            max_model_len: 最大序列长度，None 则自动检测
            quantization: 量化方法，如 "awq", "gptq", "fp8"（vLLM 不支持 bitsandbytes）
            enable_lora: 是否启用 LoRA 支持
            max_loras: 最大同时加载的 LoRA 数量
            enable_prefix_caching: 启用前缀缓存（Automatic Prefix Caching）
            kv_cache_dtype: KV Cache 数据类型，"auto" 或 "fp8"
            allow_auto_tuning: 是否允许自动调整参数以避免 OOM（默认 True）
        """
        _ensure_vllm_imported()

        self.allow_auto_tuning = allow_auto_tuning

        # 验证 gpu_memory_utilization 安全性
        if gpu_memory_utilization > 0.90:
            if allow_auto_tuning:
                LOGGER.warning(
                    "gpu_memory_utilization=%.2f 过高，可能导致 OOM。自动调整为 0.85",
                    gpu_memory_utilization
                )
                gpu_memory_utilization = 0.85
            else:
                LOGGER.warning(
                    "gpu_memory_utilization=%.2f 过高，可能导致 OOM（allow_auto_tuning=False，保持原值）",
                    gpu_memory_utilization
                )

        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.quantization = quantization
        self.enable_lora = enable_lora
        self.max_loras = max_loras
        self.enable_prefix_caching = enable_prefix_caching
        self.kv_cache_dtype = kv_cache_dtype
        self.extra_kwargs = kwargs

        self._llm: Optional[Any] = None  # vLLM LLM 实例
        self._model_path: Optional[str] = None
        self._loaded_loras: Dict[str, str] = {}  # lora_name -> lora_path

        LOGGER.info(
            "VllmCudaEngine 初始化: tp=%d, mem=%.2f, quant=%s, lora=%s, prefix_cache=%s, kv_dtype=%s",
            tensor_parallel_size, gpu_memory_utilization, quantization, enable_lora,
            enable_prefix_caching, kv_cache_dtype
        )

    def load_model(
        self,
        model_path: str,
        adapter_path: Optional[str] = None,
        skip_vram_check: bool = False,
        **kwargs
    ) -> None:
        """
        加载模型到 GPU。
        
        Args:
            model_path: HuggingFace 模型 ID 或本地路径
            adapter_path: LoRA 适配器路径（可选）
            skip_vram_check: 跳过 VRAM 预检查（默认 False）
            **kwargs: 额外参数传递给 vLLM
                - max_model_len: 覆盖实例配置的最大序列长度
                - tensor_parallel_size: 覆盖实例配置的张量并行数
                - enable_prefix_caching: 覆盖实例配置的前缀缓存
                - kv_cache_dtype: 覆盖实例配置的 KV Cache 数据类型
                - gpu_memory_utilization: 覆盖实例配置的显存利用率
        """
        LOGGER.info("正在加载模型: %s", model_path)
        
        # === VRAM 预检查 ===
        if not skip_vram_check:
            try:
                from worker.utils.vram_optimizer import (
                    estimate_vram_requirements,
                    optimize_vram_config,
                    format_vram_report,
                )

                estimate = estimate_vram_requirements(
                    model_name_or_params=model_path,
                    max_model_len=self.max_model_len or 2048,
                    dtype="fp16",
                    quantization=self.quantization,
                    engine_type="vllm",
                    tensor_parallel_size=self.tensor_parallel_size,
                )

                # 显示详细的 VRAM 估算报告
                LOGGER.info("\n%s", format_vram_report(estimate, verbose=True))

                if not estimate.is_safe:
                    if not self.allow_auto_tuning:
                        LOGGER.warning(
                            "⚠️  VRAM 不足，但 allow_auto_tuning=False，保持原始配置加载"
                        )
                        # 仍显示建议，但不自动调整
                        if estimate.suggestions:
                            LOGGER.warning("💡 建议:")
                            for suggestion in estimate.suggestions:
                                LOGGER.warning("   - %s", suggestion)
                    else:
                        # 尝试优化配置
                        current_config = {
                            "gpu_memory_utilization": self.gpu_memory_utilization,
                            "max_model_len": self.max_model_len or 2048,
                        }
                        optimized = optimize_vram_config(estimate, current_config)

                        # 应用优化后的配置
                        if "gpu_memory_utilization" in optimized:
                            old_util = self.gpu_memory_utilization
                            self.gpu_memory_utilization = optimized["gpu_memory_utilization"]
                            LOGGER.warning(
                                "⚙️  自动调整 gpu_memory_utilization: %.2f -> %.2f",
                                old_util, self.gpu_memory_utilization
                            )

                        if "max_model_len" in optimized and self.max_model_len != optimized["max_model_len"]:
                            old_len = self.max_model_len or 2048
                            self.max_model_len = optimized["max_model_len"]
                            LOGGER.warning(
                                "⚙️  自动调整 max_model_len: %d -> %d",
                                old_len, self.max_model_len
                            )

                        # 显示优化建议
                        if estimate.suggestions:
                            LOGGER.warning("💡 其他建议:")
                            for suggestion in estimate.suggestions:
                                LOGGER.warning("   - %s", suggestion)

            except ImportError:
                LOGGER.warning("vram_optimizer 未找到，跳过 VRAM 检查")
        
        # 从 kwargs 提取可覆盖的配置
        max_model_len = kwargs.pop("max_model_len", None) or self.max_model_len
        tensor_parallel_size = kwargs.pop("tensor_parallel_size", None) or self.tensor_parallel_size
        enable_prefix_caching = kwargs.pop("enable_prefix_caching", None)
        if enable_prefix_caching is None:
            enable_prefix_caching = self.enable_prefix_caching
        kv_cache_dtype = kwargs.pop("kv_cache_dtype", None) or self.kv_cache_dtype
        gpu_memory_utilization = kwargs.pop("gpu_memory_utilization", None) or self.gpu_memory_utilization
        
        # 合并配置
        llm_kwargs = {
            "model": model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": True,
            "dtype": "auto",
        }
        
        if max_model_len:
            llm_kwargs["max_model_len"] = max_model_len

        if self.enable_lora:
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_loras"] = self.max_loras
            llm_kwargs["max_lora_rank"] = kwargs.get("max_lora_rank", 64)
        
        # KV Cache 优化配置
        if enable_prefix_caching:
            llm_kwargs["enable_prefix_caching"] = True
            LOGGER.info("启用 Automatic Prefix Caching (APC)")
            
        if kv_cache_dtype:
            llm_kwargs["kv_cache_dtype"] = kv_cache_dtype
            LOGGER.info("KV Cache 数据类型: %s", kv_cache_dtype)
            
        # 量化配置（vLLM 支持 AWQ/GPTQ/FP8）
        if self.quantization:
            valid_vllm_quant = ["awq", "gptq", "fp8", "fp8_e5m2"]
            if self.quantization.lower() in valid_vllm_quant:
                llm_kwargs["quantization"] = self.quantization
                LOGGER.info("使用量化方法: %s", self.quantization)
            else:
                raise ValueError(
                    f"vLLM 不支持量化方法 '{self.quantization}'。"
                    f"支持的方法: {', '.join(valid_vllm_quant)}。"
                    f"如需使用 bitsandbytes，请切换到 nvidia 引擎。"
                )

        # 合并额外参数
        llm_kwargs.update(self.extra_kwargs)
        llm_kwargs.update(kwargs)

        # === OOM 自动重试机制 ===
        # 生成渐进式降级配置
        base_config = {
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
        }

        try:
            from worker.utils.vram_optimizer import progressive_retry_configs
            retry_configs = progressive_retry_configs(base_config)
        except ImportError:
            # 如果 vram_optimizer 不可用，只使用原始配置
            retry_configs = [base_config]

        last_error = None
        for attempt, config in enumerate(retry_configs, start=1):
            try:
                # 应用当前配置到 llm_kwargs
                current_kwargs = llm_kwargs.copy()
                current_kwargs["gpu_memory_utilization"] = config.get(
                    "gpu_memory_utilization", gpu_memory_utilization
                )
                if "max_model_len" in config and config["max_model_len"]:
                    current_kwargs["max_model_len"] = config["max_model_len"]

                if attempt > 1:
                    LOGGER.warning(
                        "🔄 OOM 重试 [%d/%d]: gpu_mem_util=%.2f, max_model_len=%s",
                        attempt,
                        len(retry_configs),
                        current_kwargs["gpu_memory_utilization"],
                        current_kwargs.get("max_model_len", "auto"),
                    )

                # 加载进度反馈
                LOGGER.info("⏳ 开始加载模型（这可能需要几分钟，请耐心等待）...")
                LOGGER.info("📥 步骤 1/3: 初始化 vLLM 引擎配置")

                import time
                start_time = time.time()

                # 尝试加载模型
                LOGGER.info("📦 步骤 2/3: 加载模型权重到 GPU 显存")
                self._llm = _LLM(**current_kwargs)

                LOGGER.info("🔧 步骤 3/3: 初始化 KV Cache 和推理引擎")
                elapsed = time.time() - start_time
                LOGGER.info(f"✅ 模型加载完成，耗时 {elapsed:.1f} 秒")
                self._model_path = model_path

                if attempt > 1:
                    LOGGER.info("✅ 模型加载成功（第 %d 次尝试）: %s", attempt, model_path)
                else:
                    LOGGER.info("模型加载成功: %s", model_path)

                # 如果提供了 LoRA 适配器，加载它
                if adapter_path:
                    self.load_lora(adapter_path, "default")

                # 成功加载，退出重试循环
                break

            except Exception as e:
                last_error = e
                error_msg = str(e).lower()

                # 检查是否是 OOM 错误
                is_oom = (
                    "out of memory" in error_msg
                    or "cuda error" in error_msg
                    or "cuda out of memory" in error_msg
                    or "allocate" in error_msg
                )

                if is_oom and attempt < len(retry_configs):
                    LOGGER.warning("⚠️  显存不足 (OOM)，准备使用更保守的配置重试...")
                    # 清理显存
                    try:
                        import torch
                        torch.cuda.empty_cache()
                        gc.collect()
                    except Exception:
                        pass
                    # 继续下一次重试
                    continue
                else:
                    # 非 OOM 错误，或已用尽所有重试配置
                    if attempt == len(retry_configs):
                        LOGGER.error(
                            "❌ 所有 %d 个配置均失败，无法加载模型: %s",
                            len(retry_configs),
                            last_error,
                        )
                    else:
                        LOGGER.error("模型加载失败: %s", e)
                    raise

    def load_lora(self, adapter_path: str, lora_name: str = "default") -> None:
        """
        加载 LoRA 适配器。
        
        Args:
            adapter_path: LoRA 适配器路径
            lora_name: 适配器名称（用于后续切换）
        """
        if not self.enable_lora:
            raise RuntimeError("LoRA 未启用，请在初始化时设置 enable_lora=True")
            
        if self._llm is None:
            raise RuntimeError("请先调用 load_model() 加载基座模型")
            
        LOGGER.info("加载 LoRA 适配器: %s as '%s'", adapter_path, lora_name)
        # vLLM 的 LoRA 通过请求时指定，这里只记录路径
        self._loaded_loras[lora_name] = adapter_path

    def infer(
        self,
        prompt: str,
        lora_name: Optional[str] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        流式推理。
        
        Args:
            prompt: 输入提示
            lora_name: 使用的 LoRA 适配器名称
            **kwargs: 生成参数（temperature, top_p, max_tokens 等）
            
        Yields:
            生成的文本 token
        """
        if self._llm is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")
            
        # 构建采样参数
        sampling_params = _SamplingParams(
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            max_tokens=kwargs.get("max_new_tokens", kwargs.get("max_tokens", 512)),
            repetition_penalty=kwargs.get("repetition_penalty", 1.1),
            stop=kwargs.get("stop", None),
        )
        
        # 构建请求参数
        request_kwargs = {"sampling_params": sampling_params}
        
        # 如果指定了 LoRA
        if lora_name and lora_name in self._loaded_loras:
            from vllm.lora.request import LoRARequest
            lora_path = self._loaded_loras[lora_name]
            # vLLM 使用 LoRARequest 指定适配器
            request_kwargs["lora_request"] = LoRARequest(
                lora_name=lora_name,
                lora_int_id=hash(lora_name) % (2**31),
                lora_local_path=lora_path,
            )
        
        # vLLM 的 generate 是同步的，但返回完整输出
        # 为了兼容流式接口，我们逐字符 yield
        # 注意：vLLM 本身支持真正的流式，但需要使用 AsyncLLMEngine
        # 这里简化处理，后续可升级为 AsyncLLMEngine
        outputs = self._llm.generate([prompt], **request_kwargs)
        
        if outputs and len(outputs) > 0:
            generated_text = outputs[0].outputs[0].text
            # 模拟流式输出（实际生产中应使用 AsyncLLMEngine）
            for char in generated_text:
                yield char

    def infer_batch(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[str]:
        """
        批量推理（vLLM 的强项）。
        
        Args:
            prompts: 输入提示列表
            **kwargs: 生成参数
            
        Returns:
            生成的文本列表
        """
        if self._llm is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")
            
        sampling_params = _SamplingParams(
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            max_tokens=kwargs.get("max_new_tokens", kwargs.get("max_tokens", 512)),
            repetition_penalty=kwargs.get("repetition_penalty", 1.1),
        )
        
        outputs = self._llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]

    def unload_model(self) -> None:
        """卸载模型并释放显存。"""
        if self._llm is not None:
            LOGGER.info("正在卸载模型...")
            del self._llm
            self._llm = None
            self._model_path = None
            self._loaded_loras.clear()
            
            # 强制垃圾回收
            gc.collect()
            
            try:
                import torch
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass
                
            LOGGER.info("模型已卸载")

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
        """获取当前加载的模型信息。"""
        return {
            "model_path": self._model_path,
            "engine": "vllm-cuda",
            "tensor_parallel_size": self.tensor_parallel_size,
            "quantization": self.quantization,
            "loaded_loras": list(self._loaded_loras.keys()),
            "is_loaded": self._llm is not None,
        }
