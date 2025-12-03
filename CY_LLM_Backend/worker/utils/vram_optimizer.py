"""VRAM 显存优化工具"""

from dataclasses import dataclass
from typing import Optional
import torch
import re

BYTE_PER_GB = 1024 ** 3


@dataclass
class VRAMEstimate:
    """显存估算结果"""
    model_weights_gb: float
    kv_cache_gb: float
    activation_gb: float
    overhead_gb: float
    total_per_gpu: float
    available_gb: float
    is_safe: bool
    recommendation: str


def extract_param_count(model_name_or_path: str) -> float:
    """从模型名称提取参数量（单位：10^9）"""
    # 匹配 "7B", "13B", "72B" 等
    match = re.search(r'(\d+\.?\d*)[Bb]', model_name_or_path)
    if match:
        return float(match.group(1))
    # 默认假设 7B
    return 7.0


def estimate_model_weights(num_params: float, dtype: str, quantization: Optional[str]) -> float:
    """估算模型权重显存占用（GB）"""
    dtype_bytes = {
        "fp32": 4, "fp16": 2, "bf16": 2,
        "fp8": 1, "int8": 1, "int4": 0.5
    }

    if quantization in ["awq", "gptq", "bitsandbytes"]:
        bytes_per_param = 0.5  # 4-bit
    elif quantization in ["fp8", "fp8_e5m2"]:
        bytes_per_param = 1
    else:
        bytes_per_param = dtype_bytes.get(dtype, 2)

    return num_params * bytes_per_param


def estimate_kv_cache(
    num_params: float,
    max_model_len: int,
    dtype: str = "fp16",
    tensor_parallel_size: int = 1
) -> float:
    """估算 KV Cache 显存占用（GB）"""
    # 简化公式：2 * num_layers * max_model_len * hidden_size * bytes
    # 假设 num_layers ≈ sqrt(num_params * 1e9 / 8192)
    # hidden_size ≈ 4096 for 7B, 8192 for 72B
    hidden_size = min(4096 + (num_params - 7) * 100, 8192)
    num_layers = int((num_params * 1e9 / hidden_size / hidden_size) ** 0.5)

    dtype_bytes = {"fp32": 4, "fp16": 2, "fp8": 1}
    bytes_per_elem = dtype_bytes.get(dtype, 2)

    kv_cache_bytes = 2 * num_layers * max_model_len * hidden_size * bytes_per_elem
    return kv_cache_bytes / BYTE_PER_GB / tensor_parallel_size


def estimate_vram_requirements(
    model_name_or_params: str | float,
    max_model_len: int = 2048,
    dtype: str = "fp16",
    quantization: Optional[str] = None,
    engine_type: str = "vllm",
    tensor_parallel_size: int = 1,
) -> VRAMEstimate:
    """估算模型加载所需的 VRAM"""
    # 提取参数量
    if isinstance(model_name_or_params, str):
        num_params = extract_param_count(model_name_or_params)
    else:
        num_params = model_name_or_params

    # 计算各部分占用
    model_weights_gb = estimate_model_weights(num_params, dtype, quantization)
    kv_cache_gb = estimate_kv_cache(num_params, max_model_len, dtype, tensor_parallel_size)
    activation_gb = num_params * 0.15  # 经验值：15% 的模型大小

    # 框架开销
    overhead_map = {"vllm": 2.0, "trt": 1.5, "nvidia": 1.0}
    overhead_gb = overhead_map.get(engine_type, 1.5)

    # 总计（考虑张量并行）
    total_per_gpu = (
        model_weights_gb / tensor_parallel_size +
        kv_cache_gb +
        activation_gb +
        overhead_gb
    )

    # 获取可用显存
    available_gb = get_available_vram()

    # 判断是否安全（保留 15% 余量）
    is_safe = total_per_gpu < available_gb * 0.85

    # 生成建议
    if is_safe:
        recommendation = "✅ 显存充足，可以加载"
    else:
        suggestions = []
        if quantization is None:
            suggestions.append("启用 4-bit 量化（AWQ/GPTQ）")
        if max_model_len > 2048:
            suggestions.append(f"降低 max_model_len 至 2048（当前 {max_model_len}）")
        if tensor_parallel_size == 1 and available_gb < total_per_gpu / 2:
            suggestions.append("使用多 GPU 张量并行")
        recommendation = f"❌ 显存不足 (需要 {total_per_gpu:.1f}GB, 可用 {available_gb:.1f}GB)\n建议: " + "; ".join(suggestions)

    return VRAMEstimate(
        model_weights_gb=model_weights_gb,
        kv_cache_gb=kv_cache_gb,
        activation_gb=activation_gb,
        overhead_gb=overhead_gb,
        total_per_gpu=total_per_gpu,
        available_gb=available_gb,
        is_safe=is_safe,
        recommendation=recommendation
    )


def get_available_vram() -> float:
    """获取可用 VRAM（GB）"""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / BYTE_PER_GB
    return 0.0


def optimize_vram_config(estimate: VRAMEstimate) -> dict:
    """根据估算结果优化配置"""
    config = {}

    if not estimate.is_safe:
        # 降低 gpu_memory_utilization
        ratio = estimate.available_gb / estimate.total_per_gpu
        config["gpu_memory_utilization"] = max(0.5, ratio * 0.7)

    return config
