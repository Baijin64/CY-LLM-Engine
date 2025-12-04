"""VRAM 显存优化与预检工具。"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
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
    required_gb: float
    available_gb: float
    total_gb: float
    is_safe: bool
    recommendation: str
    suggestions: List[str] = field(default_factory=list)


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

    # 获取可用/总显存
    free_gb, total_gb = get_vram_stats()
    available_gb = free_gb or total_gb

    # 判断是否安全（保留 10% 余量，至少 1GB）
    safety_budget = max(available_gb - 1.0, available_gb * 0.9)
    is_safe = total_per_gpu <= max(safety_budget, 0.0)

    suggestions: List[str] = []
    if not is_safe:
        # 检查是否可以通过张量并行解决
        gpu_count = get_gpu_count()
        if tensor_parallel_size == 1 and gpu_count > 1:
            recommended_tp, tp_reason = recommend_tensor_parallel_size(
                model_weights_gb, total_per_gpu, total_gb, gpu_count
            )
            if recommended_tp > 1:
                suggestions.append(
                    f"🎯 启用 tensor_parallel_size={recommended_tp}（{tp_reason}）"
                )

        # 量化模型推荐
        if quantization is None:
            quant_suggestions = suggest_quantized_models(
                model_name_or_params if isinstance(model_name_or_params, str) else "",
                num_params
            )
            if quant_suggestions:
                suggestions.extend(quant_suggestions)
            else:
                suggestions.append("启用 4-bit 量化（AWQ/GPTQ 或 bitsandbytes）")

        # max_model_len 建议
        if max_model_len > 2048:
            suggestions.append(f"降低 max_model_len 至 2048（当前 {max_model_len}）")

        # 通用建议
        suggestions.append("降低 gpu_memory_utilization 或在配置中腾出更多显存")

        recommendation = (
            f"❌ 显存不足 (需要 {total_per_gpu:.1f}GB, 可用 {available_gb:.1f}GB)"
        )
    else:
        recommendation = "✅ 显存充足，可以加载"

    return VRAMEstimate(
        model_weights_gb=model_weights_gb,
        kv_cache_gb=kv_cache_gb,
        activation_gb=activation_gb,
        overhead_gb=overhead_gb,
        required_gb=total_per_gpu,
        available_gb=available_gb,
        total_gb=total_gb,
        is_safe=is_safe,
        recommendation=recommendation,
        suggestions=suggestions,
    )


def get_vram_stats() -> tuple[float, float]:
    """返回 (free_gb, total_gb)。"""
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        return free / BYTE_PER_GB, total / BYTE_PER_GB
    return 0.0, 0.0


def get_gpu_count() -> int:
    """返回可用的 GPU 数量。"""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def recommend_tensor_parallel_size(
    model_weights_gb: float,
    total_required_gb: float,
    per_gpu_vram_gb: float,
    available_gpus: int
) -> tuple[int, str]:
    """推荐 tensor_parallel_size

    Args:
        model_weights_gb: 模型权重大小（GB）
        total_required_gb: 单卡总需求（GB）
        per_gpu_vram_gb: 单卡显存大小（GB）
        available_gpus: 可用 GPU 数量

    Returns:
        (推荐的 tp_size, 推荐理由)
    """
    if available_gpus <= 1:
        return 1, "只有 1 个 GPU 可用"

    # 如果单卡足够，不需要张量并行
    if total_required_gb <= per_gpu_vram_gb * 0.85:
        return 1, "单卡显存充足"

    # 计算需要多少个 GPU 才能容纳模型权重
    # 权重会被分片，其他部分（KV Cache、激活值）每卡都需要
    min_gpus_for_weights = max(1, int(model_weights_gb / (per_gpu_vram_gb * 0.6)) + 1)

    # 限制在可用 GPU 数量内
    recommended_tp = min(min_gpus_for_weights, available_gpus)

    # 优先选择 2 的幂次
    if recommended_tp > 1:
        power_of_two = 1
        while power_of_two < recommended_tp:
            power_of_two *= 2
        if power_of_two <= available_gpus:
            recommended_tp = power_of_two

    reason = f"模型权重 {model_weights_gb:.1f}GB 需要分片到 {recommended_tp} 个 GPU"
    if recommended_tp < min_gpus_for_weights:
        reason += f"（理想需要 {min_gpus_for_weights} 个，但只有 {available_gpus} 个可用）"

    return recommended_tp, reason


def suggest_quantized_models(model_path: str, num_params: float) -> List[str]:
    """推荐量化版本的模型

    Args:
        model_path: 原始模型路径
        num_params: 模型参数量（单位：10^9）

    Returns:
        量化模型建议列表
    """
    suggestions = []

    # 如果已经是量化模型，不再建议
    quantization_suffixes = ["-awq", "-gptq", "-gguf", "-int4", "-int8", "-fp8"]
    model_lower = model_path.lower()
    if any(suffix in model_lower for suffix in quantization_suffixes):
        return suggestions

    # 大模型（>30B）优先推荐 AWQ/GPTQ
    if num_params >= 30:
        suggestions.append(
            f"💡 考虑使用 AWQ 量化版本（节省 75% 显存）：在 HuggingFace 搜索 '{model_path}-AWQ'"
        )
        suggestions.append(
            f"💡 或使用 GPTQ 量化版本：在 HuggingFace 搜索 '{model_path}-GPTQ'"
        )

    # 中小模型（7B-30B）可以使用 bitsandbytes 或 AWQ
    elif num_params >= 7:
        suggestions.append(
            f"💡 可启用 4-bit 量化节省显存：设置 quantization='bitsandbytes' 或使用预量化模型"
        )

    # 提示常见的量化模型命名规范
    if "/" in model_path:
        org, model_name = model_path.rsplit("/", 1)
        suggestions.append(
            f"提示：量化模型通常命名为 '{org}/{model_name}-AWQ' 或 '{org}/{model_name}-GPTQ'"
        )

    return suggestions


def optimize_vram_config(estimate: VRAMEstimate, current_config: Optional[Dict] = None) -> Dict:
    """根据估算结果优化配置

    Args:
        estimate: VRAM 估算结果
        current_config: 当前配置（可选）

    Returns:
        优化后的配置字典
    """
    optimized = current_config.copy() if current_config else {}

    if not estimate.is_safe:
        # 降低 gpu_memory_utilization
        ratio = estimate.available_gb / max(estimate.required_gb, 1e-6)
        optimized["gpu_memory_utilization"] = max(0.5, min(ratio * 0.7, 0.85))

        # 如果仍然不够，降低 max_model_len
        if ratio < 0.8:
            current_len = optimized.get("max_model_len", 2048)
            new_len = max(1024, current_len // 2)
            optimized["max_model_len"] = new_len

    return optimized


def progressive_retry_configs(base_config: Dict) -> List[Dict]:
    """生成渐进式降级配置列表，用于 OOM 重试

    Args:
        base_config: 基础配置

    Returns:
        配置列表，按保守程度排序
    """
    configs = [base_config.copy()]  # 配置 1: 用户原始配置

    # 配置 2: 降低 gpu_memory_utilization
    config2 = base_config.copy()
    current_util = config2.get("gpu_memory_utilization", 0.75)
    config2["gpu_memory_utilization"] = max(0.5, current_util - 0.10)
    configs.append(config2)

    # 配置 3: 进一步降低 + 减少 max_model_len
    config3 = base_config.copy()
    config3["gpu_memory_utilization"] = 0.60
    config3["max_model_len"] = min(
        config3.get("max_model_len", 4096), 4096
    )
    configs.append(config3)

    # 配置 4: 最保守配置
    config4 = base_config.copy()
    config4["gpu_memory_utilization"] = 0.50
    config4["max_model_len"] = 2048
    configs.append(config4)

    return configs


def suggest_kv_cache_strategy(
    kv_cache_gb: float,
    available_vram_gb: float,
    max_model_len: int,
    current_gpu_util: float,
    expected_qps: Optional[int] = None
) -> List[str]:
    """KV Cache 预分配策略建议

    Args:
        kv_cache_gb: 当前 KV Cache 预估占用
        available_vram_gb: 可用显存
        max_model_len: 最大序列长度
        current_gpu_util: 当前 gpu_memory_utilization 设置
        expected_qps: 预期 QPS（可选）

    Returns:
        KV Cache 优化建议列表
    """
    suggestions = []
    kv_ratio = kv_cache_gb / max(available_vram_gb, 0.1)

    # 基于并发场景的建议
    if expected_qps is not None:
        if expected_qps <= 10:
            # 低并发：可以降低 gpu_memory_utilization
            if current_gpu_util > 0.70:
                suggestions.append(
                    f"🎯 低并发场景 (QPS≤10)：建议降低 gpu_memory_utilization 至 0.70 "
                    f"（当前 {current_gpu_util:.2f}）以节省显存"
                )
        elif expected_qps <= 50:
            # 中并发：推荐 0.75
            if current_gpu_util < 0.70 or current_gpu_util > 0.80:
                suggestions.append(
                    f"🎯 中并发场景 (QPS 10-50)：建议设置 gpu_memory_utilization=0.75 "
                    f"（当前 {current_gpu_util:.2f}）"
                )
        else:
            # 高并发：推荐 0.85
            if current_gpu_util < 0.80:
                suggestions.append(
                    f"🎯 高并发场景 (QPS>50)：建议提升 gpu_memory_utilization 至 0.85 "
                    f"（当前 {current_gpu_util:.2f}）以支持更多并发请求"
                )

    # 基于序列长度的建议
    if max_model_len <= 2048:
        suggestions.append(
            f"✅ max_model_len={max_model_len} 较小，KV Cache 占用低，适合高并发"
        )
    elif max_model_len <= 8192:
        suggestions.append(
            f"⚡ max_model_len={max_model_len}：平衡配置，"
            f"KV Cache 占用 {kv_cache_gb:.1f}GB ({kv_ratio*100:.0f}% 显存)"
        )
    else:
        suggestions.append(
            f"⚠️  max_model_len={max_model_len} 较大，"
            f"KV Cache 占用 {kv_cache_gb:.1f}GB ({kv_ratio*100:.0f}% 显存)，"
            "高并发时需要监控显存使用率"
        )
        if kv_ratio > 0.5:
            suggestions.append(
                "💡 考虑降低 max_model_len 或启用 Prefix Caching 以优化长序列场景"
            )

    # KV Cache dtype 优化建议
    if kv_cache_gb > 5.0:
        suggestions.append(
            f"💡 KV Cache 占用 {kv_cache_gb:.1f}GB 较大，"
            "可考虑设置 kv_cache_dtype='fp8' 以节省 50% KV Cache 显存（略微损失精度）"
        )

    return suggestions


def suggest_batch_optimization(
    engine_type: str,
    kv_cache_gb: float,
    available_vram_gb: float,
    max_model_len: int
) -> List[str]:
    """生成批处理优化建议

    Args:
        engine_type: 引擎类型
        kv_cache_gb: KV Cache 显存占用
        available_vram_gb: 可用显存
        max_model_len: 最大序列长度

    Returns:
        批处理优化建议列表
    """
    suggestions = []
    engine_lower = engine_type.lower()

    # vLLM 批处理建议
    if "vllm" in engine_lower:
        suggestions.append(
            "⚡ vLLM Continuous Batching 自动启用，无需手动配置"
        )

        # 如果 KV Cache 占用较小，可以增加并发
        if kv_cache_gb < available_vram_gb * 0.3:
            suggestions.append(
                f"💡 KV Cache 仅占用 {kv_cache_gb:.1f}GB，"
                f"可支持高并发请求（建议配置网关层的并发限制）"
            )

        # 长序列场景的建议
        if max_model_len > 8192:
            suggestions.append(
                f"⚠️  max_model_len={max_model_len} 较大，"
                "高并发时 KV Cache 占用会显著增加，建议监控显存使用率"
            )

    # TensorRT-LLM 批处理建议
    elif "trt" in engine_lower or "tensorrt" in engine_lower:
        # 估算可支持的批处理大小
        # KV Cache 是按 max_batch_size * max_model_len 预分配的
        estimated_max_batch = max(1, int(available_vram_gb * 0.6 / max(kv_cache_gb, 0.1)))

        suggestions.append(
            f"🎯 TensorRT Inflight Batching: 建议设置 max_batch_size={min(estimated_max_batch, 32)}"
        )
        suggestions.append(
            "💡 TRT 批处理需要在构建引擎时指定 max_batch_size，"
            "运行时无法动态调整"
        )

        if estimated_max_batch > 16:
            suggestions.append(
                f"✨ 显存充足，可支持最多 {estimated_max_batch} 的批处理（推荐 16-32）"
            )

    # Nvidia (transformers) 批处理建议
    elif "nvidia" in engine_lower or engine_type == "cuda":
        suggestions.append(
            "💡 transformers 引擎支持简单批处理，"
            "但性能不如 vLLM/TRT 的动态批处理"
        )
        suggestions.append(
            "🎯 高并发场景建议切换到 vLLM 引擎以获得更好的吞吐量"
        )

    return suggestions


def format_vram_report(estimate: VRAMEstimate, verbose: bool = True) -> str:
    """格式化 VRAM 估算报告

    Args:
        estimate: VRAM 估算结果
        verbose: 是否显示详细信息

    Returns:
        格式化的报告字符串
    """
    lines = []

    if verbose:
        lines.append("显存需求估算:")
        lines.append(f"  模型权重: {estimate.model_weights_gb:.2f} GB")
        lines.append(f"  KV Cache:  {estimate.kv_cache_gb:.2f} GB")
        lines.append(f"  激活值:    {estimate.activation_gb:.2f} GB")
        lines.append(f"  框架开销:  {estimate.overhead_gb:.2f} GB")
        lines.append(f"  总计:      {estimate.required_gb:.2f} GB")
        lines.append(f"  可用显存:  {estimate.available_gb:.2f} GB")
        lines.append("")

    lines.append(estimate.recommendation)

    if estimate.suggestions:
        lines.append("\n建议:")
        for suggestion in estimate.suggestions:
            lines.append(f"  - {suggestion}")

    return "\n".join(lines)
