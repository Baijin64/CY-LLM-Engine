"""Utility helpers for VRAM and dependency diagnostics."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

try:
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover - py<3.8
    import importlib_metadata

from packaging.version import InvalidVersion, parse

from ..config.config_loader import ModelSpec
from .vram_optimizer import estimate_vram_requirements, get_vram_stats

LOGGER = logging.getLogger("cy_llm.worker.diagnostic")
BYTE_PER_GB = 1024 ** 3


@dataclass
class VRAMDiagnosticReport:
    model_id: str
    model_params_billion: float
    kv_cache_gb: float
    required_vram_gb: float
    available_vram_gb: float
    success: bool
    suggestions: List[str]
    message: str


@dataclass
class DependencyStatus:
    name: str
    installed: Optional[str]
    required: str
    status: str  # ok/missing/conflict
    detail: str


def estimate_model_params(model_spec: ModelSpec) -> float:
    """从模型名称/路径猜测参数规模（单位：10^9）。"""
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*[bB]\b", model_spec.model_path)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    if "7b" in model_spec.model_path.lower():
        return 7.0
    if "13b" in model_spec.model_path.lower():
        return 13.0
    if "3b" in model_spec.model_path.lower():
        return 3.0
    return 7.0


def estimate_kv_cache(max_model_len: int, tp_size: int) -> float:
    """简化估算 KV Cache 的显存占用（单位 GB）。"""
    kv_cache_bytes = max_model_len * 4096 * max(tp_size, 1) * 4 * 2
    return kv_cache_bytes / BYTE_PER_GB


def get_available_vram_gb() -> float:
    free, _ = get_vram_stats()
    return free


def get_total_vram_gb() -> float:
    _, total = get_vram_stats()
    return total


def check_vram_for_model(model_id: str, model_spec: ModelSpec) -> VRAMDiagnosticReport:
    quantization = model_spec.quantization
    if quantization is None and model_spec.use_4bit:
        quantization = "bitsandbytes"

    max_len = model_spec.max_model_len or 8192
    tp_size = model_spec.tensor_parallel_size or 1
    engine_value = getattr(model_spec.engine, "value", model_spec.engine)
    engine_type = (engine_value or "cuda-vllm").lower()

    estimate = estimate_vram_requirements(
        model_name_or_params=model_spec.model_path,
        max_model_len=max_len,
        dtype="fp16",
        quantization=quantization,
        engine_type=engine_type,
        tensor_parallel_size=tp_size,
    )

    params_gb = estimate_model_params(model_spec)
    suggestions = list(estimate.suggestions)
    if not suggestions and not estimate.is_safe:
        suggestions.append("降低模型配置或切换到量化/多卡模式。")

    message = "显存充足" if estimate.is_safe else "显存不足，无法加载模型"

    return VRAMDiagnosticReport(
        model_id=model_id,
        model_params_billion=params_gb,
        kv_cache_gb=estimate.kv_cache_gb,
        required_vram_gb=estimate.required_gb,
        available_vram_gb=estimate.available_gb,
        success=estimate.is_safe,
        suggestions=suggestions,
        message=message,
    )


def format_vram_report(report: VRAMDiagnosticReport) -> str:
    """格式化 VRAM 诊断报告为人类可读的字符串。
    
    Args:
        report: VRAM 诊断报告对象
        
    Returns:
        格式化的报告字符串
    """
    # 安全获取值，处理潜在的 None
    model_id = report.model_id or "unknown"
    message = report.message or "未知状态"
    params = report.model_params_billion if report.model_params_billion is not None else 0.0
    required = report.required_vram_gb if report.required_vram_gb is not None else 0.0
    kv_cache = report.kv_cache_gb if report.kv_cache_gb is not None else 0.0
    available = report.available_vram_gb if report.available_vram_gb is not None else 0.0
    
    lines = [
        f"[{model_id}] {message}",
        f"  模型参数量: {params:.1f}B",
        f"  理论显存需求: {required:.1f}GB (含 KV Cache {kv_cache:.2f}GB)",
        f"  当前可用显存: {available:.1f}GB",
    ]
    if not report.success and report.suggestions:
        lines.append("  建议:")
        for suggestion in report.suggestions:
            lines.append(f"    - {suggestion}")
    return "\n".join(lines)


def gather_gpu_summary() -> Dict[str, Optional[str]]:
    """收集 GPU 硬件摘要信息。
    
    Returns:
        包含 GPU 信息的字典，如果 GPU 不可用则返回 available=False
    """
    if torch is None or not torch.cuda.is_available():
        return {"available": False}
    
    try:
        device_idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(device_idx) or "Unknown GPU"
        total = get_total_vram_gb()
        available = get_available_vram_gb()
        cuda_version = f"{torch.version.cuda}" if torch.version.cuda else "unknown"
        return {
            "available": True,
            "name": name,
            "total": f"{total:.1f}GB",
            "free": f"{available:.1f}GB",
            "cuda": cuda_version,
        }
    except Exception as e:
        LOGGER.warning("获取 GPU 信息失败: %s", e)
        return {"available": False, "error": str(e)}


def dependency_matrix() -> Dict[str, str]:
    return {
        "torch": "2.9.0",
        "torchvision": "0.24.0",
        "torchaudio": "2.9.0",
        "vllm": "0.11.2",
        "flashinfer-python": "0.5.2",
        "llguidance": "1.3.0",
        "xgrammar": "0.1.25",
        "transformers": "4.56.0",
    }


def check_dependencies(matrix: Optional[Dict[str, str]] = None) -> List[DependencyStatus]:
    statuses: List[DependencyStatus] = []
    matrix = matrix or dependency_matrix()
    for name, required in matrix.items():
        installed = None
        status = "missing"
        detail = "未安装"
        try:
            installed = importlib_metadata.version(name)
            parsed_installed = parse(installed)
            parsed_required = parse(required)
            if parsed_installed >= parsed_required:
                status = "ok"
                detail = "满足"
            else:
                status = "conflict"
                detail = f"需要 {required}，当前 {installed}"
        except importlib_metadata.PackageNotFoundError:
            detail = "缺失"
        except InvalidVersion:
            detail = f"版本解析失败（{installed or 'unknown'}）"
            status = "conflict"
        statuses.append(
            DependencyStatus(name=name, installed=installed, required=required, status=status, detail=detail)
        )
    return statuses


def format_dependency_summary(statuses: List[DependencyStatus]) -> List[str]:
    lines: List[str] = []
    for status in statuses:
        prefix = "✓" if status.status == "ok" else "✗"
        version = status.installed or "未安装"
        lines.append(f"  {prefix} {status.name} ({version}) - {status.detail}")
    return lines


@dataclass
class CompatibilityWarning:
    """版本兼容性警告"""
    level: str  # "error", "warning", "info"
    component1: str
    version1: str
    component2: str
    version2: Optional[str]
    message: str
    suggestion: str


def check_version_compatibility() -> List[CompatibilityWarning]:
    """检查包之间的版本兼容性

    Returns:
        兼容性警告列表
    """
    warnings: List[CompatibilityWarning] = []

    # 获取已安装的包版本
    try:
        torch_version = importlib_metadata.version("torch") if torch else None
        vllm_version = None
        trt_version = None
        cuda_version = None

        try:
            vllm_version = importlib_metadata.version("vllm")
        except importlib_metadata.PackageNotFoundError:
            pass

        try:
            trt_version = importlib_metadata.version("tensorrt-llm")
        except importlib_metadata.PackageNotFoundError:
            pass

        if torch and torch.version.cuda:
            cuda_version = torch.version.cuda

    except Exception as e:
        LOGGER.warning("获取包版本时出错: %s", e)
        return warnings

    # === 检查 1: PyTorch vs CUDA 兼容性 ===
    if torch_version and cuda_version:
        try:
            torch_parsed = parse(torch_version)

            # PyTorch 2.4+ 需要 CUDA 12.1+
            if torch_parsed >= parse("2.4.0"):
                cuda_major = int(cuda_version.split(".")[0])
                cuda_minor = int(cuda_version.split(".")[1]) if len(cuda_version.split(".")) > 1 else 0

                if cuda_major < 12 or (cuda_major == 12 and cuda_minor < 1):
                    warnings.append(CompatibilityWarning(
                        level="error",
                        component1="torch",
                        version1=torch_version,
                        component2="CUDA",
                        version2=cuda_version,
                        message=f"PyTorch {torch_version} 需要 CUDA 12.1+，当前 CUDA {cuda_version}",
                        suggestion="升级 CUDA 到 12.1+ 或降级 PyTorch 到 2.1-2.3 版本"
                    ))

            # PyTorch 2.1-2.3 兼容 CUDA 11.8 和 12.x
            elif parse("2.1.0") <= torch_parsed < parse("2.4.0"):
                cuda_major = int(cuda_version.split(".")[0])
                if cuda_major not in [11, 12]:
                    warnings.append(CompatibilityWarning(
                        level="warning",
                        component1="torch",
                        version1=torch_version,
                        component2="CUDA",
                        version2=cuda_version,
                        message=f"PyTorch {torch_version} 推荐使用 CUDA 11.8 或 12.x",
                        suggestion="考虑使用推荐的 CUDA 版本以获得最佳性能"
                    ))

        except (InvalidVersion, ValueError, IndexError):
            pass

    # === 检查 2: vLLM vs PyTorch 兼容性 ===
    if vllm_version and torch_version:
        try:
            vllm_parsed = parse(vllm_version)
            torch_parsed = parse(torch_version)

            # vLLM 0.6+ 需要 PyTorch 2.4+
            if vllm_parsed >= parse("0.6.0") and torch_parsed < parse("2.4.0"):
                warnings.append(CompatibilityWarning(
                    level="error",
                    component1="vllm",
                    version1=vllm_version,
                    component2="torch",
                    version2=torch_version,
                    message=f"vLLM {vllm_version} 需要 PyTorch 2.4+，当前 PyTorch {torch_version}",
                    suggestion="升级 PyTorch: pip install 'torch>=2.4.0' --extra-index-url https://download.pytorch.org/whl/cu121"
                ))

            # vLLM 0.5.x 兼容 PyTorch 2.1-2.4
            elif parse("0.5.0") <= vllm_parsed < parse("0.6.0"):
                if not (parse("2.1.0") <= torch_parsed < parse("2.5.0")):
                    warnings.append(CompatibilityWarning(
                        level="warning",
                        component1="vllm",
                        version1=vllm_version,
                        component2="torch",
                        version2=torch_version,
                        message=f"vLLM {vllm_version} 推荐 PyTorch 2.1-2.4",
                        suggestion="考虑使用推荐的 PyTorch 版本"
                    ))

        except (InvalidVersion, ValueError):
            pass

    # === 检查 3: TensorRT-LLM vs PyTorch 兼容性 ===
    if trt_version and torch_version:
        try:
            torch_parsed = parse(torch_version)

            # TensorRT-LLM 通常需要 PyTorch 2.1-2.3（不支持 2.4+）
            if torch_parsed >= parse("2.4.0"):
                warnings.append(CompatibilityWarning(
                    level="error",
                    component1="tensorrt-llm",
                    version1=trt_version,
                    component2="torch",
                    version2=torch_version,
                    message=f"TensorRT-LLM 不支持 PyTorch {torch_version}（需要 2.1-2.3）",
                    suggestion="降级 PyTorch: pip install 'torch>=2.1.0,<2.4.0' --extra-index-url https://download.pytorch.org/whl/cu121"
                ))

            elif torch_parsed < parse("2.1.0"):
                warnings.append(CompatibilityWarning(
                    level="warning",
                    component1="tensorrt-llm",
                    version1=trt_version,
                    component2="torch",
                    version2=torch_version,
                    message=f"TensorRT-LLM 推荐 PyTorch 2.1+，当前 {torch_version}",
                    suggestion="升级 PyTorch 到 2.1-2.3 版本范围"
                ))

        except (InvalidVersion, ValueError):
            pass

    # === 检查 4: vLLM 和 TensorRT-LLM 共存冲突 ===
    if vllm_version and trt_version:
        warnings.append(CompatibilityWarning(
            level="error",
            component1="vllm",
            version1=vllm_version,
            component2="tensorrt-llm",
            version2=trt_version,
            message="vLLM 和 TensorRT-LLM 不应安装在同一环境（PyTorch 版本冲突）",
            suggestion="使用 ./cy-llm setup --engine cuda-vllm 或 --engine cuda-trt 创建独立环境"
        ))

    return warnings


def format_compatibility_warnings(warnings: List[CompatibilityWarning]) -> List[str]:
    """格式化兼容性警告

    Args:
        warnings: 兼容性警告列表

    Returns:
        格式化的警告信息列表
    """
    if not warnings:
        return ["  ✓ 所有依赖版本兼容"]

    lines: List[str] = []

    # 按严重程度分组
    errors = [w for w in warnings if w.level == "error"]
    warns = [w for w in warnings if w.level == "warning"]
    infos = [w for w in warnings if w.level == "info"]

    if errors:
        lines.append("  ❌ 严重兼容性问题:")
        for w in errors:
            lines.append(f"     • {w.message}")
            lines.append(f"       建议: {w.suggestion}")

    if warns:
        lines.append("  ⚠️  兼容性警告:")
        for w in warns:
            lines.append(f"     • {w.message}")
            lines.append(f"       建议: {w.suggestion}")

    if infos:
        lines.append("  ℹ️  信息:")
        for w in infos:
            lines.append(f"     • {w.message}")

    return lines