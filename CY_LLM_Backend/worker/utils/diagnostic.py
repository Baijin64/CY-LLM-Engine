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
    if torch is None or not torch.cuda.is_available():
        return 0.0
    free, _ = torch.cuda.mem_get_info()
    return free / BYTE_PER_GB


def get_total_vram_gb() -> float:
    if torch is None or not torch.cuda.is_available():
        return 0.0
    _, total = torch.cuda.mem_get_info()
    return total / BYTE_PER_GB


def check_vram_for_model(model_id: str, model_spec: ModelSpec) -> VRAMDiagnosticReport:
    params_gb = estimate_model_params(model_spec)
    base_vram = params_gb * 2.0
    max_len = model_spec.max_model_len or 8192
    tp_size = model_spec.tensor_parallel_size or 1
    kv_cache = estimate_kv_cache(max_len, tp_size)
    available = get_available_vram_gb()
    required = base_vram + kv_cache
    success = available >= required
    suggestions: List[str] = []
    if not success:
        if not model_spec.use_4bit:
            suggestions.append("启用 4bit/INT4 量化以显著降低模型体积。")
        suggested_len = max(1024, int(max_len * available / max(required, 1.0)))
        if suggested_len < max_len:
            suggestions.append(f"降低 max_model_len 至 {suggested_len} (当前 {max_len}) 以缩减 KV Cache。")
        current_util = model_spec.gpu_memory_utilization or 0.9
        optimized_util = max(0.5, min(current_util * 0.85, 0.75))
        suggestions.append(f"将 gpu_memory_utilization 调整至 {optimized_util:.2f}，留出足够余量。")
    message = (
        "显存充足" if success else "显存不足，无法加载模型"
    )
    return VRAMDiagnosticReport(
        model_id=model_id,
        model_params_billion=params_gb,
        kv_cache_gb=kv_cache,
        required_vram_gb=required,
        available_vram_gb=available,
        success=success,
        suggestions=suggestions,
        message=message,
    )


def format_vram_report(report: VRAMDiagnosticReport) -> str:
    lines = [
        f"[{report.model_id}] {report.message}",
        f"  模型参数量: {report.model_params_billion:.1f}B",
        f"  理论显存需求: {report.required_vram_gb:.1f}GB (含 KV Cache {report.kv_cache_gb:.2f}GB)",
        f"  当前可用显存: {report.available_vram_gb:.1f}GB",
    ]
    if not report.success and report.suggestions:
        lines.append("  建议:")
        for suggestion in report.suggestions:
            lines.append(f"    - {suggestion}")
    return "\n".join(lines)


def gather_gpu_summary() -> Dict[str, Optional[str]]:
    if torch is None or not torch.cuda.is_available():
        return {"available": False}
    device_idx = torch.cuda.current_device()
    name = torch.cuda.get_device_name(device_idx)
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