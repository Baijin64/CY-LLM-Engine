"""
config.validator - 配置验证器
提供配置验证和规范化功能
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    EngineType,
    HardwareProfile,
    ModelSpec,
    WorkerConfig,
    get_supported_engines,
    is_valid_engine,
)

LOGGER = logging.getLogger("ew.worker.config.validator")


class ConfigValidationError(Exception):
    """配置验证错误"""
    
    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__(f"配置验证失败: {'; '.join(errors)}")


class ConfigValidator:
    """
    配置验证器
    
    职责:
    1. 验证配置完整性
    2. 检查路径存在性
    3. 验证引擎兼容性
    4. 提供修复建议
    """

    def __init__(self, strict: bool = False):
        """
        Args:
            strict: 严格模式，任何警告都会变成错误
        """
        self.strict = strict
        self._errors: List[str] = []
        self._warnings: List[str] = []

    def validate(self, config: WorkerConfig) -> Tuple[bool, List[str], List[str]]:
        """
        验证配置
        
        Returns:
            (is_valid, errors, warnings) 元组
        """
        self._errors.clear()
        self._warnings.clear()
        
        self._validate_engine(config)
        self._validate_hardware_compatibility(config)
        self._validate_model_registry(config)
        
        if self.strict:
            self._errors.extend(self._warnings)
            self._warnings.clear()
        
        is_valid = len(self._errors) == 0
        return is_valid, self._errors.copy(), self._warnings.copy()

    def _validate_engine(self, config: WorkerConfig) -> None:
        """验证引擎配置"""
        engine_type = config.get_engine_type()
        
        if not is_valid_engine(engine_type):
            self._errors.append(
                f"不支持的引擎类型: '{engine_type}'。"
                f"支持的类型: {', '.join(get_supported_engines())}"
            )

    def _validate_hardware_compatibility(self, config: WorkerConfig) -> None:
        """验证硬件兼容性"""
        engine = config.get_engine_type()
        hw = config.hardware
        
        # CUDA 引擎需要 CUDA 硬件
        if engine.startswith("cuda") and not hw.has_cuda:
            self._warnings.append(
                f"引擎 '{engine}' 需要 NVIDIA GPU，但未检测到 CUDA 设备。"
                "训练/推理可能会失败或回退到 CPU。"
            )
        
        # Ascend 引擎需要 Ascend 硬件
        if engine.startswith("ascend") and not hw.has_ascend:
            self._errors.append(
                f"引擎 '{engine}' 需要华为 Ascend NPU，但未检测到 NPU 设备。"
            )

    def _validate_model_registry(self, config: WorkerConfig) -> None:
        """验证模型注册表"""
        if not config.model_registry:
            self._warnings.append(
                "模型注册表为空。请通过 CY_LLM_MODEL_REGISTRY 或配置文件添加模型。"
            )
            return
        
        for name, spec in config.model_registry.items():
            self._validate_model_spec(name, spec)

    def _validate_model_spec(self, name: str, spec: ModelSpec) -> None:
        """验证单个模型规格"""
        # 检查模型路径
        if not spec.model_path:
            self._errors.append(f"模型 '{name}' 缺少 model_path")
            return
        
        # 本地路径检查
        if spec.model_path.startswith("/") or spec.model_path.startswith("./"):
            if not Path(spec.model_path).exists():
                self._warnings.append(
                    f"模型 '{name}' 的路径不存在: {spec.model_path}"
                )
        
        # 适配器路径检查
        if spec.adapter_path:
            if spec.adapter_path.startswith("/") or spec.adapter_path.startswith("./"):
                if not Path(spec.adapter_path).exists():
                    self._warnings.append(
                        f"模型 '{name}' 的适配器路径不存在: {spec.adapter_path}"
                    )
        
        # GPU 显存利用率范围
        if spec.gpu_memory_utilization is not None:
            if not 0.0 <= spec.gpu_memory_utilization <= 1.0:
                self._errors.append(
                    f"模型 '{name}' 的 gpu_memory_utilization 必须在 0.0-1.0 之间"
                )

    def validate_and_raise(self, config: WorkerConfig) -> None:
        """验证配置，失败时抛出异常"""
        is_valid, errors, warnings = self.validate(config)
        
        for warning in warnings:
            LOGGER.warning(warning)
        
        if not is_valid:
            raise ConfigValidationError(errors)


def validate_config(config: WorkerConfig, strict: bool = False) -> bool:
    """
    验证配置的便捷函数
    
    Args:
        config: 要验证的配置
        strict: 是否严格模式
        
    Returns:
        是否有效
    """
    validator = ConfigValidator(strict=strict)
    is_valid, errors, warnings = validator.validate(config)
    
    for error in errors:
        LOGGER.error("配置错误: %s", error)
    for warning in warnings:
        LOGGER.warning("配置警告: %s", warning)
    
    return is_valid


def suggest_engine(hardware: HardwareProfile) -> EngineType:
    """
    根据硬件建议最佳引擎
    
    Args:
        hardware: 硬件配置
        
    Returns:
        推荐的引擎类型
    """
    if hardware.has_cuda:
        return EngineType.CUDA_VLLM
    elif hardware.has_ascend:
        return EngineType.ASCEND_VLLM
    else:
        return EngineType.NVIDIA  # CPU 回退
