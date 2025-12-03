"""
engine_factory.py
[引擎工厂] 延迟导入 + 策略模式 + 插件化注册，支持多种推理引擎按需加载
说明：只有在实际创建引擎时才导入对应模块，避免无用依赖加载。

支持的引擎：
  - cuda-vllm:      NVIDIA GPU + vLLM（通用，推荐）
  - cuda-trt:       NVIDIA GPU + TensorRT-LLM（极致性能）
  - ascend-vllm:    华为 NPU + vLLM-Ascend（通用）
  - ascend-mindie:  华为 NPU + MindIE Turbo（极致性能）
  - nvidia:         旧版兼容（映射到原有 transformers 实现）
  - ascend:         旧版兼容（映射到原有 transformers 实现）

插件扩展：
  >>> from worker.engines import register_engine
  >>> register_engine("my-engine", "my_package.my_module:MyEngine", description="自定义引擎")
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Type, Union

from ..config.config_loader import HardwareProfile, WorkerConfig, detect_hardware, load_worker_config
from .abstract_engine import BaseEngine

LOGGER = logging.getLogger("cy_llm.worker.engines.factory")


# ============================================================================
# 引擎注册信息
# ============================================================================

@dataclass
class EngineInfo:
    """引擎注册信息"""
    engine_type: str
    module_path: str  # 格式: "module.path:ClassName" 或直接是类
    description: str = ""
    platform: str = "any"  # cuda, ascend, cpu, any
    priority: int = 0  # 同平台下的优先级（越高越优先）
    is_builtin: bool = True  # 是否为内置引擎
    
    @property
    def class_name(self) -> str:
        if isinstance(self.module_path, str) and ":" in self.module_path:
            return self.module_path.rsplit(":", 1)[1]
        return str(self.module_path)


# ============================================================================
# 引擎注册表（支持动态注册）
# ============================================================================

class EngineRegistry:
    """
    引擎注册表（单例）
    
    支持：
      - 内置引擎注册
      - 第三方引擎动态注册
      - 按平台筛选
      - 延迟导入
    """
    _instance: Optional['EngineRegistry'] = None
    
    def __new__(cls) -> 'EngineRegistry':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._engines: Dict[str, EngineInfo] = {}
        self._loaded_classes: Dict[str, Type[BaseEngine]] = {}
        self._hooks: List[Callable[[str, EngineInfo], None]] = []
        
        # 注册内置引擎
        self._register_builtin_engines()
        self._initialized = True
    
    def _register_builtin_engines(self):
        """注册内置引擎"""
        builtin = [
            # NVIDIA CUDA 平台
            EngineInfo(
                engine_type="cuda-vllm",
                module_path="worker.engines.vllm_cuda_engine:VllmCudaEngine",
                description="NVIDIA GPU + vLLM（通用，高性能，推荐）",
                platform="cuda",
                priority=100,
            ),
            EngineInfo(
                engine_type="cuda-vllm-async",
                module_path="worker.engines.vllm_async_engine:VllmAsyncEngine",
                description="NVIDIA GPU + vLLM 异步（真正流式推理）",
                platform="cuda",
                priority=90,
            ),
            EngineInfo(
                engine_type="cuda-trt",
                module_path="worker.engines.trt_engine:TensorRTEngine",
                description="NVIDIA GPU + TensorRT-LLM（极致性能，需预编译）",
                platform="cuda",
                priority=80,
            ),
            
            # 华为 Ascend 平台
            EngineInfo(
                engine_type="ascend-vllm",
                module_path="worker.engines.vllm_ascend_engine:VllmAscendEngine",
                description="华为 NPU + vLLM-Ascend（通用，CANN 环境）",
                platform="ascend",
                priority=100,
            ),
            EngineInfo(
                engine_type="ascend-mindie",
                module_path="worker.engines.mindie_engine:MindIEEngine",
                description="华为 NPU + MindIE Turbo（极致性能，华为原生）",
                platform="ascend",
                priority=80,
            ),
            
            # 旧版兼容别名
            EngineInfo(
                engine_type="nvidia",
                module_path="worker.engines.nvidia_engine:NvidiaEngine",
                description="NVIDIA GPU + Transformers（旧版兼容）",
                platform="cuda",
                priority=50,
            ),
            EngineInfo(
                engine_type="cuda",
                module_path="worker.engines.nvidia_engine:NvidiaEngine",
                description="NVIDIA GPU + Transformers（别名）",
                platform="cuda",
                priority=50,
            ),
            EngineInfo(
                engine_type="ascend",
                module_path="worker.engines.ascend_engine:AscendEngine",
                description="华为 NPU + Transformers（旧版兼容）",
                platform="ascend",
                priority=50,
            ),
            EngineInfo(
                engine_type="npu",
                module_path="worker.engines.ascend_engine:AscendEngine",
                description="华为 NPU + Transformers（别名）",
                platform="ascend",
                priority=50,
            ),
            EngineInfo(
                engine_type="hybrid",
                module_path="worker.engines.hybrid_engine:HybridEngine",
                description="混合引擎（实验性）",
                platform="any",
                priority=10,
            ),
        ]
        
        for info in builtin:
            self._engines[info.engine_type] = info
    
    def register(
        self,
        engine_type: str,
        module_path: Union[str, Type[BaseEngine]],
        description: str = "",
        platform: str = "any",
        priority: int = 0,
        overwrite: bool = False,
    ) -> None:
        """
        注册引擎。
        
        Args:
            engine_type: 引擎类型标识符，如 "my-engine"
            module_path: 模块路径 "package.module:ClassName" 或引擎类
            description: 引擎描述
            platform: 目标平台 (cuda/ascend/cpu/any)
            priority: 优先级（同平台下越高越优先）
            overwrite: 是否覆盖已有注册
            
        Raises:
            ValueError: 引擎已存在且未允许覆盖
        """
        engine_type = engine_type.lower().strip()
        
        if engine_type in self._engines and not overwrite:
            raise ValueError(
                f"引擎 '{engine_type}' 已注册。"
                f"使用 overwrite=True 覆盖，或使用 unregister() 先移除。"
            )
        
        # 如果传入的是类，直接缓存
        if isinstance(module_path, type):
            if not issubclass(module_path, BaseEngine):
                raise TypeError(f"引擎类必须继承 BaseEngine: {module_path}")
            self._loaded_classes[engine_type] = module_path
            module_path_str = f"{module_path.__module__}:{module_path.__name__}"
        else:
            module_path_str = module_path
        
        info = EngineInfo(
            engine_type=engine_type,
            module_path=module_path_str,
            description=description,
            platform=platform,
            priority=priority,
            is_builtin=False,
        )
        
        self._engines[engine_type] = info
        LOGGER.info("注册引擎: %s (%s) [%s]", engine_type, description, platform)
        
        # 触发钩子
        for hook in self._hooks:
            try:
                hook(engine_type, info)
            except Exception as e:
                LOGGER.warning("引擎注册钩子执行失败: %s", e)
    
    def unregister(self, engine_type: str) -> bool:
        """
        移除引擎注册。
        
        Args:
            engine_type: 引擎类型
            
        Returns:
            是否成功移除
        """
        engine_type = engine_type.lower().strip()
        if engine_type in self._engines:
            info = self._engines.pop(engine_type)
            self._loaded_classes.pop(engine_type, None)
            LOGGER.info("移除引擎注册: %s", engine_type)
            return True
        return False
    
    def get(self, engine_type: str) -> Optional[EngineInfo]:
        """获取引擎信息"""
        return self._engines.get(engine_type.lower().strip())
    
    def list_all(self) -> Dict[str, EngineInfo]:
        """列出所有注册的引擎"""
        return dict(self._engines)
    
    def list_by_platform(self, platform: str) -> List[EngineInfo]:
        """按平台筛选引擎，按优先级排序"""
        engines = [
            info for info in self._engines.values()
            if info.platform in (platform, "any")
        ]
        return sorted(engines, key=lambda x: -x.priority)
    
    def get_class(self, engine_type: str) -> Type[BaseEngine]:
        """
        获取引擎类（延迟导入）。
        
        Args:
            engine_type: 引擎类型
            
        Returns:
            引擎类
            
        Raises:
            ValueError: 未知的引擎类型
            ImportError: 导入失败
        """
        engine_type = engine_type.lower().strip()
        
        # 检查缓存
        if engine_type in self._loaded_classes:
            return self._loaded_classes[engine_type]
        
        # 获取注册信息
        info = self._engines.get(engine_type)
        if info is None:
            available = ", ".join(sorted(self._engines.keys()))
            raise ValueError(
                f"未知的引擎类型: '{engine_type}'。\n"
                f"可用选项: {available}"
            )
        
        # 延迟导入
        engine_cls = _lazy_import_engine(info.module_path)
        self._loaded_classes[engine_type] = engine_cls
        return engine_cls
    
    def add_hook(self, hook: Callable[[str, EngineInfo], None]) -> None:
        """添加引擎注册钩子（在新引擎注册时调用）"""
        self._hooks.append(hook)


# 全局注册表实例
_registry = EngineRegistry()


# ============================================================================
# 公开 API
# ============================================================================

# 保持向后兼容
ENGINE_REGISTRY: Dict[str, str] = {
    info.engine_type: info.module_path
    for info in _registry.list_all().values()
    if isinstance(info.module_path, str)
}

# 默认引擎优先级（按硬件自动选择时使用）
DEFAULT_ENGINE_PRIORITY = {
    "cuda": "cuda-vllm",
    "ascend": "ascend-vllm",
}


def register_engine(
    engine_type: str,
    module_path: Union[str, Type[BaseEngine]],
    description: str = "",
    platform: str = "any",
    priority: int = 0,
    overwrite: bool = False,
) -> None:
    """
    注册第三方引擎（公开 API）。
    
    Args:
        engine_type: 引擎类型标识符
        module_path: 模块路径 "package.module:ClassName" 或引擎类
        description: 引擎描述
        platform: 目标平台
        priority: 优先级
        overwrite: 是否覆盖已有注册
        
    示例:
        >>> from worker.engines import register_engine
        >>> register_engine(
        ...     "my-custom-engine",
        ...     "my_package.engines:MyCustomEngine",
        ...     description="我的自定义推理引擎",
        ...     platform="cuda",
        ... )
    """
    _registry.register(
        engine_type=engine_type,
        module_path=module_path,
        description=description,
        platform=platform,
        priority=priority,
        overwrite=overwrite,
    )
    # 更新兼容字典
    ENGINE_REGISTRY[engine_type] = (
        module_path if isinstance(module_path, str) 
        else f"{module_path.__module__}:{module_path.__name__}"
    )


def unregister_engine(engine_type: str) -> bool:
    """移除引擎注册"""
    result = _registry.unregister(engine_type)
    ENGINE_REGISTRY.pop(engine_type, None)
    return result


def get_engine_info(engine_type: str) -> Optional[EngineInfo]:
    """获取引擎信息"""
    return _registry.get(engine_type)


class EngineFactory:
    """兼容旧 API: 工厂类，创建引擎实例"""

    @staticmethod
    def create(engine_type: str, *args, **kwargs) -> BaseEngine:
        engine_cls = _registry.get_class(engine_type)
        return engine_cls(*args, **kwargs)


def get_engine(engine_type: str, *args, **kwargs) -> BaseEngine:
    """兼容旧 API: 直接创建引擎实例"""
    return EngineFactory.create(engine_type, *args, **kwargs)


def _lazy_import_engine(engine_path: str) -> type:
    """
    延迟导入引擎类。
    
    Args:
        engine_path: 格式为 "module.path:ClassName"
        
    Returns:
        引擎类（未实例化）
        
    Raises:
        ImportError: 模块导入失败
        AttributeError: 类不存在
    """
    module_path, class_name = engine_path.rsplit(":", 1)
    
    try:
        module = importlib.import_module(module_path)
        engine_cls = getattr(module, class_name)
        LOGGER.debug("成功加载引擎: %s from %s", class_name, module_path)
        return engine_cls
    except ImportError as e:
        LOGGER.error("无法导入引擎模块 %s: %s", module_path, e)
        raise ImportError(
            f"引擎 '{class_name}' 的依赖未安装。"
            f"请检查是否安装了对应的推理框架。\n"
            f"原始错误: {e}"
        ) from e
    except AttributeError as e:
        LOGGER.error("模块 %s 中找不到类 %s", module_path, class_name)
        raise


def create_engine(engine_type: str, **kwargs) -> BaseEngine:
    """
    按类型创建推理引擎实例（延迟导入）。
    
    Args:
        engine_type: 引擎类型标识符，如 "cuda-vllm", "ascend-mindie" 等
        **kwargs: 传递给引擎构造函数的参数
        
    Returns:
        BaseEngine 实例
        
    Raises:
        ValueError: 未知的引擎类型
        ImportError: 引擎依赖未安装
        
    示例:
        >>> engine = create_engine("cuda-vllm")
        >>> engine = create_engine("ascend-mindie", device_id=0)
    """
    engine_type_normalized = engine_type.lower().strip()
    
    LOGGER.info("正在加载引擎: %s", engine_type_normalized)
    
    engine_cls = _registry.get_class(engine_type_normalized)
    return engine_cls(**kwargs)


def detect_default_engine_type(hardware: Optional[HardwareProfile] = None) -> str:
    """
    根据硬件自动检测推荐的引擎类型。
    
    Args:
        hardware: 硬件配置，如不提供则自动检测
        
    Returns:
        推荐的引擎类型字符串
    """
    hardware = hardware or detect_hardware()
    
    if hardware.has_ascend:
        return DEFAULT_ENGINE_PRIORITY["ascend"]
    elif hardware.has_cuda:
        return DEFAULT_ENGINE_PRIORITY["cuda"]
    else:
        # 无 GPU 时回退到 CPU 模式的 nvidia 引擎（transformers 支持 CPU）
        LOGGER.warning("未检测到 GPU，将使用 CPU 模式")
        return "nvidia"


def detect_default_engine(hardware: Optional[HardwareProfile] = None) -> BaseEngine:
    """
    创建一个与当前硬件最匹配的引擎实例。
    
    Args:
        hardware: 硬件配置，如不提供则自动检测
        
    Returns:
        BaseEngine 实例
    """
    engine_type = detect_default_engine_type(hardware)
    return create_engine(engine_type)


def create_engine_from_config(config: Optional[WorkerConfig] = None) -> BaseEngine:
    """
    利用 WorkerConfig（或自动加载）生成对应的引擎。
    
    Args:
        config: Worker 配置，如不提供则自动加载
        
    Returns:
        BaseEngine 实例
    """
    config = config or load_worker_config()
    return create_engine(config.preferred_backend)


def list_available_engines() -> Dict[str, str]:
    """
    列出所有可用的引擎类型及其描述。
    
    Returns:
        引擎类型到描述的映射
    """
    return {
        info.engine_type: info.description
        for info in _registry.list_all().values()
    }


def list_engines_by_platform(platform: str) -> List[EngineInfo]:
    """
    按平台列出引擎（按优先级排序）。
    
    Args:
        platform: cuda, ascend, cpu, any
        
    Returns:
        引擎信息列表
    """
    return _registry.list_by_platform(platform)


def check_engine_available(engine_type: str) -> bool:
    """
    检查指定引擎是否可用（依赖是否已安装）。
    
    Args:
        engine_type: 引擎类型
        
    Returns:
        True 如果引擎可用，否则 False
    """
    engine_type_normalized = engine_type.lower().strip()
    info = _registry.get(engine_type_normalized)
    if info is None:
        return False
    
    try:
        _registry.get_class(engine_type_normalized)
        return True
    except (ImportError, AttributeError):
        return False


# 添加导入到 List
from typing import List

