"""
engines 模块
[引擎工厂与抽象接口]

本模块提供推理引擎的工厂方法和抽象基类。
具体引擎实现在各自文件中，通过延迟导入按需加载。

使用示例：
    >>> from worker.engines import create_engine, list_available_engines
    >>> 
    >>> # 查看可用引擎
    >>> print(list_available_engines())
    >>> 
    >>> # 创建引擎（延迟导入，只加载需要的依赖）
    >>> engine = create_engine("cuda-vllm")
    >>> engine.load_model("deepseek-ai/deepseek-llm-7b-chat")
    >>> 
    >>> # 使用异步流式引擎（推荐，更低 TTFB）
    >>> async_engine = create_engine("cuda-vllm-async", enable_prefix_caching=True)
    >>> async_engine.load_model("deepseek-ai/deepseek-llm-7b-chat")
    >>> async for token in async_engine.infer_stream("你好"):
    ...     print(token, end="", flush=True)

支持的引擎：
    - cuda-vllm:       NVIDIA GPU + vLLM（同步，兼容）
    - cuda-vllm-async: NVIDIA GPU + vLLM AsyncEngine（异步流式，推荐）
    - cuda-trt:        NVIDIA GPU + TensorRT-LLM（极致性能）
    - ascend-vllm:     华为 NPU + vLLM-Ascend（通用）
    - ascend-mindie:   华为 NPU + MindIE Turbo（极致性能）
    - nvidia:          旧版兼容（transformers）
    - ascend:          旧版兼容（transformers）
"""

# 只导出工厂方法和抽象基类，不导入具体引擎实现
# 这确保了延迟加载：只有在 create_engine() 被调用时才加载对应的引擎模块

from .abstract_engine import BaseEngine
from .engine_factory import (
    create_engine,
    create_engine_from_config,
    detect_default_engine,
    detect_default_engine_type,
    list_available_engines,
    list_engines_by_platform,
    check_engine_available,
    register_engine,
    unregister_engine,
    get_engine_info,
    EngineInfo,
    ENGINE_REGISTRY,
)

__all__ = [
    # 抽象基类
    "BaseEngine",
    
    # 工厂方法
    "create_engine",
    "create_engine_from_config",
    "detect_default_engine",
    "detect_default_engine_type",
    
    # 工具函数
    "list_available_engines",
    "list_engines_by_platform",
    "check_engine_available",
    
    # 插件化 API
    "register_engine",
    "unregister_engine",
    "get_engine_info",
    "EngineInfo",
    
    # 注册表（供高级用户扩展）
    "ENGINE_REGISTRY",
]
