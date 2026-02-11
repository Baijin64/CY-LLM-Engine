"""
abstract_engine.py
推理引擎抽象基类

定义所有推理引擎必须实现的接口，支持：
- 模型加载/卸载
- 同步和异步推理
- 显存监控
- 健康检查
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, Generator, Optional


class BaseEngine(ABC):
    """
    推理引擎抽象基类。
    
    所有具体引擎（vLLM、TensorRT、MindIE 等）必须实现此接口。
    
    生命周期：
        1. 创建引擎实例
        2. load_model() 加载模型
        3. infer() 执行推理（可多次）
        4. unload_model() 卸载释放资源
    
    示例：
        >>> engine = VllmCudaEngine()
        >>> engine.load_model("deepseek-ai/deepseek-llm-7b")
        >>> for token in engine.infer("你好"):
        ...     print(token, end="")
        >>> engine.unload_model()
    """

    @abstractmethod
    def load_model(
        self,
        model_path: str,
        adapter_path: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        加载模型到显存。
        
        Args:
            model_path: 模型路径（HuggingFace ID 或本地路径）
            adapter_path: LoRA 适配器路径（可选）
            **kwargs: 引擎特定参数
        
        Raises:
            RuntimeError: 模型加载失败
            FileNotFoundError: 模型路径不存在
        """

    @abstractmethod
    def infer(
        self,
        prompt: str,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        同步流式推理。
        
        Args:
            prompt: 输入提示文本
            **kwargs: 生成参数（temperature, top_p, max_tokens 等）
        
        Yields:
            生成的文本 token
        
        Raises:
            RuntimeError: 模型未加载或推理失败
        """

    @abstractmethod
    def unload_model(self) -> None:
        """
        卸载模型并释放显存。
        
        应确保：
        - 删除所有模型引用
        - 调用 gc.collect() 和 torch.cuda.empty_cache()
        - 可安全多次调用
        """

    @abstractmethod
    def get_memory_usage(self) -> Dict[str, float]:
        """
        获取当前显存使用情况。
        
        Returns:
            包含以下键的字典：
            - allocated_gb: 已分配显存 (GB)
            - reserved_gb: 已预留显存 (GB)
            - total_gb: 总显存 (GB)
            - utilization: 利用率 (0-1)
        """

    # ==================== 可选方法（有默认实现） ====================

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取当前加载的模型信息。
        
        Returns:
            模型元信息字典
        """
        return {
            "is_loaded": False,
            "engine": self.__class__.__name__,
        }

    def health_check(self) -> bool:
        """
        健康检查。
        
        Returns:
            引擎是否健康可用
        """
        try:
            mem = self.get_memory_usage()
            return mem.get("total_gb", 0) > 0
        except Exception:
            return False

    @property
    def is_async(self) -> bool:
        """是否为异步引擎"""
        return False

    # ==================== 异步接口（可选实现） ====================

    async def async_infer(
        self,
        prompt: str,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        异步流式推理。
        
        默认实现将同步 infer 包装为异步，子类可覆盖以提供真正的异步实现。
        
        Args:
            prompt: 输入提示文本
            **kwargs: 生成参数
        
        Yields:
            生成的文本 token
        """
        import asyncio
        loop = asyncio.get_running_loop()
        
        # 在线程池中运行同步生成
        for token in self.infer(prompt, **kwargs):
            yield token
            await asyncio.sleep(0)  # 让出控制权
