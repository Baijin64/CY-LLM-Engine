"""Abstract base class for inference engines."""

from abc import ABC, abstractmethod
from typing import Dict, Generator, Optional


class BaseEngine(ABC):
    """定义所有推理引擎共用的接口。"""

    @abstractmethod
    def load_model(self, model_path: str, adapter_path: Optional[str] = None, **kwargs) -> None:
        """加载模型到显存。"""

    @abstractmethod
    def infer(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """按流式方式生成回复。"""

    @abstractmethod
    def unload_model(self) -> None:
        """卸载模型并释放显存。"""

    @abstractmethod
    def get_memory_usage(self) -> Dict[str, float]:
        """返回当前显存使用情况。"""
