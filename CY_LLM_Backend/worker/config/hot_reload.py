"""
config/hot_reload.py
[配置热重载] 监听配置文件变化并自动重载

功能：
  - 监听配置文件变化（基于文件修改时间或 inotify）
  - 自动重载配置
  - 通知订阅者配置变更
  - 支持回调钩子

使用示例：
    >>> watcher = ConfigWatcher(config_path="/etc/worker/config.json")
    >>> watcher.on_reload(lambda config: print(f"Config reloaded: {config}"))
    >>> watcher.start()
    >>> # ... 修改配置文件 ...
    >>> watcher.stop()
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any

LOGGER = logging.getLogger("cy_llm.worker.config.hot_reload")

# 尝试导入 watchdog（可选依赖）
_HAS_WATCHDOG = False
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent
    _HAS_WATCHDOG = True
except ImportError:
    LOGGER.info("watchdog 未安装，将使用轮询模式监听配置变化")


@dataclass
class ReloadEvent:
    """重载事件"""
    path: str
    old_hash: str
    new_hash: str
    timestamp: float = field(default_factory=time.time)
    success: bool = True
    error: Optional[str] = None


class ConfigWatcher:
    """
    配置文件监听器。
    
    支持两种模式：
      1. watchdog 模式（推荐）：使用 inotify 实时监听
      2. 轮询模式：定期检查文件修改时间
    """
    
    def __init__(
        self,
        config_path: str,
        check_interval: float = 5.0,
        use_watchdog: bool = True,
        debounce_seconds: float = 1.0,
    ):
        """
        初始化监听器。
        
        Args:
            config_path: 配置文件路径
            check_interval: 轮询检查间隔（秒）
            use_watchdog: 是否使用 watchdog（如果可用）
            debounce_seconds: 防抖时间（秒），避免频繁重载
        """
        self._config_path = Path(config_path).resolve()
        self._check_interval = check_interval
        self._debounce_seconds = debounce_seconds
        self._use_watchdog = use_watchdog and _HAS_WATCHDOG
        
        self._callbacks: List[Callable[[Any], None]] = []
        self._error_callbacks: List[Callable[[Exception], None]] = []
        self._reload_callbacks: List[Callable[[ReloadEvent], None]] = []
        
        self._current_hash: Optional[str] = None
        self._last_reload_time: float = 0
        self._running = False
        self._lock = threading.Lock()
        
        self._thread: Optional[threading.Thread] = None
        self._observer: Optional[Any] = None  # watchdog Observer
        
        # 初始化 hash
        self._update_hash()
        
        LOGGER.info(
            "ConfigWatcher 初始化: path=%s, mode=%s, interval=%.1fs",
            self._config_path,
            "watchdog" if self._use_watchdog else "polling",
            self._check_interval,
        )
    
    def _compute_file_hash(self) -> str:
        """计算文件内容哈希"""
        try:
            content = self._config_path.read_bytes()
            return hashlib.md5(content).hexdigest()
        except Exception as e:
            LOGGER.warning("计算文件哈希失败: %s", e)
            return ""
    
    def _update_hash(self) -> bool:
        """更新当前哈希，返回是否变化"""
        new_hash = self._compute_file_hash()
        if new_hash and new_hash != self._current_hash:
            self._current_hash = new_hash
            return True
        return False
    
    def on_config(self, callback: Callable[[Any], None]) -> 'ConfigWatcher':
        """
        注册配置变更回调（传入新配置对象）。
        
        Args:
            callback: 回调函数，接收新配置
            
        Returns:
            self（支持链式调用）
        """
        self._callbacks.append(callback)
        return self
    
    def on_error(self, callback: Callable[[Exception], None]) -> 'ConfigWatcher':
        """
        注册错误回调。
        
        Args:
            callback: 回调函数，接收异常
            
        Returns:
            self（支持链式调用）
        """
        self._error_callbacks.append(callback)
        return self
    
    def on_reload(self, callback: Callable[[ReloadEvent], None]) -> 'ConfigWatcher':
        """
        注册重载事件回调。
        
        Args:
            callback: 回调函数，接收 ReloadEvent
            
        Returns:
            self（支持链式调用）
        """
        self._reload_callbacks.append(callback)
        return self
    
    def _should_reload(self) -> bool:
        """检查是否应该重载（防抖）"""
        now = time.time()
        if now - self._last_reload_time < self._debounce_seconds:
            return False
        return True
    
    def _do_reload(self) -> None:
        """执行重载"""
        with self._lock:
            if not self._should_reload():
                return
            
            old_hash = self._current_hash
            new_hash = self._compute_file_hash()
            
            if not new_hash or new_hash == old_hash:
                return
            
            self._last_reload_time = time.time()
            self._current_hash = new_hash
            
            event = ReloadEvent(
                path=str(self._config_path),
                old_hash=old_hash or "",
                new_hash=new_hash,
            )
            
            try:
                # 加载新配置
                from .config_loader import load_worker_config
                new_config = load_worker_config(str(self._config_path))
                
                LOGGER.info("配置已重载: %s", self._config_path)
                
                # 触发配置回调
                for callback in self._callbacks:
                    try:
                        callback(new_config)
                    except Exception as e:
                        LOGGER.error("配置回调执行失败: %s", e)
                
                # 触发重载事件回调
                for callback in self._reload_callbacks:
                    try:
                        callback(event)
                    except Exception as e:
                        LOGGER.error("重载事件回调执行失败: %s", e)
                        
            except Exception as e:
                event.success = False
                event.error = str(e)
                LOGGER.error("配置重载失败: %s", e)
                
                # 触发错误回调
                for callback in self._error_callbacks:
                    try:
                        callback(e)
                    except Exception as e2:
                        LOGGER.error("错误回调执行失败: %s", e2)
    
    def _polling_loop(self) -> None:
        """轮询检查循环"""
        while self._running:
            try:
                if self._update_hash():
                    self._do_reload()
            except Exception as e:
                LOGGER.error("轮询检查失败: %s", e)
            
            time.sleep(self._check_interval)
    
    def start(self) -> None:
        """启动监听"""
        if self._running:
            return
        
        self._running = True
        
        if self._use_watchdog:
            self._start_watchdog()
        else:
            self._start_polling()
        
        LOGGER.info("ConfigWatcher 已启动")
    
    def _start_watchdog(self) -> None:
        """启动 watchdog 模式"""
        class ConfigFileHandler(FileSystemEventHandler):
            def __init__(handler_self, watcher: 'ConfigWatcher'):
                handler_self.watcher = watcher
            
            def on_modified(handler_self, event: FileModifiedEvent):
                if not event.is_directory:
                    event_path = Path(event.src_path).resolve()
                    if event_path == handler_self.watcher._config_path:
                        handler_self.watcher._do_reload()
        
        self._observer = Observer()
        handler = ConfigFileHandler(self)
        self._observer.schedule(
            handler,
            str(self._config_path.parent),
            recursive=False,
        )
        self._observer.start()
    
    def _start_polling(self) -> None:
        """启动轮询模式"""
        self._thread = threading.Thread(
            target=self._polling_loop,
            daemon=True,
            name="ConfigWatcher-Polling",
        )
        self._thread.start()
    
    def stop(self) -> None:
        """停止监听"""
        self._running = False
        
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
        
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        
        LOGGER.info("ConfigWatcher 已停止")
    
    def reload_now(self) -> bool:
        """
        立即重载配置（忽略防抖）。
        
        Returns:
            是否成功重载
        """
        self._last_reload_time = 0  # 重置防抖
        try:
            self._do_reload()
            return True
        except Exception as e:
            LOGGER.error("手动重载失败: %s", e)
            return False
    
    @property
    def config_path(self) -> str:
        """当前监听的配置路径"""
        return str(self._config_path)
    
    @property
    def is_running(self) -> bool:
        """是否正在运行"""
        return self._running


class ModelRegistryWatcher:
    """
    模型注册表监听器。
    
    专门用于监听 models.json 变化，并通知相关组件更新。
    """
    
    _instance: Optional['ModelRegistryWatcher'] = None
    
    def __new__(cls) -> 'ModelRegistryWatcher':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._watcher: Optional[ConfigWatcher] = None
        self._model_change_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self._initialized = True
    
    def setup(
        self,
        registry_path: str,
        check_interval: float = 5.0,
    ) -> 'ModelRegistryWatcher':
        """
        设置监听器。
        
        Args:
            registry_path: models.json 路径
            check_interval: 检查间隔
            
        Returns:
            self
        """
        if self._watcher:
            self._watcher.stop()
        
        self._watcher = ConfigWatcher(
            config_path=registry_path,
            check_interval=check_interval,
        )
        
        # 注册内部回调
        self._watcher.on_config(self._on_registry_change)
        
        return self
    
    def _on_registry_change(self, config: Any) -> None:
        """内部回调：处理注册表变更"""
        # 提取模型注册表
        if hasattr(config, 'model_registry'):
            registry = config.model_registry
        else:
            registry = {}
        
        LOGGER.info("模型注册表已更新: %d 个模型", len(registry))
        
        # 触发外部回调
        for callback in self._model_change_callbacks:
            try:
                callback(registry)
            except Exception as e:
                LOGGER.error("模型变更回调失败: %s", e)
    
    def on_model_change(
        self,
        callback: Callable[[Dict[str, Any]], None],
    ) -> 'ModelRegistryWatcher':
        """
        注册模型变更回调。
        
        Args:
            callback: 回调函数，接收新的模型注册表
            
        Returns:
            self
        """
        self._model_change_callbacks.append(callback)
        return self
    
    def start(self) -> None:
        """启动监听"""
        if self._watcher:
            self._watcher.start()
    
    def stop(self) -> None:
        """停止监听"""
        if self._watcher:
            self._watcher.stop()
    
    def reload_now(self) -> bool:
        """立即重载"""
        if self._watcher:
            return self._watcher.reload_now()
        return False


# 全局实例
_model_registry_watcher = ModelRegistryWatcher()


def get_model_registry_watcher() -> ModelRegistryWatcher:
    """获取全局模型注册表监听器"""
    return _model_registry_watcher


__all__ = [
    "ConfigWatcher",
    "ModelRegistryWatcher",
    "ReloadEvent",
    "get_model_registry_watcher",
]
