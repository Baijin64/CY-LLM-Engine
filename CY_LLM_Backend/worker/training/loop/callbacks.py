"""
training.loop.callbacks - 训练回调
负责收集训练进度、保存检查点等
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

LOGGER = logging.getLogger("ew.worker.training.loop.callbacks")

# 延迟导入
_transformers_imported = False
_TrainerCallback = None
_TrainerState = None
_TrainerControl = None
_torch = None


def _ensure_imports():
    """确保依赖已导入"""
    global _transformers_imported, _TrainerCallback, _TrainerState, _TrainerControl, _torch
    if not _transformers_imported:
        try:
            import torch
            from transformers import TrainerCallback, TrainerState, TrainerControl
            _torch = torch
            _TrainerCallback = TrainerCallback
            _TrainerState = TrainerState
            _TrainerControl = TrainerControl
            _transformers_imported = True
        except ImportError as e:
            raise ImportError(f"transformers 未安装: {e}") from e


class JobStatus(Enum):
    """任务状态"""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingProgress:
    """训练进度"""
    job_id: str
    status: JobStatus
    
    # 进度
    current_epoch: int = 0
    total_epochs: int = 0
    current_step: int = 0
    total_steps: int = 0
    
    # 指标
    loss: float = 0.0
    learning_rate: float = 0.0
    grad_norm: float = 0.0
    
    # 资源使用
    gpu_memory_used: float = 0.0  # GB
    gpu_utilization: float = 0.0  # %
    
    # 时间
    elapsed_seconds: float = 0.0
    eta_seconds: float = 0.0
    
    # 消息
    message: str = ""
    error: str = ""
    
    # 时间戳
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def progress_percent(self) -> float:
        """计算进度百分比"""
        if self.total_steps > 0:
            return (self.current_step / self.total_steps) * 100
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "progress_percent": self.progress_percent,
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "grad_norm": self.grad_norm,
            "gpu_memory_used": self.gpu_memory_used,
            "gpu_utilization": self.gpu_utilization,
            "elapsed_seconds": self.elapsed_seconds,
            "eta_seconds": self.eta_seconds,
            "message": self.message,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
        }


class ProgressCallback:
    """
    进度回调
    
    收集训练过程中的进度信息并推送到队列
    
    继承自 transformers.TrainerCallback
    """

    def __init__(
        self,
        job_id: str,
        progress_queue: List[TrainingProgress],
        total_epochs: int = 1,
    ):
        _ensure_imports()
        self.job_id = job_id
        self.progress_queue = progress_queue
        self.total_epochs = total_epochs
        self.start_time = time.time()
        self._last_log_time = 0.0
        self._min_log_interval = 0.5  # 最小日志间隔（秒）

    def _get_gpu_stats(self) -> tuple:
        """获取 GPU 统计信息"""
        gpu_memory = 0.0
        gpu_util = 0.0
        
        if _torch.cuda.is_available():
            try:
                # 显存使用
                gpu_memory = _torch.cuda.memory_allocated() / (1024 ** 3)  # GB
                
                # GPU 利用率（需要 nvidia-ml-py）
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = util.gpu
                except ImportError:
                    pass
            except Exception:
                pass
        
        return gpu_memory, gpu_util

    def _calc_eta(self, current_step: int, max_steps: int, elapsed: float) -> float:
        """计算预计剩余时间"""
        if current_step <= 0 or max_steps <= 0:
            return 0.0
        
        steps_per_second = current_step / elapsed
        remaining_steps = max_steps - current_step
        
        if steps_per_second > 0:
            return remaining_steps / steps_per_second
        return 0.0

    def _push_progress(self, progress: TrainingProgress):
        """推送进度（带节流）"""
        now = time.time()
        if now - self._last_log_time >= self._min_log_interval:
            self.progress_queue.append(progress)
            self._last_log_time = now

    def on_train_begin(self, args, state, control, **kwargs):
        """训练开始"""
        progress = TrainingProgress(
            job_id=self.job_id,
            status=JobStatus.RUNNING,
            total_epochs=self.total_epochs,
            message="训练开始",
        )
        self.progress_queue.append(progress)
        LOGGER.info("[%s] 训练开始", self.job_id)

    def on_train_end(self, args, state, control, **kwargs):
        """训练结束"""
        elapsed = time.time() - self.start_time
        progress = TrainingProgress(
            job_id=self.job_id,
            status=JobStatus.COMPLETED,
            current_step=state.global_step,
            total_steps=state.max_steps,
            current_epoch=int(state.epoch) if state.epoch else self.total_epochs,
            total_epochs=self.total_epochs,
            elapsed_seconds=elapsed,
            message="训练完成",
        )
        self.progress_queue.append(progress)
        LOGGER.info("[%s] 训练完成，耗时 %.1f 秒", self.job_id, elapsed)

    def on_log(self, args, state, control, logs=None, **kwargs):
        """日志记录时触发"""
        if logs is None:
            return
        
        elapsed = time.time() - self.start_time
        eta = self._calc_eta(state.global_step, state.max_steps, elapsed)
        gpu_memory, gpu_util = self._get_gpu_stats()
        
        progress = TrainingProgress(
            job_id=self.job_id,
            status=JobStatus.RUNNING,
            current_epoch=int(state.epoch) if state.epoch else 0,
            total_epochs=self.total_epochs,
            current_step=state.global_step,
            total_steps=state.max_steps,
            loss=logs.get("loss", 0.0),
            learning_rate=logs.get("learning_rate", 0.0),
            grad_norm=logs.get("grad_norm", 0.0),
            gpu_memory_used=gpu_memory,
            gpu_utilization=gpu_util,
            elapsed_seconds=elapsed,
            eta_seconds=eta,
            message=f"Step {state.global_step}/{state.max_steps}",
        )
        self._push_progress(progress)

    def on_step_end(self, args, state, control, **kwargs):
        """每步结束时触发"""
        # 检查是否需要早停（由外部设置）
        pass

    def on_save(self, args, state, control, **kwargs):
        """保存检查点时触发"""
        LOGGER.info(
            "[%s] 保存检查点: step=%d",
            self.job_id,
            state.global_step,
        )


class CheckpointCallback:
    """
    检查点回调
    
    处理检查点保存、恢复等逻辑
    """

    def __init__(
        self,
        job_id: str,
        output_dir: str,
        save_steps: int = 100,
        max_checkpoints: int = 3,
    ):
        _ensure_imports()
        self.job_id = job_id
        self.output_dir = output_dir
        self.save_steps = save_steps
        self.max_checkpoints = max_checkpoints
        self._saved_checkpoints: List[str] = []

    def on_save(self, args, state, control, **kwargs):
        """检查点保存时触发"""
        import os
        import glob
        
        # 获取所有检查点
        checkpoint_dirs = sorted(
            glob.glob(os.path.join(self.output_dir, "checkpoint-*")),
            key=lambda x: int(os.path.basename(x).split("-")[1]),
        )
        
        # 清理旧检查点
        while len(checkpoint_dirs) > self.max_checkpoints:
            oldest = checkpoint_dirs.pop(0)
            try:
                import shutil
                shutil.rmtree(oldest)
                LOGGER.info("[%s] 清理旧检查点: %s", self.job_id, oldest)
            except Exception as e:
                LOGGER.warning("[%s] 清理检查点失败: %s", self.job_id, e)


class EarlyStoppingCallback:
    """
    早停回调
    
    当验证损失不再下降时停止训练
    """

    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.0,
        metric_name: str = "eval_loss",
    ):
        _ensure_imports()
        self.patience = patience
        self.min_delta = min_delta
        self.metric_name = metric_name
        self.best_metric: Optional[float] = None
        self.counter = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """评估时触发"""
        if metrics is None:
            return
        
        current = metrics.get(self.metric_name)
        if current is None:
            return
        
        if self.best_metric is None:
            self.best_metric = current
            return
        
        if current < self.best_metric - self.min_delta:
            self.best_metric = current
            self.counter = 0
        else:
            self.counter += 1
            LOGGER.info(
                "EarlyStopping: %d/%d (best=%.4f, current=%.4f)",
                self.counter,
                self.patience,
                self.best_metric,
                current,
            )
            
            if self.counter >= self.patience:
                LOGGER.info("触发早停，停止训练")
                control.should_training_stop = True


class LoggingCallback:
    """
    日志回调
    
    将训练日志写入文件
    """

    def __init__(
        self,
        job_id: str,
        log_file: str,
    ):
        _ensure_imports()
        self.job_id = job_id
        self.log_file = log_file
        self._logs: List[Dict[str, Any]] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        """记录日志"""
        if logs is None:
            return
        
        log_entry = {
            "step": state.global_step,
            "epoch": state.epoch,
            "timestamp": datetime.now().isoformat(),
            **logs,
        }
        self._logs.append(log_entry)
        
        # 定期写入文件
        if len(self._logs) >= 10:
            self._flush()

    def on_train_end(self, args, state, control, **kwargs):
        """训练结束时刷新"""
        self._flush()

    def _flush(self):
        """写入文件"""
        import json
        
        try:
            with open(self.log_file, "a") as f:
                for log in self._logs:
                    f.write(json.dumps(log) + "\n")
            self._logs.clear()
        except Exception as e:
            LOGGER.warning("写入日志文件失败: %s", e)


def create_callback_wrapper(callback_instance) -> Any:
    """
    创建 TrainerCallback 包装器
    
    由于我们的回调类没有继承 TrainerCallback，
    这个函数创建一个包装器来适配 Trainer
    """
    _ensure_imports()
    
    class CallbackWrapper(_TrainerCallback):
        def __init__(self, wrapped):
            self._wrapped = wrapped
        
        def on_train_begin(self, args, state, control, **kwargs):
            if hasattr(self._wrapped, "on_train_begin"):
                self._wrapped.on_train_begin(args, state, control, **kwargs)
        
        def on_train_end(self, args, state, control, **kwargs):
            if hasattr(self._wrapped, "on_train_end"):
                self._wrapped.on_train_end(args, state, control, **kwargs)
        
        def on_log(self, args, state, control, logs=None, **kwargs):
            if hasattr(self._wrapped, "on_log"):
                self._wrapped.on_log(args, state, control, logs=logs, **kwargs)
        
        def on_step_end(self, args, state, control, **kwargs):
            if hasattr(self._wrapped, "on_step_end"):
                self._wrapped.on_step_end(args, state, control, **kwargs)
        
        def on_save(self, args, state, control, **kwargs):
            if hasattr(self._wrapped, "on_save"):
                self._wrapped.on_save(args, state, control, **kwargs)
        
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if hasattr(self._wrapped, "on_evaluate"):
                self._wrapped.on_evaluate(args, state, control, metrics=metrics, **kwargs)
    
    return CallbackWrapper(callback_instance)
