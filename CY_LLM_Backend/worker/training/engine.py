"""
training.engine - 训练引擎核心类
协调数据加载、模型设置、训练循环等模块
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Generator, List, Optional

from .data import DatasetConfig, DatasetLoader, InstructionFormatter, FormatterConfig
from .model import ModelSetup, ModelConfig, DeviceConfig, QuantizationConfig, LoRASetup, LoRAConfig
from .loop import (
    TrainerFactory,
    TrainerConfig,
    TrainerConfigBuilder,
    ProgressCallback,
    CheckpointCallback,
    TrainingProgress,
    JobStatus,
    create_callback_wrapper,
)

LOGGER = logging.getLogger("cy_llm.worker.training.engine")


@dataclass
class TrainingJob:
    """训练任务"""
    job_id: str
    base_model: str
    output_dir: str
    character_name: str
    
    # 状态
    status: JobStatus = JobStatus.QUEUED
    progress: Optional[TrainingProgress] = None
    output_path: Optional[str] = None
    error: Optional[str] = None
    
    # 时间
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    
    # 控制
    cancel_event: threading.Event = field(default_factory=threading.Event)
    
    # 选项
    resume_from_checkpoint: Optional[str] = None
    training_mode: str = "lora"  # lora, full, custom


@dataclass
class TrainingRequest:
    """训练请求（聚合所有配置）"""
    # 必需
    base_model: str
    output_dir: str
    dataset_path: str
    character_name: str = ""
    
    # 数据配置
    settings_path: Optional[str] = None
    settings_oversample: int = 5
    
    # LoRA 配置
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None
    
    # 量化配置
    use_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    
    # 训练超参数
    num_train_epochs: int = 3
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    max_seq_length: int = 2048
    logging_steps: int = 10
    save_steps: int = 100
    
    # 选项
    resume_from_checkpoint: Optional[str] = None
    job_id: Optional[str] = None


class TrainingEngine:
    """
    训练引擎
    
    协调各模块完成完整的训练流程：
    1. 数据加载和格式化
    2. 模型加载和量化
    3. LoRA 配置和应用
    4. 训练循环执行
    5. 检查点管理
    
    示例:
        >>> engine = TrainingEngine()
        >>> request = TrainingRequest(
        ...     base_model="deepseek-ai/deepseek-llm-7b",
        ...     output_dir="/output",
        ...     dataset_path="/data/train.json",
        ... )
        >>> for progress in engine.train(request):
        ...     print(f"Step {progress.current_step}: loss={progress.loss:.4f}")
    """

    # 默认配置
    DEFAULT_JOB_RETENTION_DAYS = 7
    DEFAULT_CLEANUP_INTERVAL = 3600

    def __init__(
        self,
        max_concurrent_jobs: int = 1,
        checkpoint_dir: str = "./checkpoints",
        job_retention_days: int = DEFAULT_JOB_RETENTION_DAYS,
    ):
        self._jobs: Dict[str, TrainingJob] = {}
        self._lock = threading.Lock()
        self._checkpoint_dir = checkpoint_dir
        self._job_retention_days = job_retention_days
        
        # 并发控制
        self._semaphore = threading.Semaphore(max_concurrent_jobs)
        self._running_count = 0
        
        # 后台清理
        self._cleanup_stop_event = threading.Event()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="TrainingEngine-Cleanup"
        )
        self._cleanup_thread.start()
        
        LOGGER.info(
            "TrainingEngine 初始化: max_concurrent=%d, retention_days=%d",
            max_concurrent_jobs,
            job_retention_days
        )

    def train(self, request: TrainingRequest) -> Generator[TrainingProgress, None, None]:
        """
        执行训练
        
        Args:
            request: 训练请求
            
        Yields:
            TrainingProgress: 训练进度
        """
        # 创建任务
        job = self._create_job(request)
        
        # 进度队列
        progress_queue: List[TrainingProgress] = []
        
        def _train_thread():
            nonlocal progress_queue
            
            # 获取信号量
            acquired = self._semaphore.acquire(blocking=False)
            if not acquired:
                LOGGER.info("[%s] 等待其他训练任务完成...", job.job_id)
                job.status = JobStatus.QUEUED
                progress_queue.append(TrainingProgress(
                    job_id=job.job_id,
                    status=JobStatus.QUEUED,
                    message="排队中，等待其他训练任务完成",
                ))
                self._semaphore.acquire(blocking=True)
            
            try:
                with self._lock:
                    self._running_count += 1
                
                job.status = JobStatus.RUNNING
                job.started_at = datetime.now()
                job.updated_at = datetime.now()
                
                self._execute_training(job, request, progress_queue)
                
            except Exception as e:
                LOGGER.exception("[%s] 训练失败: %s", job.job_id, e)
                job.status = JobStatus.FAILED
                job.error = str(e)
                job.updated_at = datetime.now()
                progress_queue.append(TrainingProgress(
                    job_id=job.job_id,
                    status=JobStatus.FAILED,
                    error=str(e),
                ))
            finally:
                with self._lock:
                    self._running_count -= 1
                self._semaphore.release()
                job.finished_at = datetime.now()
        
        # 启动训练线程
        train_thread = threading.Thread(target=_train_thread, daemon=True)
        train_thread.start()
        
        # 生成进度流
        last_idx = 0
        while train_thread.is_alive() or last_idx < len(progress_queue):
            # 检查取消
            if job.cancel_event.is_set():
                yield TrainingProgress(
                    job_id=job.job_id,
                    status=JobStatus.CANCELLED,
                    message="训练已取消",
                )
                break
            
            # 发送新进度
            while last_idx < len(progress_queue):
                yield progress_queue[last_idx]
                last_idx += 1
            
            time.sleep(0.5)
        
        train_thread.join(timeout=5)

    # ==================== 训练执行辅助方法 ====================

    def _detect_device(self, job_id: str) -> DeviceConfig:
        """检测设备配置"""
        device_config = DeviceConfig.detect()
        LOGGER.info(
            "[%s] 设备检测: CUDA=%s, BnB=%s, bf16=%s",
            job_id,
            device_config.cuda_available,
            device_config.bitsandbytes_available,
            device_config.bf16_supported,
        )
        return device_config

    def _load_dataset(self, job_id: str, request: TrainingRequest):
        """加载并准备数据集"""
        LOGGER.info("[%s] 加载数据集: %s", job_id, request.dataset_path)
        dataset_config = DatasetConfig(
            path=request.dataset_path,
            settings_path=request.settings_path,
            settings_oversample=request.settings_oversample,
        )
        dataset_loader = DatasetLoader(dataset_config)
        dataset = dataset_loader.load()
        LOGGER.info("[%s] 数据集加载完成: %d 条样本", job_id, len(dataset))
        return dataset

    def _load_model_with_quantization(
        self,
        job_id: str,
        request: TrainingRequest,
        device_config: DeviceConfig,
    ):
        """加载模型并应用量化配置"""
        LOGGER.info("[%s] 加载模型: %s", job_id, request.base_model)
        
        # 量化配置
        use_quantization = (
            request.use_4bit and
            device_config.bitsandbytes_available
        )
        quant_config = QuantizationConfig(
            enabled=use_quantization,
            bnb_4bit_quant_type=request.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=request.bnb_4bit_compute_dtype,
        ) if use_quantization else None
        
        model_config = ModelConfig(
            model_path=request.base_model,
            quantization=quant_config,
        )
        model_setup = ModelSetup(model_config, device_config)
        model, tokenizer = model_setup.load()
        
        return model, tokenizer, use_quantization

    def _apply_lora(self, job_id: str, request: TrainingRequest, model, use_quantization: bool):
        """应用 LoRA 配置"""
        LOGGER.info("[%s] 配置 LoRA", job_id)
        lora_config = LoRAConfig(
            r=request.lora_r,
            lora_alpha=request.lora_alpha,
            lora_dropout=request.lora_dropout,
            target_modules=request.lora_target_modules,
        )
        lora_setup = LoRASetup(lora_config)
        return lora_setup.apply(model, is_quantized=use_quantization)

    def _create_trainer(
        self,
        job_id: str,
        request: TrainingRequest,
        model,
        tokenizer,
        dataset,
        progress_queue: List[TrainingProgress],
    ):
        """创建并配置 Trainer"""
        LOGGER.info("[%s] 创建 Trainer", job_id)
        
        # 构建配置
        trainer_config = (
            TrainerConfigBuilder(request.output_dir)
            .with_epochs(request.num_train_epochs)
            .with_batch_size(
                request.per_device_batch_size,
                gradient_accumulation=request.gradient_accumulation_steps,
            )
            .with_learning_rate(request.learning_rate, warmup_ratio=request.warmup_ratio)
            .with_max_seq_length(request.max_seq_length)
            .with_logging(
                logging_steps=request.logging_steps,
                save_steps=request.save_steps,
            )
            .build()
        )
        
        # 创建回调
        progress_callback = ProgressCallback(
            job_id=job_id,
            progress_queue=progress_queue,
            total_epochs=request.num_train_epochs,
        )
        callbacks = [create_callback_wrapper(progress_callback)]
        
        # 创建格式化器和 Trainer
        formatter = InstructionFormatter(FormatterConfig())
        trainer_factory = TrainerFactory(trainer_config)
        trainer = trainer_factory.create(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            formatting_func=formatter,
            callbacks=callbacks,
        )
        
        return trainer, trainer_factory

    def _run_training(
        self,
        job_id: str,
        trainer,
        trainer_factory,
        resume_from_checkpoint: Optional[str],
    ) -> None:
        """执行训练循环"""
        resume_checkpoint = resume_from_checkpoint
        if resume_checkpoint == "auto":
            resume_checkpoint = trainer_factory.find_checkpoint()
        
        if resume_checkpoint:
            LOGGER.info("[%s] 从检查点恢复: %s", job_id, resume_checkpoint)
            trainer.train(resume_from_checkpoint=resume_checkpoint)
        else:
            LOGGER.info("[%s] 开始训练", job_id)
            trainer.train()

    def _save_model(self, job_id: str, trainer, output_dir: str) -> None:
        """保存训练后的模型"""
        LOGGER.info("[%s] 保存模型到 %s", job_id, output_dir)
        trainer.save_model(output_dir)

    def _execute_training(
        self,
        job: TrainingJob,
        request: TrainingRequest,
        progress_queue: List[TrainingProgress],
    ) -> None:
        """
        执行实际的训练逻辑
        
        训练流程：
        1. 检测设备
        2. 加载数据集
        3. 加载模型（含量化）
        4. 应用 LoRA
        5. 创建 Trainer
        6. 执行训练
        7. 保存模型
        """
        job_id = job.job_id
        
        # 1. 检测设备
        device_config = self._detect_device(job_id)
        
        # 2. 加载数据集
        dataset = self._load_dataset(job_id, request)
        
        # 3. 加载模型（含量化）
        model, tokenizer, use_quantization = self._load_model_with_quantization(
            job_id, request, device_config
        )
        
        # 4. 应用 LoRA
        model = self._apply_lora(job_id, request, model, use_quantization)
        
        # 5. 创建 Trainer
        trainer, trainer_factory = self._create_trainer(
            job_id, request, model, tokenizer, dataset, progress_queue
        )
        
        # 6. 执行训练
        self._run_training(job_id, trainer, trainer_factory, request.resume_from_checkpoint)
        
        # 7. 保存模型
        self._save_model(job_id, trainer, request.output_dir)
        
        # 更新任务状态
        job.status = JobStatus.COMPLETED
        job.output_path = request.output_dir
        job.updated_at = datetime.now()

    def _create_job(self, request: TrainingRequest) -> TrainingJob:
        """创建训练任务"""
        job_id = request.job_id or str(uuid.uuid4())
        job = TrainingJob(
            job_id=job_id,
            base_model=request.base_model,
            output_dir=request.output_dir,
            character_name=request.character_name,
            resume_from_checkpoint=request.resume_from_checkpoint,
        )
        with self._lock:
            self._jobs[job_id] = job
        return job

    # ==================== 任务管理 ====================

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """获取训练任务"""
        return self._jobs.get(job_id)

    def list_jobs(
        self,
        status_filter: Optional[List[JobStatus]] = None,
        limit: int = 100,
    ) -> List[TrainingJob]:
        """列出训练任务"""
        jobs = list(self._jobs.values())
        if status_filter:
            jobs = [j for j in jobs if j.status in status_filter]
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def cancel_job(self, job_id: str) -> bool:
        """取消训练任务"""
        job = self._jobs.get(job_id)
        if job is None:
            return False
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            return False
        job.cancel_event.set()
        job.status = JobStatus.CANCELLED
        job.updated_at = datetime.now()
        return True

    def get_stats(self) -> Dict[str, Any]:
        """获取引擎统计信息"""
        with self._lock:
            jobs_by_status = {}
            for job in self._jobs.values():
                status = job.status.value
                jobs_by_status[status] = jobs_by_status.get(status, 0) + 1
            
            return {
                "total_jobs": len(self._jobs),
                "running_count": self._running_count,
                "jobs_by_status": jobs_by_status,
            }

    # ==================== 检查点管理 ====================

    def list_checkpoints(self, job_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出检查点"""
        import glob
        
        checkpoints = []
        search_paths = []
        
        if job_id:
            job = self._jobs.get(job_id)
            if job:
                search_paths.append(job.output_dir)
        else:
            search_paths.append(self._checkpoint_dir)
            for job in self._jobs.values():
                search_paths.append(job.output_dir)
        
        for base_path in search_paths:
            if not os.path.exists(base_path):
                continue
            
            for checkpoint_dir in glob.glob(os.path.join(base_path, "checkpoint-*")):
                if os.path.isdir(checkpoint_dir):
                    try:
                        step = int(os.path.basename(checkpoint_dir).split("-")[1])
                        mtime = os.path.getmtime(checkpoint_dir)
                        checkpoints.append({
                            "path": checkpoint_dir,
                            "job_id": job_id or "unknown",
                            "step": step,
                            "timestamp": datetime.fromtimestamp(mtime).isoformat(),
                        })
                    except (ValueError, IndexError):
                        pass
        
        checkpoints.sort(key=lambda c: c["step"], reverse=True)
        return checkpoints

    # ==================== 生命周期 ====================

    def shutdown(self):
        """关闭训练引擎"""
        self._cleanup_stop_event.set()
        if self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)
        LOGGER.info("TrainingEngine 已关闭")

    def _cleanup_loop(self):
        """后台清理循环"""
        while not self._cleanup_stop_event.wait(timeout=self.DEFAULT_CLEANUP_INTERVAL):
            self._cleanup_old_jobs()

    def _cleanup_old_jobs(self):
        """清理过期任务"""
        cutoff = datetime.now() - timedelta(days=self._job_retention_days)
        removed_count = 0
        
        with self._lock:
            jobs_to_remove = [
                job_id for job_id, job in self._jobs.items()
                if job.status not in (JobStatus.QUEUED, JobStatus.RUNNING)
                and job.updated_at < cutoff
            ]
            for job_id in jobs_to_remove:
                del self._jobs[job_id]
                removed_count += 1
        
        if removed_count > 0:
            LOGGER.info("已清理 %d 个过期任务", removed_count)


# 全局实例
_training_engine: Optional[TrainingEngine] = None


def get_training_engine() -> TrainingEngine:
    """获取全局训练引擎"""
    global _training_engine
    if _training_engine is None:
        _training_engine = TrainingEngine()
    return _training_engine
