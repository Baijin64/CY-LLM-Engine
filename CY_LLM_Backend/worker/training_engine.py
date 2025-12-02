"""
training_servicer.py
[gRPC 训练服务] 实现 AiTrainingServicer，封装 LoRA 训练功能
"""

from __future__ import annotations

import logging
import os
import signal
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional

import grpc

LOGGER = logging.getLogger("ew.worker.training")

# 尝试导入训练相关依赖
TRAINING_AVAILABLE = False
BITSANDBYTES_AVAILABLE = False
CUDA_AVAILABLE = False

try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        TrainerCallback,
        TrainerState,
        TrainerControl,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    from trl import SFTTrainer
    from datasets import load_dataset, concatenate_datasets
    
    TRAINING_AVAILABLE = True
    
    # 尝试导入 BitsAndBytes (4bit 量化依赖)
    try:
        from transformers import BitsAndBytesConfig
        import bitsandbytes
        BITSANDBYTES_AVAILABLE = CUDA_AVAILABLE  # bitsandbytes 需要 CUDA
    except ImportError:
        BITSANDBYTES_AVAILABLE = False
        LOGGER.info("bitsandbytes 未安装，将使用非量化模式")
        # 创建占位 BitsAndBytesConfig
        BitsAndBytesConfig = None
        
except ImportError as e:
    LOGGER.warning("训练依赖未完全安装: %s", e)
    TRAINING_AVAILABLE = False


def get_device_info() -> dict:
    """获取设备信息"""
    info = {
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_device_name": None,
        "bitsandbytes_available": BITSANDBYTES_AVAILABLE,
    }
    if TRAINING_AVAILABLE:
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
    return info


class JobStatus(Enum):
    """训练任务状态"""
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
    current_epoch: int = 0
    total_epochs: int = 0
    current_step: int = 0
    total_steps: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    gpu_memory_used: float = 0.0
    gpu_utilization: float = 0.0
    elapsed_seconds: float = 0.0
    eta_seconds: float = 0.0
    message: str = ""
    error: str = ""

    @property
    def progress_percent(self) -> float:
        if self.total_steps > 0:
            return (self.current_step / self.total_steps) * 100
        return 0.0


@dataclass
class TrainingJob:
    """训练任务"""
    job_id: str
    base_model: str
    output_dir: str
    character_name: str
    status: JobStatus = JobStatus.QUEUED
    progress: Optional[TrainingProgress] = None
    output_path: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    error: Optional[str] = None
    # 新增：检查点恢复支持
    resume_from_checkpoint: Optional[str] = None
    # 新增：训练模式 (lora, full, custom)
    training_mode: str = "lora"


class ProgressCallback(TrainerCallback if TRAINING_AVAILABLE else object):
    """自定义回调，用于收集训练进度"""

    def __init__(self, job: TrainingJob, progress_queue: "list"):
        self.job = job
        self.progress_queue = progress_queue
        self.start_time = time.time()

    def on_log(self, args, state: "TrainerState", control: "TrainerControl", logs=None, **kwargs):
        if logs is None:
            return
        
        elapsed = time.time() - self.start_time
        eta = 0.0
        if state.global_step > 0 and state.max_steps > 0:
            eta = (elapsed / state.global_step) * (state.max_steps - state.global_step)

        # 获取 GPU 信息
        gpu_memory = 0.0
        gpu_util = 0.0
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
                # GPU 利用率需要 nvidia-smi，这里简化处理
            except Exception:
                pass

        progress = TrainingProgress(
            job_id=self.job.job_id,
            status=JobStatus.RUNNING,
            current_epoch=int(state.epoch) if state.epoch else 0,
            total_epochs=int(args.num_train_epochs),
            current_step=state.global_step,
            total_steps=state.max_steps,
            loss=logs.get("loss", 0.0),
            learning_rate=logs.get("learning_rate", 0.0),
            gpu_memory_used=gpu_memory,
            gpu_utilization=gpu_util,
            elapsed_seconds=elapsed,
            eta_seconds=eta,
            message=f"Step {state.global_step}/{state.max_steps}",
        )
        self.progress_queue.append(progress)

    def on_train_begin(self, args, state, control, **kwargs):
        progress = TrainingProgress(
            job_id=self.job.job_id,
            status=JobStatus.RUNNING,
            message="训练开始",
        )
        self.progress_queue.append(progress)

    def on_train_end(self, args, state, control, **kwargs):
        progress = TrainingProgress(
            job_id=self.job.job_id,
            status=JobStatus.COMPLETED,
            current_step=state.global_step,
            total_steps=state.max_steps,
            elapsed_seconds=time.time() - self.start_time,
            message="训练完成",
        )
        self.progress_queue.append(progress)


class TrainingEngine:
    """训练引擎，管理训练任务"""

    # 默认任务保留天数
    DEFAULT_JOB_RETENTION_DAYS = 7
    # 默认清理间隔（秒）
    DEFAULT_CLEANUP_INTERVAL = 3600

    def __init__(
        self,
        max_concurrent_jobs: int = 1,
        checkpoint_dir: str = "./checkpoints",
        job_retention_days: int = DEFAULT_JOB_RETENTION_DAYS,
    ):
        self._jobs: Dict[str, TrainingJob] = {}
        self._lock = threading.Lock()
        self._max_concurrent = max_concurrent_jobs
        self._checkpoint_dir = checkpoint_dir
        self._job_retention_days = job_retention_days
        
        # 使用 Semaphore 进行并发控制（替代简单计数）
        self._semaphore = threading.Semaphore(max_concurrent_jobs)
        self._running_count = 0  # 保留用于状态查询
        
        # 启动后台清理线程
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

    def _cleanup_loop(self):
        """后台清理循环：定期清理过期任务"""
        while not self._cleanup_stop_event.wait(timeout=self.DEFAULT_CLEANUP_INTERVAL):
            self._cleanup_old_jobs()

    def _cleanup_old_jobs(self):
        """清理过期的已完成任务，防止内存泄漏"""
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
            LOGGER.info("已清理 %d 个过期任务（超过 %d 天）", removed_count, self._job_retention_days)

    def shutdown(self):
        """关闭训练引擎，停止后台线程"""
        self._cleanup_stop_event.set()
        if self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)
        LOGGER.info("TrainingEngine 已关闭")

    def create_job(
        self,
        base_model: str,
        output_dir: str,
        character_name: str,
        job_id: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None,
        training_mode: str = "lora",
    ) -> TrainingJob:
        """
        创建训练任务
        
        Args:
            base_model: 基础模型路径或 HuggingFace ID
            output_dir: 输出目录
            character_name: 角色名称
            job_id: 任务 ID（可选）
            resume_from_checkpoint: 检查点路径，用于恢复训练
            training_mode: 训练模式 (lora, full, custom)
        """
        job_id = job_id or str(uuid.uuid4())
        job = TrainingJob(
            job_id=job_id,
            base_model=base_model,
            output_dir=output_dir,
            character_name=character_name,
            resume_from_checkpoint=resume_from_checkpoint,
            training_mode=training_mode,
        )
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """获取训练任务"""
        return self._jobs.get(job_id)

    def list_jobs(self, status_filter: Optional[List[JobStatus]] = None, limit: int = 100) -> List[TrainingJob]:
        """列出训练任务"""
        jobs = list(self._jobs.values())
        if status_filter:
            jobs = [j for j in jobs if j.status in status_filter]
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def list_checkpoints(self, job_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        列出可用的检查点
        
        Args:
            job_id: 任务 ID（可选，不指定则列出所有）
            
        Returns:
            检查点列表，每个包含 path, job_id, step, timestamp
        """
        import os
        import glob
        
        checkpoints = []
        
        # 搜索检查点目录
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
            
            # 查找 checkpoint-* 目录
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
        
        # 按步数排序
        checkpoints.sort(key=lambda c: c["step"], reverse=True)
        return checkpoints

    def find_latest_checkpoint(self, output_dir: str) -> Optional[str]:
        """
        查找最新的检查点
        
        Args:
            output_dir: 输出目录
            
        Returns:
            最新检查点的路径，如果没有则返回 None
        """
        import os
        import glob
        
        if not os.path.exists(output_dir):
            return None
        
        checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
        if not checkpoints:
            return None
        
        # 按步数排序
        def get_step(path):
            try:
                return int(os.path.basename(path).split("-")[1])
            except (ValueError, IndexError):
                return 0
        
        checkpoints.sort(key=get_step, reverse=True)
        return checkpoints[0] if checkpoints else None

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

    def run_training(
        self,
        job: TrainingJob,
        dataset_path: str,
        settings_path: Optional[str] = None,
        lora_config: Optional[Dict[str, Any]] = None,
        quant_config: Optional[Dict[str, Any]] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
    ) -> Generator[TrainingProgress, None, None]:
        """运行训练任务，生成进度流"""
        if not TRAINING_AVAILABLE:
            job.status = JobStatus.FAILED
            job.error = "训练依赖未安装"
            yield TrainingProgress(
                job_id=job.job_id,
                status=JobStatus.FAILED,
                error="训练依赖未安装（torch, transformers, peft, trl）",
            )
            return

        progress_queue: List[TrainingProgress] = []
        
        # 默认配置
        lora_config = lora_config or {}
        quant_config = quant_config or {}
        hyperparams = hyperparams or {}

        def _train_thread():
            nonlocal progress_queue
            # 使用 Semaphore 进行并发控制
            acquired = self._semaphore.acquire(blocking=False)
            if not acquired:
                # 无法获取信号量，任务排队等待
                LOGGER.info("[%s] 等待其他训练任务完成...", job.job_id)
                job.status = JobStatus.QUEUED
                progress_queue.append(TrainingProgress(
                    job_id=job.job_id,
                    status=JobStatus.QUEUED,
                    message="排队中，等待其他训练任务完成",
                ))
                # 阻塞等待
                self._semaphore.acquire(blocking=True)
            
            try:
                with self._lock:
                    self._running_count += 1
                job.status = JobStatus.RUNNING
                job.updated_at = datetime.now()

                # 检测设备能力
                device_info = get_device_info()
                use_cuda = device_info["cuda_available"]
                use_quantization = device_info["bitsandbytes_available"] and quant_config.get("use_4bit", True)
                
                LOGGER.info(
                    "[%s] 设备信息: CUDA=%s, 量化=%s",
                    job.job_id,
                    use_cuda,
                    use_quantization
                )

                # 配置模型加载参数
                model_kwargs = {
                    "trust_remote_code": True,
                }
                
                # 配置量化（仅在 CUDA + bitsandbytes 可用时）
                if use_quantization and BitsAndBytesConfig is not None:
                    LOGGER.info("[%s] 使用 4bit 量化加载", job.job_id)
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type=quant_config.get("bnb_4bit_quant_type", "nf4"),
                        bnb_4bit_compute_dtype=getattr(
                            torch, quant_config.get("bnb_4bit_compute_dtype", "bfloat16")
                        ),
                        bnb_4bit_use_double_quant=True,
                    )
                    model_kwargs["quantization_config"] = bnb_config
                    model_kwargs["device_map"] = "auto"
                elif use_cuda:
                    # CUDA 可用但不使用量化
                    LOGGER.info("[%s] 使用 CUDA 加载 (无量化)", job.job_id)
                    model_kwargs["device_map"] = "auto"
                    model_kwargs["torch_dtype"] = torch.float16
                else:
                    # CPU 模式
                    LOGGER.info("[%s] 使用 CPU 加载 (无量化)", job.job_id)
                    model_kwargs["device_map"] = "cpu"
                    model_kwargs["torch_dtype"] = torch.float32
                    # CPU 模式下使用低内存设置
                    model_kwargs["low_cpu_mem_usage"] = True

                # 加载模型
                LOGGER.info("[%s] 加载基础模型: %s", job.job_id, job.base_model)
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        job.base_model,
                        **model_kwargs
                    )
                except Exception as load_err:
                    LOGGER.warning(
                        "[%s] 首次加载失败 (%s)，尝试 CPU fallback",
                        job.job_id,
                        load_err
                    )
                    # Fallback: 强制 CPU 加载
                    model = AutoModelForCausalLM.from_pretrained(
                        job.base_model,
                        device_map="cpu",
                        torch_dtype=torch.float32,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                    )
                
                model.config.use_cache = False
                
                # 仅在量化模式下调用 prepare_model_for_kbit_training
                if use_quantization:
                    model = prepare_model_for_kbit_training(model)

                # 配置 LoRA
                peft_config = LoraConfig(
                    r=lora_config.get("r", 64),
                    lora_alpha=lora_config.get("lora_alpha", 16),
                    target_modules=lora_config.get(
                        "target_modules",
                        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                    ),
                    lora_dropout=lora_config.get("lora_dropout", 0.05),
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                )
                model = get_peft_model(model, peft_config)

                # 加载 Tokenizer
                tokenizer = AutoTokenizer.from_pretrained(job.base_model, trust_remote_code=True)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                tokenizer.padding_side = "right"

                # 加载数据集
                LOGGER.info("[%s] 加载数据集: %s", job.job_id, dataset_path)
                dialogue_ds = load_dataset("json", data_files=dataset_path, split="train")
                
                if settings_path:
                    settings_ds = load_dataset("json", data_files=settings_path, split="train")
                    oversample = int(hyperparams.get("setting_oversample", 5))
                    settings_ds_oversampled = concatenate_datasets([settings_ds] * oversample)
                    dataset = concatenate_datasets([dialogue_ds, settings_ds_oversampled])
                    dataset = dataset.shuffle(seed=42)
                else:
                    dataset = dialogue_ds

                # 格式化函数
                def formatting_prompts_func(example):
                    output_texts = []
                    for i in range(len(example['instruction'])):
                        instruction = example['instruction'][i]
                        input_str = example['input'][i]
                        output_str = example['output'][i]
                        
                        if input_str:
                            text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_str}\n\n### Response:\n{output_str}<|endoftext|>"
                        else:
                            text = f"### Instruction:\n{instruction}\n\n### Response:\n{output_str}<|endoftext|>"
                        output_texts.append(text)
                    return output_texts

                # 训练参数 - 根据设备能力调整
                # 检查是否支持 bf16
                use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
                use_fp16 = use_cuda and not use_bf16
                
                # 根据设备选择优化器
                if use_quantization:
                    optim = "paged_adamw_32bit"
                elif use_cuda:
                    optim = "adamw_torch"
                else:
                    optim = "adamw_torch"  # CPU 使用标准 adamw
                
                LOGGER.info(
                    "[%s] 训练配置: bf16=%s, fp16=%s, optim=%s",
                    job.job_id,
                    use_bf16,
                    use_fp16,
                    optim
                )
                
                training_args = TrainingArguments(
                    output_dir=job.output_dir,
                    per_device_train_batch_size=hyperparams.get("per_device_batch_size", 2),
                    gradient_accumulation_steps=hyperparams.get("gradient_accumulation_steps", 8),
                    learning_rate=hyperparams.get("learning_rate", 2e-4),
                    logging_steps=hyperparams.get("logging_steps", 10),
                    num_train_epochs=hyperparams.get("num_train_epochs", 3),
                    save_steps=hyperparams.get("save_steps", 100),
                    fp16=use_fp16,
                    bf16=use_bf16,
                    optim=optim,
                    report_to="none",
                    gradient_checkpointing=use_cuda,  # CPU 模式关闭 gradient checkpointing
                    max_grad_norm=0.3,
                    warmup_ratio=hyperparams.get("warmup_ratio", 0.03),
                    lr_scheduler_type="cosine",
                    # CPU 特殊设置
                    dataloader_pin_memory=use_cuda,
                    use_cpu=not use_cuda,
                )

                # 创建回调
                callback = ProgressCallback(job, progress_queue)

                # 创建 Trainer
                trainer = SFTTrainer(
                    model=model,
                    train_dataset=dataset,
                    peft_config=peft_config,
                    formatting_func=formatting_prompts_func,
                    args=training_args,
                    tokenizer=tokenizer,
                    max_seq_length=hyperparams.get("max_seq_length", 2048),
                    packing=False,
                    callbacks=[callback],
                )

                # 开始训练（支持检查点恢复）
                resume_checkpoint = job.resume_from_checkpoint
                if resume_checkpoint == "auto":
                    # 自动查找最新检查点
                    resume_checkpoint = self.find_latest_checkpoint(job.output_dir)
                    if resume_checkpoint:
                        LOGGER.info("[%s] 自动恢复从检查点: %s", job.job_id, resume_checkpoint)
                
                if resume_checkpoint:
                    LOGGER.info("[%s] 从检查点恢复训练: %s", job.job_id, resume_checkpoint)
                    trainer.train(resume_from_checkpoint=resume_checkpoint)
                else:
                    LOGGER.info("[%s] 开始训练", job.job_id)
                    trainer.train()

                # 保存模型
                LOGGER.info("[%s] 保存模型到 %s", job.job_id, job.output_dir)
                trainer.save_model(job.output_dir)

                job.status = JobStatus.COMPLETED
                job.output_path = job.output_dir
                job.updated_at = datetime.now()

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
                # 释放信号量，允许下一个任务执行
                self._semaphore.release()

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

        # 等待线程结束
        train_thread.join(timeout=5)


# 全局训练引擎实例
_training_engine: Optional[TrainingEngine] = None


def get_training_engine() -> TrainingEngine:
    """获取全局训练引擎"""
    global _training_engine
    if _training_engine is None:
        _training_engine = TrainingEngine()
    return _training_engine