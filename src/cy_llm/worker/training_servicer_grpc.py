"""
training_servicer_grpc.py
[gRPC 训练服务实现] 实现 AiTrainingServicer
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Generator, Optional, cast, List

import grpc

from .proto_gen import (
    AiTrainingServicer,
    TrainingRequest,
    TrainingProgress as TrainingProgressProto,
    TrainingStatusRequest,
    TrainingStatusResponse,
    CancelTrainingRequest,
    CancelTrainingResponse,
    ListTrainingJobsRequest,
    ListTrainingJobsResponse,
    TrainingJobSummary,
    TrainingStatus as TrainingStatusProto,
    add_AiTrainingServicer_to_server,
    CustomScriptExecutionRequest,
    CustomScriptExecutionProgress as CustomScriptExecutionProgressProto,
    CancelCustomScriptRequest,
    CancelCustomScriptResponse,
)
from .training_engine import (
    TrainingEngine,
    TrainingJob,
    TrainingProgress,
    JobStatus,
    get_training_engine,
)
from .training.custom_script_runner import CustomScriptRunner, ScriptStatus, ScriptProgress

LOGGER = logging.getLogger("cy_llm.worker.training.grpc")

# 内部认证 Token
INTERNAL_TOKEN = os.getenv("CY_LLM_INTERNAL_TOKEN", "")


def _verify_internal_token(context: grpc.ServicerContext) -> bool:
    """验证内部 Token"""
    if not INTERNAL_TOKEN:
        return True
    
    metadata = dict(context.invocation_metadata())
    auth_header = metadata.get("authorization", "")
    
    if auth_header == f"Bearer {INTERNAL_TOKEN}":
        return True
    
    LOGGER.warning("训练服务内部认证失败")
    context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid internal token")
    return False


def _job_status_to_proto(status: JobStatus) -> TrainingStatusProto:
    """转换任务状态到 Proto 枚举"""
    mapping = {
        JobStatus.QUEUED: TrainingStatusProto.TRAINING_STATUS_QUEUED,
        JobStatus.RUNNING: TrainingStatusProto.TRAINING_STATUS_RUNNING,
        JobStatus.COMPLETED: TrainingStatusProto.TRAINING_STATUS_COMPLETED,
        JobStatus.FAILED: TrainingStatusProto.TRAINING_STATUS_FAILED,
        JobStatus.CANCELLED: TrainingStatusProto.TRAINING_STATUS_CANCELLED,
    }
    return cast(TrainingStatusProto, mapping.get(status, TrainingStatusProto.TRAINING_STATUS_UNKNOWN))


def _progress_to_proto(progress: TrainingProgress) -> TrainingProgressProto:
    """转换进度对象到 Proto 消息"""
    return TrainingProgressProto(
        job_id=progress.job_id,
        status=cast(TrainingStatusProto, _job_status_to_proto(progress.status)),
        current_epoch=progress.current_epoch,
        total_epochs=progress.total_epochs,
        current_step=progress.current_step,
        total_steps=progress.total_steps,
        progress_percent=progress.progress_percent,
        loss=progress.loss,
        learning_rate=progress.learning_rate,
        gpu_memory_used=progress.gpu_memory_used,
        gpu_utilization=progress.gpu_utilization,
        elapsed_seconds=progress.elapsed_seconds,
        eta_seconds=progress.eta_seconds,
        message=progress.message,
        error=progress.error,
    )


class AiTrainingServicerImpl(AiTrainingServicer):
    """AiTraining gRPC 服务实现"""

    def __init__(self, engine: Optional[TrainingEngine] = None):
        self._engine = engine if engine is not None else get_training_engine()
        self._script_runner = CustomScriptRunner(max_concurrent=2)

    def StartTraining(
        self,
        request: TrainingRequest,
        context: grpc.ServicerContext,
    ) -> Generator[TrainingProgressProto, None, None]:
        """启动训练任务"""
        if not _verify_internal_token(context):
            return

        job_id = request.job_id or str(uuid.uuid4())
        trace_id = request.metadata.trace_id if request.metadata else job_id

        LOGGER.info(
            "[%s] 收到训练请求: model=%s character=%s",
            trace_id, request.base_model, request.character_name,
        )

        # 创建训练任务
        job = self._engine.create_job(
            base_model=request.base_model,
            output_dir=request.output_dir or f"/checkpoints/{request.character_name}_lora",
            character_name=request.character_name,
            job_id=job_id,
        )

        # 解析配置
        lora_config = {}
        if request.lora:
            if request.lora.r > 0:
                lora_config["r"] = request.lora.r
            if request.lora.lora_alpha > 0:
                lora_config["lora_alpha"] = request.lora.lora_alpha
            if request.lora.lora_dropout > 0:
                lora_config["lora_dropout"] = request.lora.lora_dropout
            if request.lora.target_modules:
                lora_config["target_modules"] = list(request.lora.target_modules)

        quant_config = {}
        if request.quantization:
            quant_config["use_4bit"] = request.quantization.use_4bit
            if request.quantization.bnb_4bit_compute_dtype:
                quant_config["bnb_4bit_compute_dtype"] = request.quantization.bnb_4bit_compute_dtype
            if request.quantization.bnb_4bit_quant_type:
                quant_config["bnb_4bit_quant_type"] = request.quantization.bnb_4bit_quant_type

        hyperparams = {}
        if request.hyperparams:
            hp = request.hyperparams
            if hp.num_train_epochs > 0:
                hyperparams["num_train_epochs"] = hp.num_train_epochs
            if hp.per_device_batch_size > 0:
                hyperparams["per_device_batch_size"] = hp.per_device_batch_size
            if hp.gradient_accumulation_steps > 0:
                hyperparams["gradient_accumulation_steps"] = hp.gradient_accumulation_steps
            if hp.learning_rate > 0:
                hyperparams["learning_rate"] = hp.learning_rate
            if hp.warmup_ratio > 0:
                hyperparams["warmup_ratio"] = hp.warmup_ratio
            if hp.max_seq_length > 0:
                hyperparams["max_seq_length"] = hp.max_seq_length
            if hp.save_steps > 0:
                hyperparams["save_steps"] = hp.save_steps
            if hp.logging_steps > 0:
                hyperparams["logging_steps"] = hp.logging_steps

        # 数据集配置
        dataset_path = None
        if request.dataset and request.dataset.path:
            dataset_path = request.dataset.path
        elif request.metadata and request.metadata.extra:
            dataset_path = request.metadata.extra.get("dataset_path")
        settings_path = None
        if request.metadata and request.metadata.extra:
            settings_path = request.metadata.extra.get("settings_path")

        if not dataset_path:
            LOGGER.error("[%s] 缺少数据集路径", trace_id)
            yield TrainingProgressProto(
                job_id=job_id,
                status=TrainingStatusProto.TRAINING_STATUS_FAILED,
                error="缺少数据集路径",
            )
            return

        # 运行训练
        try:
            for progress in self._engine.run_training(
                job=job,
                dataset_path=dataset_path,
                settings_path=settings_path,
                lora_config=lora_config or None,
                quant_config=quant_config or None,
                hyperparams=hyperparams or None,
            ):
                yield _progress_to_proto(progress)
        except Exception as exc:
            LOGGER.exception("[%s] 训练异常: %s", trace_id, exc)
            yield TrainingProgressProto(
                job_id=job_id,
                status=TrainingStatusProto.TRAINING_STATUS_FAILED,
                error=str(exc),
            )

    def GetTrainingStatus(
        self,
        request: TrainingStatusRequest,
        context: grpc.ServicerContext,
    ) -> TrainingStatusResponse:
        """查询训练状态"""
        if not _verify_internal_token(context):
            return TrainingStatusResponse()

        job = self._engine.get_job(request.job_id)
        if job is None:
            context.abort(grpc.StatusCode.NOT_FOUND, f"Job {request.job_id} not found")
            return TrainingStatusResponse()

        response = TrainingStatusResponse(
            job_id=job.job_id,
            status=cast(TrainingStatusProto, _job_status_to_proto(job.status)),
            output_path=job.output_path or "",
        )

        if job.progress:
            response.latest_progress.CopyFrom(_progress_to_proto(job.progress))

        return response

    def CancelTraining(
        self,
        request: CancelTrainingRequest,
        context: grpc.ServicerContext,
    ) -> CancelTrainingResponse:
        """取消训练任务"""
        if not _verify_internal_token(context):
            return CancelTrainingResponse()

        success = self._engine.cancel_job(request.job_id)
        return CancelTrainingResponse(
            job_id=request.job_id,
            success=success,
            message="已取消" if success else "取消失败（任务不存在或已完成）",
        )

    def ListTrainingJobs(
        self,
        request: ListTrainingJobsRequest,
        context: grpc.ServicerContext,
    ) -> ListTrainingJobsResponse:
        """列出训练任务"""
        if not _verify_internal_token(context):
            return ListTrainingJobsResponse()

        # 转换状态过滤
        status_filter = None
        if request.status_filter:
            proto_to_status = {
                TrainingStatusProto.TRAINING_STATUS_QUEUED: JobStatus.QUEUED,
                TrainingStatusProto.TRAINING_STATUS_RUNNING: JobStatus.RUNNING,
                TrainingStatusProto.TRAINING_STATUS_COMPLETED: JobStatus.COMPLETED,
                TrainingStatusProto.TRAINING_STATUS_FAILED: JobStatus.FAILED,
                TrainingStatusProto.TRAINING_STATUS_CANCELLED: JobStatus.CANCELLED,
            }
            status_filter = [proto_to_status[s] for s in request.status_filter if s in proto_to_status]

        jobs = self._engine.list_jobs(
            status_filter=status_filter,
            limit=request.limit or 100,
        )

        summaries = []
        for job in jobs:
            summaries.append(TrainingJobSummary(
                job_id=job.job_id,
                status=cast(TrainingStatusProto, _job_status_to_proto(job.status)),
                base_model=job.base_model,
                character_name=job.character_name,
                progress_percent=job.progress.progress_percent if job.progress else 0.0,
                created_at=job.created_at.isoformat(),
                updated_at=job.updated_at.isoformat(),
            ))

        return ListTrainingJobsResponse(jobs=summaries)

    def ExecuteCustomScript(
        self,
        request: CustomScriptExecutionRequest,
        context: grpc.ServicerContext,
    ) -> Generator[CustomScriptExecutionProgressProto, None, None]:
        """执行自定义脚本（流式返回进度）"""
        if not _verify_internal_token(context):
            return

        job_id = request.job_id or str(uuid.uuid4())

        LOGGER.info("[script:%s] ExecuteCustomScript called: %s", job_id, request.script_path)

        job = self._script_runner.create_job(
            script_path=request.script_path,
            args=list(request.args) if request.args else [],
            env=dict(request.env) if request.env else {},
            working_dir=request.working_dir or None,
            timeout_seconds=int(request.timeout_seconds or 86400),
            job_id=job_id,
        )

        try:
            for progress in self._script_runner.run_script(job):
                proto = CustomScriptExecutionProgressProto(
                    job_id=progress.job_id,
                    status=progress.status.value if isinstance(progress.status, ScriptStatus) else str(progress.status),
                    stdout_lines=progress.stdout_lines,
                    stderr_lines=progress.stderr_lines,
                    return_code=progress.return_code or 0,
                    elapsed_seconds=progress.elapsed_seconds,
                    metrics={k: str(v) for k, v in (progress.metrics or {}).items()},
                    error=progress.error or "",
                )
                yield proto
        except Exception as exc:
            LOGGER.exception("[script:%s] ExecuteCustomScript error: %s", job_id, exc)
            yield CustomScriptExecutionProgressProto(
                job_id=job_id,
                status=ScriptStatus.FAILED.value,
                error=str(exc),
            )

    def CancelCustomScript(
        self,
        request: CancelCustomScriptRequest,
        context: grpc.ServicerContext,
    ) -> CancelCustomScriptResponse:
        """取消自定义脚本"""
        if not _verify_internal_token(context):
            return CancelCustomScriptResponse()

        success = self._script_runner.cancel_job(request.job_id)
        return CancelCustomScriptResponse(job_id=request.job_id, success=success, message="已取消" if success else "取消失败")
