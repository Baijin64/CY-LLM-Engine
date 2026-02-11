"""
grpc_servicer_async.py
[gRPC 异步服务实现] 基于 grpc.aio 的高性能异步服务器

优势：
  - 无线程池开销，使用 asyncio 事件循环
  - 更低的内存占用
  - 更好的并发性能
  - 原生支持异步推理引擎

使用：
    >>> server = await create_async_grpc_server(inference_server, config, port=50051)
    >>> await server.start()
    >>> await server.wait_for_termination()
"""

from __future__ import annotations

import asyncio
import logging
import uuid
import time
from typing import AsyncGenerator, Optional

import grpc
from grpc import aio

from .proto_gen import (
    AiInferenceServicer,
    ControlMessage,
    StreamPredictRequest,
    StreamPredictResponse,
    WorkerHealthRequest,
    WorkerHealthResponse,
    add_AiInferenceServicer_to_server,
)
from .config.config_loader import WorkerConfig, load_worker_config
from .core.server import InferenceServer
from .core.telemetry import Telemetry
from .constants import GRPCDefaults
from .utils.auth import verify_token, extract_token_from_metadata
from .utils.tls import GRPCTLSConfig, load_grpc_tls_config

try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None
    Status = None
    StatusCode = None

LOGGER = logging.getLogger("cy_llm.worker.grpc.async")


def _verify_internal_token(context: aio.ServicerContext) -> bool:
    """验证来自 Gateway 的内部 Token（使用共享认证模块）"""
    try:
        invocation_metadata = context.invocation_metadata()
        if invocation_metadata is None:
            metadata = {}
        else:
            metadata = dict(invocation_metadata)
    except Exception:
        metadata = {}
    
    provided_token = extract_token_from_metadata(metadata)
    is_valid, _ = verify_token(provided_token)
    
    if not is_valid:
        LOGGER.warning("内部认证失败")
    
    return is_valid


class AsyncAiInferenceServicer(AiInferenceServicer):
    """异步 AiInference gRPC 服务实现"""

    def __init__(
        self,
        inference_server: InferenceServer,
        config: WorkerConfig,
        telemetry: Optional[Telemetry] = None,
        tracer=None,
    ) -> None:
        self._server = inference_server
        self._config = config
        self._telemetry = telemetry or Telemetry()
        self._executor = None  # 用于同步代码的线程池
        self._tracer = tracer

    async def StreamPredict(
        self,
        request_iterator: AsyncGenerator[StreamPredictRequest, None],
        context: aio.ServicerContext,
    ) -> AsyncGenerator[StreamPredictResponse, None]:
        """异步双向流式推理接口"""
        try:
            first_request = await request_iterator.__anext__()
        except StopAsyncIteration:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Empty request stream")
            return

        trace_id = (
            first_request.metadata.trace_id 
            if (first_request.metadata and first_request.metadata.trace_id) 
            else str(uuid.uuid4())
        )
        model_id = first_request.model_id or "default"
        prompt = first_request.prompt
        adapter = first_request.adapter or None
        priority = first_request.priority or 0

        # 解析生成参数
        gen_params = first_request.generation
        generation_kwargs = {}
        if gen_params:
            if gen_params.max_new_tokens > 0:
                generation_kwargs["max_new_tokens"] = gen_params.max_new_tokens
            if gen_params.temperature > 0:
                generation_kwargs["temperature"] = gen_params.temperature
            if gen_params.top_p > 0:
                generation_kwargs["top_p"] = gen_params.top_p
            if gen_params.repetition_penalty > 0:
                generation_kwargs["repetition_penalty"] = gen_params.repetition_penalty

        # OpenTelemetry span
        span = None
        if OTEL_AVAILABLE and hasattr(self, '_tracer') and self._tracer:
            span = self._tracer.start_as_current_span("StreamPredict")
            span.set_attribute("model_id", model_id)
            span.set_attribute("priority", priority)

        LOGGER.info(
            "[%s] AsyncStreamPredict 请求: model=%s prompt_len=%d priority=%d",
            trace_id, model_id, len(prompt), priority,
        )

        # 获取模型配置
        spec = self._config.model_registry.get(model_id)
        if spec is None:
            LOGGER.error("[%s] 模型 %s 未在注册表中找到", trace_id, model_id)
            await context.abort(
                grpc.StatusCode.NOT_FOUND,
                f"Model '{model_id}' not found in registry",
            )
            if span:
                span.set_status(Status(StatusCode.ERROR, "NOT_FOUND"))
                span.end()
            return

        model_path = spec.model_path
        adapter_path = adapter or spec.adapter_path
        engine_type = spec.engine or self._config.preferred_backend
        engine_kwargs = self._build_engine_kwargs(spec)
        enable_prompt_cache = spec.enable_prompt_cache or False
        prompt_cache_ttl = spec.prompt_cache_ttl

        start_time = time.perf_counter()
        first_token_time = None

        try:
            chunk_index = 0

            # 检查是否有异步推理方法
            if hasattr(self._server, 'async_stream_predict'):
                # 原生异步推理
                async for chunk in self._server.async_stream_predict(
                    model_id=model_id,
                    prompt=prompt,
                    model_path=model_path,
                    adapter_path=adapter_path,
                    engine_type=engine_type,
                    generation_kwargs=generation_kwargs or None,
                    engine_kwargs=engine_kwargs or None,
                    priority=priority,
                    enable_prompt_cache=enable_prompt_cache,
                    prompt_cache_ttl=prompt_cache_ttl,
                ):
                    if chunk_index == 0:
                        first_token_time = time.perf_counter()

                    yield StreamPredictResponse(
                        trace_id=trace_id,
                        chunk=str(chunk),
                        end_of_stream=False,
                        index=chunk_index,
                    )
                    chunk_index += 1
            else:
                # 兼容同步推理（在线程池中执行）
                sync_gen = self._server.stream_predict(
                    model_id=model_id,
                    prompt=prompt,
                    model_path=model_path,
                    adapter_path=adapter_path,
                    engine_type=engine_type,
                    generation_kwargs=generation_kwargs or None,
                    engine_kwargs=engine_kwargs or None,
                    priority=priority,
                    enable_prompt_cache=enable_prompt_cache,
                    prompt_cache_ttl=prompt_cache_ttl,
                )

                # 将同步生成器转换为异步
                for chunk in sync_gen:
                    if chunk_index == 0:
                        first_token_time = time.perf_counter()

                    yield StreamPredictResponse(
                        trace_id=trace_id,
                        chunk=str(chunk),
                        end_of_stream=False,
                        index=chunk_index,
                    )
                    chunk_index += 1
                    # 让出控制权，避免阻塞事件循环
                    await asyncio.sleep(0)

            # 统计计算
            end_time = time.perf_counter()
            ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0
            
            # 计算生成速度 (TPS)
            gen_duration = end_time - first_token_time if first_token_time else (end_time - start_time)
            gen_duration = max(gen_duration, 0.001)
            
            tokens_count = chunk_index
            engine_type_str = str(engine_type).lower()
            if "vllm" in engine_type_str and "async" not in engine_type_str:
                tokens_count = tokens_count / 3.0
            
            tps = tokens_count / gen_duration

            # 发送结束标记
            yield StreamPredictResponse(
                trace_id=trace_id,
                chunk="",
                end_of_stream=True,
                index=chunk_index,
                ttft_ms=ttft_ms,
                tokens_per_sec=tps,
            )
            
            LOGGER.info(
                "[%s] AsyncStreamPredict 完成: chunks=%d, speed=%.2f tok/s, ttft=%.2fms", 
                trace_id, chunk_index, tps, ttft_ms
            )
            
            if span:
                span.set_status(Status(StatusCode.OK))
                span.end()

        except Exception as exc:
            LOGGER.exception("[%s] AsyncStreamPredict 异常: %s", trace_id, exc)
            if span:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                span.end()
            await context.abort(
                grpc.StatusCode.INTERNAL,
                f"Inference error: {exc}",
            )

    def _build_engine_kwargs(self, spec) -> dict:
        """构建引擎参数"""
        engine_kwargs = {}
        if spec.use_4bit is not None:
            engine_kwargs["use_4bit"] = spec.use_4bit
        if spec.max_model_len is not None:
            engine_kwargs["max_model_len"] = spec.max_model_len
        if spec.tensor_parallel_size is not None:
            engine_kwargs["tensor_parallel_size"] = spec.tensor_parallel_size
        if spec.enable_prefix_caching is not None:
            engine_kwargs["enable_prefix_caching"] = spec.enable_prefix_caching
        if spec.kv_cache_dtype is not None:
            engine_kwargs["kv_cache_dtype"] = spec.kv_cache_dtype
        if spec.gpu_memory_utilization is not None:
            engine_kwargs["gpu_memory_utilization"] = spec.gpu_memory_utilization
        return engine_kwargs

    async def Control(
        self,
        request: ControlMessage,
        context: aio.ServicerContext,
    ) -> ControlMessage:
        """异步控制指令接口"""
        if not _verify_internal_token(context):
            await context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid internal token")
            return ControlMessage()

        trace_id = request.trace_id or str(uuid.uuid4())
        command = request.command
        payload = dict(request.payload)

        LOGGER.info("[%s] AsyncControl 指令: cmd=%s", trace_id, command)

        response_payload = {}

        if command == "unload_model":
            model_id = payload.get("model_id", "")
            if model_id:
                # 异步卸载（如果支持）
                if hasattr(self._server, 'async_unload_model'):
                    await self._server.async_unload_model(model_id)
                else:
                    self._server.unload_model(model_id)
                response_payload["status"] = "ok"
                response_payload["message"] = f"Model {model_id} unloaded"
            else:
                response_payload["status"] = "error"
                response_payload["message"] = "Missing model_id"

        elif command == "list_models":
            models = list(self._config.model_registry.keys())
            response_payload["status"] = "ok"
            response_payload["models"] = ",".join(models)

        elif command == "reload_registry":
            # 热更新模型注册表
            try:
                self._config = load_worker_config()
                response_payload["status"] = "ok"
                response_payload["message"] = "Registry reloaded"
                response_payload["models"] = ",".join(self._config.model_registry.keys())
            except Exception as e:
                response_payload["status"] = "error"
                response_payload["message"] = str(e)

        elif command == "ping":
            response_payload["status"] = "ok"
            response_payload["message"] = "pong"

        else:
            response_payload["status"] = "error"
            response_payload["message"] = f"Unknown command: {command}"

        return ControlMessage(
            trace_id=trace_id,
            command=f"{command}_response",
            payload=response_payload,
        )

    async def Health(
        self,
        request: WorkerHealthRequest,
        context: aio.ServicerContext,
    ) -> WorkerHealthResponse:
        """异步健康检查接口"""
        trace_id = request.trace_id or str(uuid.uuid4())

        metrics = {}
        try:
            snapshot = self._telemetry.snapshot()
            metrics["requests_inflight"] = str(snapshot.get("requests_inflight", 0))
            metrics["requests_success"] = str(snapshot.get("requests_success", 0))
            metrics["requests_failed"] = str(snapshot.get("requests_failed", 0))
            metrics["backend"] = self._config.preferred_backend
            metrics["cuda_available"] = str(self._config.hardware.has_cuda)
            metrics["ascend_available"] = str(self._config.hardware.has_ascend)
            metrics["async_mode"] = "true"
        except Exception as exc:
            LOGGER.warning("[%s] 获取指标失败: %s", trace_id, exc)

        return WorkerHealthResponse(
            healthy=True,
            metrics=metrics,
        )


async def create_async_grpc_server(
    inference_server: InferenceServer,
    config: WorkerConfig,
    port: int = GRPCDefaults.PORT,
    telemetry: Optional[Telemetry] = None,
    enable_training: bool = True,
) -> aio.Server:
    """创建异步 gRPC 服务器
    
    Args:
        inference_server: 推理服务器实例
        config: Worker 配置
        port: gRPC 监听端口
        telemetry: 遥测实例
        enable_training: 是否启用训练服务
        
    Returns:
        grpc.aio.Server 实例
    """
    server = aio.server(
        options=[
            ("grpc.max_send_message_length", GRPCDefaults.MAX_MESSAGE_SIZE_BYTES),
            ("grpc.max_receive_message_length", GRPCDefaults.MAX_MESSAGE_SIZE_BYTES),
            ("grpc.keepalive_time_ms", GRPCDefaults.KEEPALIVE_TIME_MS),
            ("grpc.keepalive_timeout_ms", GRPCDefaults.KEEPALIVE_TIMEOUT_MS),
            ("grpc.keepalive_permit_without_calls", True),
            ("grpc.http2.max_pings_without_data", 0),
            ("grpc.http2.min_ping_interval_without_data_ms", GRPCDefaults.MIN_PING_INTERVAL_MS),
        ],
    )

    # 注册异步推理服务
    inference_servicer = AsyncAiInferenceServicer(
        inference_server=inference_server,
        config=config,
        tracer=tracer,
    )
    add_AiInferenceServicer_to_server(inference_servicer, server)
    LOGGER.info("已注册异步推理服务 (AiInference)")

    # 注册训练服务（可选）
    if enable_training:
        try:
            from .training_servicer_grpc import AiTrainingServicerImpl
            from .proto_gen import add_AiTrainingServicer_to_server
            
            training_servicer = AiTrainingServicerImpl(engine=engine)
            add_AiTrainingServicer_to_server(training_servicer, server)
            LOGGER.info("已注册训练服务 (AiTraining)")
        except ImportError as e:
            LOGGER.warning("训练服务未启用（依赖缺失）: %s", e)

    tls_config = load_grpc_tls_config()
    if tls_config:
        server.add_secure_port(f"[::]:{port}", tls_config.credentials)
        LOGGER.info("异步 gRPC TLS 已启用，端口: %d", port)
    else:
        server.add_insecure_port(f"[::]:{port}")
        LOGGER.warning("异步 gRPC 未启用 TLS，端口: %d", port)

    return server


async def run_async_server(
    inference_server: InferenceServer,
    config: WorkerConfig,
    port: int = GRPCDefaults.PORT,
    telemetry: Optional[Telemetry] = None,
):
    """运行异步 gRPC 服务器（阻塞直到终止）"""
    server = await create_async_grpc_server(
        inference_server=inference_server,
        config=config,
        port=args.port,
        max_workers=args.max_workers,
        telemetry=telemetry,
        enable_training=enable_training,
        tracer=tracer,
    )
    
    await server.start()
    LOGGER.info("异步 gRPC 服务器已启动")
    
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        LOGGER.info("收到终止信号，正在关闭...")
        await server.stop(grace=5)
