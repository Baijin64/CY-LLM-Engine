"""
grpc_servicer.py
[gRPC 服务实现] 实现 AiInferenceServicer，桥接 gRPC 接口与 InferenceServer
"""

from __future__ import annotations

import logging
import time
import uuid
from concurrent import futures
from typing import Generator, Iterator, Optional

import grpc

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
from .exceptions import (
    EWWorkerError,
    EngineError,
    ModelError,
    ResourceError,
    GPUMemoryError,
)
from .utils.auth import verify_grpc_context

LOGGER = logging.getLogger("cy_llm.worker.grpc")


def _verify_internal_token(context: grpc.ServicerContext) -> bool:
    """验证来自 Gateway 的内部 Token"""
    return verify_grpc_context(context)


class AiInferenceServicerImpl(AiInferenceServicer):
    """AiInference gRPC 服务实现"""

    def __init__(
        self,
        inference_server: InferenceServer,
        config: Optional[WorkerConfig] = None,
        telemetry: Optional[Telemetry] = None,
    ) -> None:
        self._server = inference_server
        self._config = config or load_worker_config()
        self._telemetry = telemetry or Telemetry()

    # ====== 兼容旧测试 API ======
    
    def LoadModel(self, request, context):
        """加载模型 (兼容旧测试 API)"""
        model_id = getattr(request, 'model_id', '')
        model_path = getattr(request, 'model_path', '')
        self._server.load_model(model_id, model_path)
        from .proto_gen import ControlMessage
        return ControlMessage(command="load_response", payload={"status": "ok"})
    
    def UnloadModel(self, request, context):
        """卸载模型 (兼容旧测试 API)"""
        model_id = getattr(request, 'model_id', '')
        self._server.unload_model(model_id)
        from .proto_gen import ControlMessage
        return ControlMessage(command="unload_response", payload={"status": "ok"})
    
    def GetStatus(self, request, context):
        """获取状态 (兼容旧测试 API)"""
        from .proto_gen import WorkerHealthResponse
        return WorkerHealthResponse(healthy=True, metrics={})

    def StreamPredict(
        self,
        request_iterator: Iterator[StreamPredictRequest],
        context: grpc.ServicerContext,
    ) -> Generator[StreamPredictResponse, None, None]:
        """双向流推理接口"""
        if not _verify_internal_token(context):
            return

        # 从第一个请求获取推理参数
        try:
            first_request = next(request_iterator)
        except StopIteration:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Empty request stream")
            return

        trace_id = first_request.metadata.trace_id if first_request.metadata else str(uuid.uuid4())
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

        LOGGER.info(
            "[%s] StreamPredict 请求: model=%s prompt_len=%d priority=%d",
            trace_id, model_id, len(prompt), priority,
        )

        # 从模型注册表获取模型配置
        spec = self._config.model_registry.get(model_id)
        if spec is None:
            LOGGER.error("[%s] 模型 %s 未在注册表中找到", trace_id, model_id)
            context.abort(
                grpc.StatusCode.NOT_FOUND,
                f"Model '{model_id}' not found in registry",
            )
            return

        model_path = spec.model_path
        adapter_path = adapter or spec.adapter_path
        engine_type = spec.engine or self._config.preferred_backend
        engine_kwargs = {}
        if spec.use_4bit is not None:
            # 兼容旧的 use_4bit 字段：不要直接将其传入引擎构造函数
            # 将布尔值映射为 quantization 参数（若为 True 则使用 AWQ），
            # 否则不设置量化，让引擎使用默认设置。
            use_4bit_val = spec.use_4bit
            if use_4bit_val:
                engine_kwargs["quantization"] = "awq"
        # KV Cache 配置
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

        # Prompt 缓存配置
        enable_prompt_cache = spec.enable_prompt_cache or False
        prompt_cache_ttl = spec.prompt_cache_ttl

        try:
            chunk_index = 0
            for chunk in self._server.stream_predict(
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
                yield StreamPredictResponse(
                    trace_id=trace_id,
                    chunk=chunk,
                    end_of_stream=False,
                    index=chunk_index,
                )
                chunk_index += 1

            # 发送结束标记
            yield StreamPredictResponse(
                trace_id=trace_id,
                chunk="",
                end_of_stream=True,
                index=chunk_index,
            )
            LOGGER.info("[%s] StreamPredict 完成: chunks=%d", trace_id, chunk_index)

        except GPUMemoryError as exc:
            LOGGER.error("[%s] GPU 显存不足: %s", trace_id, exc)
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(exc))
        except ResourceError as exc:
            LOGGER.error("[%s] 资源错误: %s", trace_id, exc)
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(exc))
        except ModelError as exc:
            LOGGER.error("[%s] 模型错误: %s", trace_id, exc)
            context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        except EngineError as exc:
            LOGGER.error("[%s] 引擎错误: %s", trace_id, exc)
            context.abort(grpc.StatusCode.INTERNAL, str(exc))
        except EWWorkerError as exc:
            LOGGER.error("[%s] Worker 错误: %s", trace_id, exc)
            context.abort(grpc.StatusCode.INTERNAL, str(exc))
        except RuntimeError as exc:
            LOGGER.error("[%s] 运行时错误: %s", trace_id, exc)
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(exc))
        except Exception as exc:
            LOGGER.exception("[%s] 未预期异常: %s", trace_id, exc)
            context.abort(grpc.StatusCode.INTERNAL, f"Internal error: {type(exc).__name__}")

    def Control(
        self,
        request: ControlMessage,
        context: grpc.ServicerContext,
    ) -> ControlMessage:
        """控制指令接口"""
        if not _verify_internal_token(context):
            return ControlMessage()

        trace_id = request.trace_id or str(uuid.uuid4())
        command = request.command
        payload = dict(request.payload)

        LOGGER.info("[%s] Control 指令: cmd=%s", trace_id, command)

        response_payload = {}

        if command == "unload_model":
            model_id = payload.get("model_id", "")
            if model_id:
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

    def Health(
        self,
        request: WorkerHealthRequest,
        context: grpc.ServicerContext,
    ) -> WorkerHealthResponse:
        """健康检查接口"""
        trace_id = request.trace_id or str(uuid.uuid4())

        # 收集指标
        metrics = {}
        try:
            snapshot = self._telemetry.snapshot()
            metrics["requests_inflight"] = str(snapshot.get("requests_inflight", 0))
            metrics["requests_success"] = str(snapshot.get("requests_success", 0))
            metrics["requests_failed"] = str(snapshot.get("requests_failed", 0))
            metrics["backend"] = self._config.preferred_backend
            metrics["cuda_available"] = str(self._config.hardware.has_cuda)
            metrics["ascend_available"] = str(self._config.hardware.has_ascend)
        except Exception as exc:
            LOGGER.warning("[%s] 获取指标失败: %s", trace_id, exc)

        return WorkerHealthResponse(
            healthy=True,
            metrics=metrics,
        )


def create_grpc_server(
    inference_server: InferenceServer,
    config: WorkerConfig,
    port: int = GRPCDefaults.PORT,
    max_workers: int = GRPCDefaults.DEFAULT_WORKERS,
    telemetry: Optional[Telemetry] = None,
    enable_training: bool = True,
) -> grpc.Server:
    """创建并配置 gRPC 服务器
    
    Args:
        inference_server: 推理服务器实例
        config: Worker 配置
        port: gRPC 监听端口
        max_workers: 线程池大小
        telemetry: 遥测实例
        enable_training: 是否启用训练服务
    """
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ("grpc.max_send_message_length", GRPCDefaults.MAX_MESSAGE_SIZE_BYTES),
            ("grpc.max_receive_message_length", GRPCDefaults.MAX_MESSAGE_SIZE_BYTES),
            ("grpc.keepalive_time_ms", GRPCDefaults.KEEPALIVE_TIME_MS),
            ("grpc.keepalive_timeout_ms", GRPCDefaults.KEEPALIVE_TIMEOUT_MS),
            ("grpc.keepalive_permit_without_calls", True),
        ],
    )

    # 注册推理服务
    inference_servicer = AiInferenceServicerImpl(
        inference_server=inference_server,
        config=config,
        telemetry=telemetry,
    )
    add_AiInferenceServicer_to_server(inference_servicer, server)
    LOGGER.info("已注册推理服务 (AiInference)")

    # 注册训练服务（可选）
    if enable_training:
        try:
            from .training_servicer_grpc import AiTrainingServicerImpl
            from .proto_gen import add_AiTrainingServicer_to_server
            
            training_servicer = AiTrainingServicerImpl()
            add_AiTrainingServicer_to_server(training_servicer, server)
            LOGGER.info("已注册训练服务 (AiTraining)")
        except ImportError as e:
            LOGGER.warning("训练服务未启用（依赖缺失）: %s", e)

    server.add_insecure_port(f"[::]:{port}")
    LOGGER.info("gRPC 服务器配置完成，端口: %d", port)

    return server


# ====== 兼容旧测试 API ======
# 测试文件期望导入 AIServiceServicer 名称
AIServiceServicer = AiInferenceServicerImpl
