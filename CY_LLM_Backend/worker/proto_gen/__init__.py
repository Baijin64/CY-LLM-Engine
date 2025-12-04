"""proto_gen - gRPC 生成代码包"""
from .ai_service_pb2 import (
    # 推理服务消息
    StreamMetadata,
    GenerationParameters,
    StreamPredictRequest,
    StreamPredictResponse,
    ControlMessage,
    WorkerHealthRequest,
    WorkerHealthResponse,
    # 训练服务消息
    TrainingRequest,
    TrainingProgress,
    TrainingStatus,
    CancelTrainingRequest,
    CancelTrainingResponse,
    TrainingStatusRequest,
    TrainingStatusResponse,
    ListTrainingJobsRequest,
    ListTrainingJobsResponse,
    TrainingJobSummary,
    # 自定义脚本消息
    CustomScriptExecutionRequest,
    CustomScriptExecutionProgress,
    CancelCustomScriptRequest,
    CancelCustomScriptResponse,
)
from .ai_service_pb2_grpc import (
    # 推理服务
    AiInferenceServicer,
    AiInferenceStub,
    add_AiInferenceServicer_to_server,
    # 训练服务
    AiTrainingServicer,
    AiTrainingStub,
    add_AiTrainingServicer_to_server,
)

__all__ = [
    # 推理服务消息
    "StreamMetadata",
    "GenerationParameters",
    "StreamPredictRequest",
    "StreamPredictResponse",
    "ControlMessage",
    "WorkerHealthRequest",
    "WorkerHealthResponse",
    # 训练服务消息
    "TrainingRequest",
    "TrainingProgress",
    "TrainingStatus",
    "CancelTrainingRequest",
    "CancelTrainingResponse",
    "TrainingStatusRequest",
    "TrainingStatusResponse",
    "ListTrainingJobsRequest",
    "ListTrainingJobsResponse",
    "TrainingJobSummary",
    # 自定义脚本消息
    "CustomScriptExecutionRequest",
    "CustomScriptExecutionProgress",
    "CancelCustomScriptRequest",
    "CancelCustomScriptResponse",
    # 推理服务
    "AiInferenceServicer",
    "AiInferenceStub",
    "add_AiInferenceServicer_to_server",
    # 训练服务
    "AiTrainingServicer",
    "AiTrainingStub",
    "add_AiTrainingServicer_to_server",
]
