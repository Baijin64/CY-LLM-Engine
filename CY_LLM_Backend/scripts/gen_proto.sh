#!/bin/bash
# gen_proto.sh - 生成 Python 和 Kotlin gRPC 代码
# 用法: ./scripts/gen_proto.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PROTO_DIR="$PROJECT_ROOT/proto"
WORKER_OUT="$PROJECT_ROOT/worker/proto_gen"
GATEWAY_OUT="$PROJECT_ROOT/gateway/src/main/kotlin"

echo "=== CY-LLM Proto Generator ==="
echo "Proto 目录: $PROTO_DIR"

# 创建输出目录
mkdir -p "$WORKER_OUT"

# 生成 Python gRPC 代码
echo "[1/2] 生成 Python gRPC 代码..."
if command -v python3 &> /dev/null; then
    python3 -m grpc_tools.protoc \
        -I"$PROTO_DIR" \
        --python_out="$WORKER_OUT" \
        --pyi_out="$WORKER_OUT" \
        --grpc_python_out="$WORKER_OUT" \
        "$PROTO_DIR/ai_service.proto"
    
    # 修复 Python 相对导入问题
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' 's/^import ai_service_pb2/from . import ai_service_pb2/' "$WORKER_OUT/ai_service_pb2_grpc.py"
    else
        sed -i 's/^import ai_service_pb2/from . import ai_service_pb2/' "$WORKER_OUT/ai_service_pb2_grpc.py"
    fi
    
    # 创建 __init__.py
    cat > "$WORKER_OUT/__init__.py" << 'EOF'
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
    CancelTrainingRequest,
    CancelTrainingResponse,
    GetTrainingStatusRequest,
    GetTrainingStatusResponse,
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
    "CancelTrainingRequest",
    "CancelTrainingResponse",
    "GetTrainingStatusRequest",
    "GetTrainingStatusResponse",
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
EOF
    echo "   Python 代码已生成到: $WORKER_OUT"
else
    echo "   警告: 未找到 python3，跳过 Python 代码生成"
fi

# 生成 Kotlin gRPC 代码（由 Gradle 插件处理）
echo "[2/2] Kotlin gRPC 代码由 Gradle 插件自动生成"
echo "   运行 'cd gateway && ./gradlew generateProto' 生成 Kotlin 代码"

echo ""
echo "=== 完成 ==="
