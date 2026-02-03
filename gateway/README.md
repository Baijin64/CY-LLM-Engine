# Gateway 模块说明

## 架构定位

Gateway 是 **可替换的接入层**，符合 Open Core 架构：

- **开源版**：使用此 Python FastAPI Gateway
- **企业版**：替换为 Kotlin Backend（通过同样的 gRPC 接口）

## 技术栈

- **框架**：FastAPI + Uvicorn
- **协议**：
  - 对外：HTTP REST API (`/v1/chat/completions`)
  - 对内：gRPC (连接 Coordinator 或直连 Rust Sidecar)

## 接口契约（兼容性保证）

### 后端连接（gRPC）

```python
# 环境变量配置
COORDINATOR_UDS_PATH=/tmp/cy_coordinator.sock  # 开源版默认
# 或
COORDINATOR_GRPC_ADDR=kotlin-backend:50050     # 企业版 Kotlin Backend
```

### gRPC Service Definition

Gateway 依赖的 Protobuf 服务定义位于 `worker/proto_gen/ai_service.proto`：

```protobuf
service AiInference {
  rpc StreamPredict(stream StreamPredictRequest) returns (stream StreamPredictResponse);
  rpc Control(ControlMessage) returns (ControlMessage);
  rpc Health(WorkerHealthRequest) returns (WorkerHealthResponse);
}
```

**关键点**：
- 企业版 Kotlin Backend 必须实现**完全相同的 gRPC 接口**
- 这样可以无缝替换 Gateway，无需改动 Worker 层

## 部署模式

### 模式 1：开源版（标准链路）

```
HTTP Client → Python Gateway → Coordinator → Rust Sidecar → Worker
              (FastAPI)         (UDS)         (UDS)
```

### 模式 2：企业版（替换 Gateway）

```
HTTP Client → Kotlin Backend → Rust Sidecar → Worker
              (Spring WebFlux)  (gRPC :50050)
```

## 启动方式

```bash
# 开发环境
cd gateway/gateway_lite
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 生产环境
docker run -p 8000:8000 \
  -e COORDINATOR_UDS_PATH=/tmp/cy_coordinator.sock \
  cy-llm-gateway:latest
```

## 环境变量

| 变量名 | 说明 | 默认值 | 企业版用途 |
|--------|------|--------|------------|
| `COORDINATOR_UDS_PATH` | Coordinator UDS 路径 | `/tmp/cy_coordinator.sock` | - |
| `COORDINATOR_GRPC_ADDR` | 企业版后端地址 | - | `kotlin-backend:50050` |
| `GATEWAY_API_TOKEN` | API Key 认证 | 空（无认证） | 由 Kotlin Backend 接管 |

## 升级路径

1. **保持开源 Gateway**：
   - 适合个人开发者、小团队
   - 快速上手，无需 JVM 环境

2. **升级到企业版 Gateway**：
   - 替换 `gateway/` 目录为 Kotlin 项目
   - 修改 `COORDINATOR_GRPC_ADDR` 指向 Kotlin Backend
   - Worker 层**无需任何改动**

---

**设计哲学**：接口标准化 > 实现语言
