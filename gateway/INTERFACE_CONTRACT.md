# Gateway 接口契约（企业版对接规范）

## 概述

本文档定义了 Gateway 与后端服务之间的 **gRPC 接口契约**，确保：
- 开源版 Python Gateway 可以连接 Python Coordinator
- 企业版 Kotlin Backend 可以无缝替换，直接对接 Rust Sidecar

---

## gRPC 服务定义

### 1. 服务接口

```protobuf
service AiInference {
  // 流式推理（双向流）
  rpc StreamPredict(stream StreamPredictRequest) returns (stream StreamPredictResponse);
  
  // 控制命令（一元 RPC）
  rpc Control(ControlMessage) returns (ControlMessage);
  
  // 健康检查（一元 RPC）
  rpc Health(WorkerHealthRequest) returns (WorkerHealthResponse);
}
```

### 2. 消息定义

#### StreamPredictRequest

```protobuf
message StreamPredictRequest {
  string model_id = 1;           // 模型标识（如 "qwen-7b"）
  string prompt = 2;             // 用户输入
  string adapter = 3;            // LoRA Adapter 路径（可选）
  int32 priority = 4;            // 优先级（默认 0）
  
  GenerationParameters generation = 5;
  StreamMetadata metadata = 6;
}

message GenerationParameters {
  int32 max_new_tokens = 1;      // 最大生成长度
  float temperature = 2;          // 温度系数
  float top_p = 3;                // Top-p 采样
  float repetition_penalty = 4;   // 重复惩罚
}

message StreamMetadata {
  string trace_id = 1;            // 链路追踪 ID
}
```

#### StreamPredictResponse

```protobuf
message StreamPredictResponse {
  string trace_id = 1;            // 链路追踪 ID（对应请求）
  string chunk = 2;               // 流式返回的文本块
  bool end_of_stream = 3;         // 是否结束
  int32 index = 4;                // 块序号
}
```

---

## Gateway 调用流程

### 流程图

```
┌─────────────┐
│ HTTP Client │
└──────┬──────┘
       │ POST /v1/chat/completions
       ▼
┌─────────────────┐
│ Python Gateway  │
│ (FastAPI)       │
└──────┬──────────┘
       │ gRPC StreamPredict
       │ (UDS or TCP)
       ▼
┌─────────────────┐         ┌──────────────┐
│ Coordinator     │ ─UDS──▶ │ Rust Sidecar │
│ (Python/Kotlin) │         └──────────────┘
└─────────────────┘
```

### 请求转换示例

**HTTP 请求**（OpenAI 格式）：

```json
POST /v1/chat/completions
{
  "model": "qwen-7b",
  "messages": [
    {"role": "user", "content": "Hello"}
  ],
  "max_tokens": 256,
  "temperature": 0.7
}
```

**转换为 gRPC 请求**：

```python
StreamPredictRequest(
    model_id="qwen-7b",
    prompt="User: Hello\nAssistant:",
    generation=GenerationParameters(
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.0
    ),
    metadata=StreamMetadata(trace_id="uuid-12345")
)
```

---

## 企业版 Kotlin Backend 对接清单

### ✅ 必须实现的接口

1. **gRPC Server**
   - 监听地址：`:50050`（企业版标准端口）
   - 实现服务：`AiInference`
   - 方法：`StreamPredict`, `Control`, `Health`

2. **连接下游**
   - 目标：Rust Sidecar（UDS 或 gRPC）
   - 地址：`unix:///tmp/cy_sidecar.sock` 或 `localhost:50051`

3. **增强功能**（企业版特有）
   - OAuth2/JWT 认证
   - 多租户配额管理
   - 审计日志（写入 PostgreSQL）

### 🔄 替换步骤

#### 第 1 步：停用 Python Gateway

```bash
# 停止 Python Gateway 容器
docker stop cy-llm-gateway
```

#### 第 2 步：启动 Kotlin Backend

```bash
# 启动企业版后端
docker run -d \
  --name kotlin-backend \
  -p 50050:50050 \
  -v /tmp:/tmp \  # 挂载 UDS 目录
  -e DB_URL=postgresql://... \
  cy-llm-kotlin-backend:latest
```

#### 第 3 步：验证连接

```bash
# 测试 gRPC 健康检查
grpcurl -plaintext localhost:50050 AiInference/Health
```

#### 第 4 步：更新客户端配置

```bash
# 客户端直接连接 Kotlin Backend
curl http://kotlin-backend:8080/v1/chat/completions \
  -H "Authorization: Bearer $ENTERPRISE_TOKEN" \
  -d '{"model": "qwen-7b", "messages": [...]}'
```

---

## 环境变量对比

| 变量名 | 开源版 Gateway | 企业版 Kotlin Backend |
|--------|---------------|----------------------|
| **监听地址** | `0.0.0.0:8000` | `0.0.0.0:8080` |
| **后端连接** | `COORDINATOR_UDS_PATH=/tmp/cy_coordinator.sock` | `SIDECAR_GRPC_ADDR=unix:///tmp/cy_sidecar.sock` |
| **认证方式** | `GATEWAY_API_TOKEN=simple-key` | `OAUTH2_ISSUER=https://auth.example.com` |
| **数据库** | - | `DB_URL=postgresql://...` |

---

## Proto 文件位置

企业版团队需要复制以下文件用于 Kotlin 代码生成：

```
worker/proto_gen/ai_service.proto  # 主服务定义
```

**Kotlin 代码生成命令**：

```bash
protoc --kotlin_out=src/main/kotlin \
       --grpc-kotlin_out=src/main/kotlin \
       ai_service.proto
```

---

## 测试兼容性

### 工具

使用 `grpcurl` 测试两种后端是否兼容：

```bash
# 测试开源版 Coordinator
grpcurl -plaintext -unix /tmp/cy_coordinator.sock AiInference/Health

# 测试企业版 Kotlin Backend
grpcurl -plaintext localhost:50050 AiInference/Health
```

### 预期响应

```json
{
  "healthy": true,
  "metrics": {
    "backend": "python",  // 或 "kotlin"
    "version": "0.1.0"
  }
}
```

---

## 总结

- **开源版**：HTTP → Python Gateway → Coordinator → Sidecar
- **企业版**：HTTP → Kotlin Backend → Sidecar（跳过 Coordinator）
- **关键**：gRPC 接口契约保持一致，Rust Sidecar 无需改动
