# CY-LLM 部署指南

## 项目概述

CY-LLM 是一个支持 NVIDIA CUDA 和华为 Ascend NPU 的 AI 推理与训练服务，包含：

- **Gateway**：Kotlin/Spring Boot 网关，提供 REST API
- **Worker**：Python 推理/训练引擎，支持 gRPC 通信
- **Training**：LoRA 微调工具链

## 快速开始

### 1. 环境要求

- Docker 20.10+
- Docker Compose 2.0+
- NVIDIA GPU（需安装 nvidia-docker）或 华为 Ascend NPU

### 2. 配置

```bash
cd deploy
cp .env.example .env
vim .env  # 设置 API Key 等配置
```

### 3. 启动服务

```bash
# 构建镜像
./scripts/build.sh

# 启动服务
./scripts/deploy.sh start

# 查看状态
./scripts/deploy.sh status
```

## API 接口

### 推理接口

```bash
# 流式推理（SSE）
curl -H "X-API-Key: your-api-key" \
     -H "Content-Type: application/json" \
     -X POST http://localhost:8080/api/v1/inference/stream \
     -d '{
       "modelId": "furina",
       "prompt": "你好，芙宁娜！"
     }'

# 同步推理
     ## 集成测试与 CI

     在部署环境或开发环境中运行集成测试，需保证 Gateway、Coordinator、Worker 与 Redis 可用。

curl -H "X-API-Key: your-api-key" \
     -H "Content-Type: application/json" \
     -X POST http://localhost:8080/api/v1/inference \
     -d '{
       "modelId": "furina",
       "prompt": "你好，芙宁娜！"
     }'
```

### 训练接口

```bash
# 启动训练（SSE 流式返回进度）
curl -H "X-API-Key: your-api-key" \
     -H "Content-Type: application/json" \
     -X POST http://localhost:8080/api/v1/training/start \
     -d '{
       "baseModel": "deepseek-ai/deepseek-llm-7b-chat",
       "outputDir": "/checkpoints/furina_lora",
       "characterName": "芙宁娜",
       "datasetPath": "/data/furina_train.json"
     }'

# 查询训练状态
curl -H "X-API-Key: your-api-key" \
     http://localhost:8080/api/v1/training/{jobId}/status

# 取消训练
curl -H "X-API-Key: your-api-key" \
     -X POST http://localhost:8080/api/v1/training/{jobId}/cancel

# 列出训练任务
curl -H "X-API-Key: your-api-key" \
     http://localhost:8080/api/v1/training/jobs
```

## 架构说明

```
┌─────────────────────────────────────────────────────────────┐
│                        客户端                                │
│  (游戏客户端 / Web 界面 / 命令行工具)                          │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTPS + API Key
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Gateway (Kotlin)                          │

                    # 在全部服务可用后运行集成测试
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ REST API    │  │ 模型路由    │  │ 断路器/限流/重试     │  │
                    ```

                    ### 生产与监控
                    - 在生产中，请确保 `CY_LLM_INTERNAL_TOKEN` 配置、Prometheus 与 Grafana 正常接入。
                    - Telemetry 现在提供 P50/P95/P99 延迟统计与 token 吞吐量监控，Prometheus metrics 路径默认在 `:9090/metrics` (Worker) 与 `:8081/actuator/prometheus` (Coordinator)。
│  │ (WebFlux)   │  │ (Registry)  │  │ (Resilience4j)      │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└──────────────────────┬──────────────────────────────────────┘
                       │ gRPC + Internal Token
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Worker (Python)                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ 推理服务    │  │ 训练服务    │  │ 显存管理            │  │
│  │ (gRPC)      │  │ (gRPC)      │  │ (LRU 回收)          │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              引擎层                                      ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     ││
│  │  │ NVIDIA      │  │ Ascend      │  │ Hybrid      │     ││
│  │  │ Engine      │  │ Engine      │  │ Engine      │     ││
│  │  └─────────────┘  └─────────────┘  └─────────────┘     ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## 配置说明

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `CY_LLM_SECURITY_ENABLED` | 是否启用 API 认证 | `true` |
| `CY_LLM_API_KEY` | 外部 API Key | - |
| `CY_LLM_INTERNAL_TOKEN` | 内部通信 Token | - |
| `CY_LLM_WORKER_HOST` | Worker 地址 | `127.0.0.1` |
| `CY_LLM_WORKER_PORT` | Worker 端口 | `50051` |

### 模型配置

编辑 `deploy/models.json`：

```json
{
  "furina": {
    "model_path": "deepseek-ai/deepseek-llm-7b-chat",
    "adapter_path": "/checkpoints/furina_lora",
    "use_4bit": true
  }
}
```

## 扩展部署

### 单机多 Worker

```bash
# 扩展到 4 个 NVIDIA Worker
./scripts/deploy.sh scale 4
```

### 启用 Ascend Worker

```bash
./scripts/deploy.sh ascend
```

## 开发指南

### 生成 gRPC 代码

```bash
./scripts/gen_proto.sh
```

### 本地运行 Worker

```bash
cd worker
pip install -r requirements.txt
python -m worker.main --serve --port 50051
```

### 本地运行 Gateway

```bash
cd gateway
./gradlew bootRun
```

## 许可证

MIT License