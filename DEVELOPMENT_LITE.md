# Lite 开发与联调指南

本指南面向社区版（Community Lite），覆盖本地启动、环境变量、以及 Docker Compose 最小联调拓扑。

## 1. 本地启动（CLI）

```bash
# 初始化环境
./cy-llm setup --engine cuda-vllm

# 启动 Lite（Gateway + Coordinator + Worker）
./cy-llm lite --engine cuda-vllm --model facebook/opt-2.7b
```

## 2. Docker Compose（最小联调拓扑）

```bash
# 启动
docker compose -f docker-compose.community.yml up -d

# 查看状态
docker compose -f docker-compose.community.yml ps

# 停止
docker compose -f docker-compose.community.yml down
```

默认端口：
- Gateway Lite: 8000
- Coordinator Lite: 50051
- Worker: 50052

## 3. OpenAI 兼容请求

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"你好"}]}'
```

## 4. 关键环境变量

### Gateway Lite
- `COORDINATOR_GRPC_ADDR`：Coordinator gRPC 地址（默认 `127.0.0.1:50051`）
- `GATEWAY_API_TOKEN`：可选的静态鉴权 Token
- `GATEWAY_REQUEST_TIMEOUT`：请求超时（秒）

### Coordinator Lite
- `COORDINATOR_GRPC_BIND`：Coordinator 监听地址（默认 `0.0.0.0:50051`）
- `WORKER_GRPC_ADDRS`：Worker 列表（用逗号分隔）
- `COORDINATOR_CONFIG`：可选配置文件路径（JSON）

示例配置：
```json
{
  "workers": ["worker-1:50052", "worker-2:50052"]
}
```

### Worker
- `CY_LLM_ENGINE`：引擎类型（如 `cuda-vllm`）
- `CY_LLM_DEFAULT_MODEL`：默认模型 ID 或本地路径
- `CY_LLM_DEFAULT_ADAPTER`：LoRA 适配器路径（可选）
- `CY_LLM_MODEL_REGISTRY`：模型注册表 JSON（可选，字符串）
- `CY_LLM_MODEL_REGISTRY_PATH`：模型注册表路径（可选）
- `CY_LLM_HEALTH_PORT`：健康检查端口（默认 `9090`）

## 5. 常见问题

- **启动缓慢**：首次启动可能下载模型，等待时间较长。
- **端口冲突**：请确认 8000/50051/50052 未被占用。
- **GPU 不可用**：请确认 Docker 已开启 GPU 支持或本机可用 CUDA。
