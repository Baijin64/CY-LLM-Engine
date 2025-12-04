# Coordinator + Worker 架构说明

## 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                       Gateway (Kotlin)                          │
│  Spring WebFlux + gRPC Client                                   │
│  ├─ InferenceService (路由 + Resilience4j)                      │
│  ├─ TrainingService                                              │
│  └─ ModelRegistry                                                │
└───────────────────────────┬─────────────────────────────────────┘
                            │ gRPC (:50050)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Coordinator (Kotlin)                          │
│  Spring Boot 3 + Redis + gRPC + Coroutines                      │
│  ├─ TaskQueueService       # Redis-backed 优先级队列 (ZSET)     │
│  ├─ PromptCacheService     # Redis-backed Prompt 缓存 (TTL)     │
│  ├─ WorkerPoolManager      # Worker 健康检查 + 负载均衡          │
│  ├─ WorkerGrpcClient       # 异步 gRPC 客户端                    │
│  └─ TelemetryService       # Micrometer 指标聚合                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │ gRPC (:50051)
         ┌──────────────────┴──────────────────┐
         ▼                                     ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│    Worker (Python)      │     │    Worker (Python)      │
│    NVIDIA GPU           │     │    Ascend NPU           │
│  ├─ InferenceEngine     │     │  ├─ InferenceEngine     │
│  │   └─ vLLM/TensorRT   │     │  │   └─ MindIE/vLLM     │
│  └─ TrainingEngine      │     │  └─ TrainingEngine      │
│      └─ LoRA/PEFT       │     │      └─ LoRA/PEFT       │
└─────────────────────────┘     └─────────────────────────┘
             │                               │
             └───────────┬───────────────────┘
                         ▼
                   ┌───────────┐
                   │   Redis   │
                   │  :6379    │
                   └───────────┘
```

## 组件职责

### Gateway (Kotlin) - 端口 8080
- HTTP API 入口 (REST/WebSocket)
- 身份验证/授权 (API Key / JWT)
- 请求路由和协议转换
- 错误处理和重试策略

### Coordinator (Kotlin) - 端口 50050
- **TaskQueueService**: Redis ZSET 优先级队列，支持任务排队和背压控制
- **PromptCacheService**: Redis 缓存，SHA256 哈希 Key，减少重复推理
- **WorkerPoolManager**: Worker 健康检查、负载均衡（加权轮询）、熔断
- **WorkerGrpcClient**: 异步 gRPC 客户端，支持流式响应
- **TelemetryService**: Micrometer + Prometheus 指标聚合

### Worker (Python) - 端口 50051
- **InferenceEngine**: GPU 推理（vLLM/TensorRT/MindIE）
- **TrainingEngine**: GPU 训练（LoRA/PEFT + Modular Architecture）
  - `data/`: DataLoader, Formatter (8种格式)
  - `model/`: ModelSetup, LoRA Config (11种架构)
  - `loop/`: Trainer, Callbacks
- **MemoryManager**: 显存管理和监控
 - **MemoryManager**: 显存管理与 LRU 驱逐、锁清理和并发访问控制
 - **Telemetry**: Prometheus 指标输出、P50/P95/P99 百分位延迟统计以及 token 吞吐监控

## 快速启动

### Docker Compose (推荐)
```bash
# 启动所有服务
cd deploy
docker compose up -d

# 或使用 cy-llm 命令行工具（推荐使用 ./cy）
./cy-llm docker up

# 查看日志
docker compose logs -f coordinator
```

### 手动启动 (开发环境)

#### 1. 启动 Redis
```bash
docker run -d --name redis -p 6379:6379 redis:7-alpine
```

#### 2. 启动 Coordinator
```bash
cd coordinator
./gradlew bootRun
# 或指定 Java 21
JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 ./gradlew bootRun
```

#### 3. 启动 Worker
```bash
cd worker
python main.py
```

#### 4. 启动 Gateway
```bash
cd gateway
./gradlew bootRun
```

## 配置

### Coordinator (application.yml)
```yaml
coordinator:
  worker-pool:
    health-check-interval: 10s
    health-check-timeout: 5s
    max-retries: 3
  task-queue:
    max-size: 1000
    default-priority: 0
  prompt-cache:
    enabled: true
    ttl: 1800s
    max-size: 10000

workers:
  - id: worker-1
    host: localhost
    port: 50051
    weight: 100
    tags:
      - cuda
      - inference
      - training

spring:
  data:
    redis:
      host: localhost
      port: 6379
```

### Worker (环境变量)
```bash
PREFERRED_BACKEND=cuda-vllm
GRPC_PORT=50051
MODEL_REGISTRY_PATH=/path/to/models.json
```

### Gateway (环境变量)
```bash
CY_LLM_COORDINATOR_HOST=localhost
CY_LLM_COORDINATOR_PORT=50050
```

## 数据流

### 推理请求
```
1. Gateway → Coordinator.StreamPredict()
2. Coordinator 检查 PromptCache
3. 如果缓存命中，直接返回
4. 如果缓存未命中：
   a. WorkerPoolManager 选择最优 Worker
   b. 转发请求到 Worker
   c. 流式返回结果
   d. 缓存完整响应
```

### 训练任务
```
1. Gateway → Coordinator.StartTraining()
2. Coordinator 将任务加入 TaskQueue
3. WorkerPoolManager 选择可用 Worker
4. 转发训练请求到 Worker
5. 流式返回进度更新
6. 任务完成后更新状态
```

## 监控

### Prometheus 指标

Coordinator 指标 (`:8081/actuator/prometheus`):
- `coordinator_inference_requests_total` - 推理请求总数
- `coordinator_cache_hit_rate` - 缓存命中率
- `coordinator_queue_pending` - 待处理任务数
- `coordinator_workers_healthy` - 健康 Worker 数

Worker 指标 (`:9090/metrics`):
- `worker_inference_latency` - 推理延迟
- `worker_gpu_memory_used` - GPU 显存使用
- `worker_gpu_utilization` - GPU 利用率