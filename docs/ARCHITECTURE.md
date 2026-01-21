# 架构设计文档

本文档详细描述 CY-LLM Engine 的系统架构、组件设计、数据流和关键实现细节。

## 目录

- [系统概览](#系统概览)
- [核心组件](#核心组件)
- [数据流](#数据流)
- [引擎架构](#引擎架构)
- [训练系统](#训练系统)
- [配置与扩展](#配置与扩展)

## 系统概览

### 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              客户端层                                         │
│                    (Web App / Mobile App / API Client)                       │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │ HTTPS / SSE / WebSocket
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Gateway (Kotlin)                                  │
│                        Spring WebFlux + gRPC Client                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        API 网关功能                                   │    │
│  │  • 请求路由与负载均衡      • 认证与授权 (API Key / JWT)               │    │
│  │  • 协议转换 (REST ↔ gRPC)  • 限流与熔断 (Resilience4j)               │    │
│  │  • 请求/响应日志           • 指标采集 (Micrometer)                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              端口: 8080                                      │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │ gRPC (:50050)
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Coordinator (Kotlin)                                │
│                      Spring Boot + Redis + gRPC Server                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        任务调度核心                                   │    │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐│    │
│  │  │TaskQueueSvc  │ │PromptCache   │ │WorkerPoolMgr │ │TelemetrySvc  ││    │
│  │  │(Redis ZSET)  │ │(Redis TTL)   │ │(LB + Health) │ │(Prometheus)  ││    │
│  │  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘│    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              端口: 50050                                     │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │ gRPC (:50051)
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
┌─────────────────────────────┐           ┌─────────────────────────────┐
│      Worker (Python)        │           │      Worker (Python)        │
│      NVIDIA GPU             │           │      Ascend NPU             │
│  ┌───────────────────────┐  │           │  ┌───────────────────────┐  │
│  │   InferenceEngine     │  │           │  │   InferenceEngine     │  │
│  │  ┌─────────────────┐  │  │           │  │  ┌─────────────────┐  │  │
│  │  │  AbstractEngine │  │  │           │  │  │  AbstractEngine │  │  │
│  │  └─────────────────┘  │  │           │  │  └─────────────────┘  │  │
│  │  ┌─────────────────┐  │  │           │  │  ┌─────────────────┐  │  │
│  │  │ • VllmCuda      │  │  │           │  │  │ • VllmAscend    │  │  │
│  │  │ • VllmAsync     │  │  │           │  │  │ • MindIE        │  │  │
│  │  │ • TRTEngine     │  │  │           │  │  │ • HybridEngine  │  │  │
│  │  └─────────────────┘  │  │           │  │  └─────────────────┘  │  │
│  └───────────────────────┘  │           │  └───────────────────────┘  │
│  ┌───────────────────────┐  │           │  ┌───────────────────────┐  │
│  │   TrainingEngine      │  │           │  │   TrainingEngine      │  │
│  │  ┌─────────────────┐  │  │           │  │  ┌─────────────────┐  │  │
│  │  │ • LoRA/PEFT     │  │  │           │  │  │ • LoRA/PEFT     │  │  │
│  │  │ • Full Fine-tune│  │  │           │  │  │ • Custom Script │  │  │
│  │  │ • Custom Script │  │  │           │  │  └─────────────────┘  │  │
│  │  └─────────────────┘  │  │           │  └───────────────────────┘  │
│  └───────────────────────┘  │           │                             │
└─────────────────────────────┘           └─────────────────────────────┘
          │                                           │
          └─────────────────────┬─────────────────────┘
                                ▼
                    ┌───────────────────────┐
                    │      Redis            │
                    │   (Task Queue + Cache)│
                    │       :6379           │
                    └───────────────────────┘
```

## 核心组件

### 1. Gateway (Kotlin)

Gateway 是系统的 HTTP 入口，负责接收外部请求并转发到 Coordinator。

#### 技术栈

- **框架**: Spring Boot 3 + WebFlux (响应式)
- **通信**: Reactor Netty
- **协议**: REST API, SSE, WebSocket
- **gRPC 客户端**: Reactor gRPC

#### 核心模块

```
com.cy.llm.gateway
├── controller/
│   ├── InferenceController    # 推理 API 控制器
│   ├── TrainingController     # 训练 API 控制器
│   └── HealthController       # 健康检查控制器
├── service/
│   ├── InferenceService       # 推理服务 (路由 + Resilience4j)
│   ├── TrainingService        # 训练服务
│   └── ModelRegistry          # 模型注册表
├── security/
│   ├── ApiKeyAuth             # API Key 认证
│   └── JwtAuth                # JWT 认证
└── config/
    ├── WebFluxConfig          # WebFlux 配置
    └── GrpcClientConfig       # gRPC 客户端配置
```

#### 端口与端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/v1/inference` | POST | 非流式推理 |
| `/api/v1/inference/stream` | POST | 流式推理 (SSE) |
| `/api/v1/training/start` | POST | 启动训练 |
| `/api/v1/training/status/{id}` | GET | 查询训练状态 |
| `/api/v1/models` | GET | 列出可用模型 |
| `/api/v1/health` | GET | 健康检查 |

### 2. Coordinator (Kotlin)

Coordinator 是任务调度中心，负责请求路由、负载均衡和缓存管理。

#### 技术栈

- **框架**: Spring Boot 3
- **数据存储**: Redis (Lettuce)
- **gRPC**: grpc-spring-boot-starter
- **指标**: Micrometer + Prometheus

#### 核心模块

```
com.cy.llm.coordinator
├── service/
│   ├── TaskQueueService       # Redis ZSET 优先级队列
│   ├── PromptCacheService     # Redis TTL 缓存
│   ├── WorkerPoolManager      # Worker 健康检查 + 负载均衡
│   ├── WorkerGrpcClient       # 异步 gRPC 客户端
│   └── TelemetryService       # 指标聚合
├── grpc/
│   └── CoordinatorGrpcImpl    # gRPC 服务实现
├── config/
│   ├── RedisConfig            # Redis 配置
│   └── GrpcServerConfig       # gRPC 服务端配置
└── model/
    └── DTOs                   # 数据传输对象
```

#### 工作流程

1. **任务队列** (TaskQueueService)
   - 使用 Redis ZSET 实现优先级队列
   - 支持任务排队和背压控制
   - 可配置最大队列大小

2. **Prompt 缓存** (PromptCacheService)
   - 使用 SHA256 哈希作为缓存 Key
   - 可配置 TTL 和最大缓存大小
   - 减少重复推理请求

3. **Worker 池管理** (WorkerPoolManager)
   - 心跳检测 (健康检查)
   - 加权轮询负载均衡
   - 熔断器模式 (Resilience4j)

### 3. Worker (Python)

Worker 是推理和训练的执行器，支持多种引擎和硬件平台。

#### 技术栈

- **异步**: asyncio + concurrent.futures
- **gRPC**: grpcio + grpcio-tools
- **推理引擎**: vLLM, TensorRT-LLM, MindIE
- **配置**: Pydantic, YAML

#### 核心模块

```
worker/
├── main.py                    # 入口点
├── core/
│   ├── server.py              # InferenceServer
│   ├── task_scheduler.py      # TaskScheduler
│   ├── memory_manager.py      # 显存管理
│   └── telemetry.py           # 遥测监控
├── engines/
│   ├── abstract_engine.py     # 引擎抽象基类
│   ├── engine_factory.py      # 引擎工厂
│   ├── vllm_cuda_engine.py    # vLLM CUDA 引擎
│   ├── vllm_async_engine.py   # vLLM 异步引擎
│   ├── trt_engine.py          # TensorRT-LLM 引擎
│   ├── ascend_engine.py       # Ascend 通用引擎
│   ├── mindie_engine.py       # MindIE 引擎
│   ├── hybrid_engine.py       # 混合引擎
│   └── vllm_ascend_engine.py  # vLLM Ascend 引擎
├── training/
│   ├── engine.py              # 训练引擎
│   ├── full_finetune.py       # 全参数微调
│   ├── custom_script_runner.py # 自定义脚本
│   └── model/
│       ├── setup.py           # 模型设置
│       └── lora.py            # LoRA 配置
├── config/
│   ├── config_loader.py       # 配置加载
│   ├── models.py              # 数据模型
│   └── validator.py           # 配置验证
├── cache/
│   └── prompt_cache.py        # Prompt 缓存
├── utils/
│   ├── vram_optimizer.py      # VRAM 优化
│   ├── stream_buffer.py       # 流式缓冲
│   └── auth.py                # 认证
└── tests/                     # 单元测试
```

## 数据流

### 推理请求流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              推理请求流程                                    │
└─────────────────────────────────────────────────────────────────────────────┘

  客户端                    Gateway               Coordinator              Worker
    │                         │                       │                      │
    │  1. HTTP POST           │                       │                      │
    │───────────────────────▶│                       │                      │
    │  (modelId, prompt)      │                       │                      │
    │                        │ 2. gRPC StreamPredict  │                      │
    │                        │───────────────────────▶│                      │
    │                        │                       │                      │
    │                        │                       │ 3. 检查缓存           │
    │                        │                       │◀─────────────────────│
    │                        │                       │                      │
    │                        │                       │ 4. 缓存命中?          │
    │                        │                       │                      │
    │                        │◀──────────────────────│ 5a. 返回缓存          │
    │                        │                       │                      │
    │◀───────────────────────│                       │                      │
    │  (cached response)      │                       │                      │
    │                        │                       │                      │
    │                        │                       │ 5b. 选择 Worker       │
    │                        │                       │◀─────────────────────│
    │                        │                       │                      │
    │                        │                       │ 6. gRPC StreamPredict │
    │                        │                       │─────────────────────▶│
    │                        │                       │                      │
    │                        │                       │                      │ 7. 加载模型
    │                        │                       │                      │◀─────────▶
    │                        │                       │                      │
    │                        │                       │                      │ 8. 执行推理
    │                        │                       │                      │◀─────────▶
    │                        │                       │                      │
    │                        │                       │◀─ 9. 流式响应 ───────│
    │                        │                       │                      │
    │◀───────────────────────│ 10. SSE 流式响应       │                      │
    │  (token chunks)         │                       │                      │
    │                        │                       │                      │
    │                        │                       │ 11. 缓存完整响应      │
    │                        │                       │─────────────────────▶│
    │                        │                       │                      │
```

### 训练请求流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              训练请求流程                                    │
└─────────────────────────────────────────────────────────────────────────────┘

  客户端                    Gateway               Coordinator              Worker
    │                         │                       │                      │
    │  1. POST /training/start│                       │                      │
    │───────────────────────▶│                       │                      │
    │  (baseModel, dataset)   │                       │                      │
    │                        │ 2. gRPC StartTraining │                      │
    │                        │───────────────────────▶│                      │
    │                        │                       │                      │
    │                        │                       │ 3. 加入任务队列       │
    │                        │                       │─────────────────────▶│
    │                        │                       │                      │
    │                        │                       │ 4. 选择 Worker       │
    │                        │                       │◀─────────────────────│
    │                        │                       │                      │
    │                        │                       │ 5. gRPC StartTraining│
    │                        │                       │─────────────────────▶│
    │                        │                       │                      │
    │                        │                       │                      │ 6. 加载数据集
    │                        │                       │                      │◀─────────▶
    │                        │                       │                      │
    │                        │                       │                      │ 7. 执行训练
    │                        │                       │                      │◀─────────▶
    │                        │                       │                      │
    │                        │◀─ 8. 进度更新 ────────│                      │
    │                        │                       │                      │
    │◀───────────────────────│ 9. SSE 进度更新       │                      │
    │  (progress events)      │                       │                      │
    │                        │                       │                      │
    │                        │                       │                      │ 10. 保存检查点
    │                        │                       │                      │◀─────────▶
    │                        │                       │                      │
    │                        │◀─ 11. 训练完成 ───────│                      │
    │                        │                       │                      │
    │◀───────────────────────│ 12. 训练完成响应      │                      │
    │  (completion event)     │                       │                      │
```

## 引擎架构

### 引擎继承层次

```
                        ┌─────────────────────┐
                        │  AbstractEngine     │
                        │  (抽象基类)          │
                        └──────────┬──────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
              ▼                    ▼                    ▼
    ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
    │  VllmCudaEngine │  │  TRTEngine      │  │  MindieEngine   │
    │  (CUDA vLLM)    │  │  (TensorRT)     │  │  (华为 MindIE)  │
    └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
             │                    │                    │
    ┌────────┴────────┐          │                    │
    │                 │          │                    │
    ▼                 ▼          │                    ▼
┌──────────┐   ┌──────────┐      │        ┌─────────────────┐
│VllmAsync │   │VllmAscend│      │        │  HybridEngine   │
│Engine    │   │Engine    │      │        │  (混合部署)      │
└──────────┘   └──────────┘      │        └─────────────────┘
                                 │
                                 ▼
                       ┌─────────────────┐
                       │  AscendEngine   │
                       │  (Ascend 通用)   │
                       └─────────────────┘
```

### 引擎配置

| 引擎 | 类名 | 硬件 | 特点 |
|------|------|------|------|
| CUDA vLLM | `VllmCudaEngine` | NVIDIA GPU | PagedAttention, 高吞吐 |
| CUDA 异步 | `VllmAsyncEngine` | NVIDIA GPU | 异步请求处理 |
| CUDA TRT | `TRTEngine` | NVIDIA GPU | TensorRT 优化 |
| Ascend vLLM | `VllmAscendEngine` | Ascend NPU | vLLM API 兼容 |
| Ascend 通用 | `AscendEngine` | Ascend NPU | 通用 Ascend 支持 |
| MindIE | `MindieEngine` | Ascend NPU | 华为官方优化 |
| 混合 | `HybridEngine` | 混合部署 | 跨平台支持 |

## 训练系统

### 训练引擎架构

```
TrainingEngine
│
├── 数据层 (data/)
│   ├── DataLoader        # 数据加载
│   ├── Formatter         # 格式化 (8种格式)
│   └── DatasetConfig     # 数据集配置
│
├── 模型层 (model/)
│   ├── ModelSetup        # 模型初始化
│   ├── LoRA              # LoRA 配置 (11种架构)
│   └── Quantization      # 量化配置
│
└── 训练层 (loop/)
    ├── Trainer           # 训练循环
    └── Callbacks         # 回调函数
```

### 支持的 LoRA 架构

| 架构 | 模型类型 |
|------|----------|
| `lora` | 标准 LoRA |
| `adalora` | 自适应 LoRA |
| `adaliprompt` | 自适应 prompt |
| `ia3` | IA3 |
| `loha` | LoHA |
| `lokr` | LoKR |
| `dora` | DoRA |
| `pissa` | PiSSA |
| `vera` | Vera |
| `hyenadna` | HyenaDNA |
| `fourier` | Fourier |

### 支持的数据格式

1. JSON (标准对话格式)
2. JSONL (行 JSON)
3. Alpaca (instruction + input + output)
4. ShareGPT (对话格式)
5. OpenAI (messages 格式)
6. HuggingFace (datasets 库)
7. CSV (文本 + 标签)
8. Parquet (高效列存储)

## 配置与扩展

### 配置文件结构

```
CY_LLM_Backend/
├── deploy/
│   ├── config.json              # 模型配置
│   ├── docker-compose.yml       # Docker 编排
│   └── .env.example             # 环境变量模板
├── worker/
│   ├── config/
│   │   ├── config_loader.py     # 配置加载器
│   │   ├── models.py            # Pydantic 模型
│   │   └── validator.py         # 配置验证
│   └── requirements.txt         # Python 依赖
└── gateway/
    └── src/main/resources/
        ├── application.yml      # Spring 配置
        └── application-dev.yml  # 开发环境配置
```

### 扩展新引擎

1. 继承 `AbstractEngine` 类
2. 实现所有抽象方法
3. 在 `engine_factory.py` 中注册
4. 添加配置验证规则

```python
from worker.engines.abstract_engine import AbstractEngine

class MyCustomEngine(AbstractEngine):
    """自定义引擎实现"""
    
    async def load_model(self, model_path: str, **kwargs):
        # 实现模型加载
        pass
    
    async def stream_predict(self, prompt: str, **kwargs):
        # 实现流式推理
        pass
```

### 扩展新数据格式

1. 继承 `BaseFormatter` 类
2. 实现 `format()` 方法
3. 在 `FormatterFactory` 中注册

## 监控指标

### Prometheus 指标

| 服务 | 指标前缀 | 关键指标 |
|------|----------|----------|
| Gateway | `gateway_` | `requests_total`, `latency_seconds` |
| Coordinator | `coordinator_` | `inference_requests_total`, `cache_hit_rate` |
| Worker | `worker_` | `inference_latency`, `gpu_memory_bytes` |

### 指标端点

| 服务 | 端点 |
|------|------|
| Gateway | `http://localhost:8080/actuator/prometheus` |
| Coordinator | `http://localhost:50050/actuator/prometheus` |
| Worker | `http://localhost:50051/metrics` |
