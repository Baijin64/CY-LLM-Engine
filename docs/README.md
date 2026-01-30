# CY-LLM Engine 项目详解

> 高性能 · 使用简洁 · 高度自定义 的完整 AI 服务系统

## 项目简介

CY-LLM Engine 是一个统一的大语言模型推理和训练后端系统，支持多种推理引擎和硬件平台。项目采用现代化的微服务架构，包含 Gateway（HTTP 网关）、Coordinator（任务调度中心）和 Worker（推理/训练执行器）三个核心组件，通过 gRPC 进行内部通信。

### 核心特性

| 特性 | 描述 |
|------|------|
| **多引擎支持** | vLLM (CUDA/Ascend)、TensorRT-LLM、MindIE |
| **多硬件平台** | NVIDIA GPU、华为 Ascend NPU |
| **统一 CLI** | `./cy` / `./cy-llm` 一键部署和管理 |
| **基础推理** | OpenAI 兼容非流式输出 |
| **弹性伸缩** | 轻量版可扩展多 Worker 实例 |
| **完整训练** | LoRA/PEFT 微调支持 |

## 技术栈

### 后端 (Python)

| 组件 | 技术/框架 |
|------|----------|
| 推理引擎 | vLLM, TensorRT-LLM, MindIE |
| gRPC | grpcio, grpcio-tools |
| 异步处理 | asyncio, concurrent.futures |
| 配置管理 | Pydantic, YAML |
| 测试 | pytest, pytest-asyncio |

### 网关 (Lite / Python)

| 组件 | 技术/框架 |
|------|----------|
| 框架 | FastAPI |
| 通信 | grpcio (gRPC Client) |
| 模型接口 | OpenAI 兼容接口 |

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client                                   │
│                  (Browser / API Client)                         │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Gateway Lite (Python)                            │
│               FastAPI + gRPC Client                              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ • /v1/chat/completions (OpenAI 兼容)                     │    │
│  │ • 简单鉴权 (Token)                                       │    │
│  └─────────────────────────────────────────────────────────┘    │
└───────────────────────────┬─────────────────────────────────────┘
                            │ gRPC (:50051)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│               Coordinator Lite (Python)                          │
│                 gRPC Proxy + 简化调度                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ • 透传请求到 Worker                                     │    │
│  │ • 简单负载策略 (可选)                                   │    │
│  └─────────────────────────────────────────────────────────┘    │
└───────────────────────────┬─────────────────────────────────────┘
                            │ gRPC (:50052)
          ┌─────────────────┴─────────────────┐
          ▼                                   ▼
┌─────────────────────────┐       ┌─────────────────────────┐
│      Worker (Python)    │       │      Worker (Python)    │
│      NVIDIA GPU         │       │      Ascend NPU         │
│  ┌───────────────────┐  │       │  ┌───────────────────┐  │
│  │ InferenceEngine   │  │       │  │ InferenceEngine   │  │
│  │  └─ vLLM/TensorRT │  │       │  │  └─ MindIE/vLLM   │  │
│  └───────────────────┘  │       │  └───────────────────┘  │
│  ┌───────────────────┐  │       │  ┌───────────────────┐  │
│  │ TrainingEngine    │  │       │  │ TrainingEngine    │  │
│  │  └─ LoRA/PEFT     │  │       │  │  └─ LoRA/PEFT     │  │
│  └───────────────────┘  │       │  └───────────────────┘  │
└─────────────────────────┘       └─────────────────────────┘
```

## 项目结构

```
CY-LLM-Engine/
├── cy                           # 主 CLI 工具 (Shell 脚本)
├── cy-llm                       # CLI 别名
├── docs/                        # 项目文档 (本文档所在目录)
│   ├── README.md               # 项目详细介绍
│   ├── INSTALL.md              # 安装与运行指南
│   ├── ARCHITECTURE.md         # 架构设计详解
│   ├── API.md                  # API 接口文档
│   ├── CONTRIBUTING.md         # 贡献者规范
│   ├── TESTING.md              # 测试指南
│   ├── FAQ.md                  # 常见问题
│   └── TRT_GUIDE.md            # TensorRT-LLM 使用指南
├── CY_LLM_Backend/
|   ├── gateway_lite/           # Python Gateway Lite (FastAPI)
|   ├── coordinator_lite/       # Python Coordinator Lite (gRPC Proxy)
│   ├── worker/                 # Python Worker 服务
│   │   ├── main.py             # 入口点
│   │   ├── core/               # 核心组件
│   │   │   ├── server.py       # 推理服务器
│   │   │   ├── task_scheduler.py  # 任务调度器
│   │   │   ├── memory_manager.py  # 显存管理
│   │   │   └── telemetry.py    # 遥测监控
│   │   ├── engines/            # 推理引擎
│   │   │   ├── vllm_cuda_engine.py
│   │   │   ├── vllm_async_engine.py
│   │   │   ├── trt_engine.py
│   │   │   ├── ascend_engine.py
│   │   │   ├── mindie_engine.py
│   │   │   ├── hybrid_engine.py
│   │   │   └── engine_factory.py
│   │   ├── training/           # 训练引擎
│   │   │   ├── engine.py
│   │   │   ├── full_finetune.py
│   │   │   ├── custom_script_runner.py
│   │   │   └── model/
│   │   │       ├── setup.py
│   │   │       └── lora.py
│   │   ├── config/             # 配置管理
│   │   ├── cache/              # 缓存服务
│   │   ├── utils/              # 工具函数
│   │   ├── tests/              # 单元测试
│   │   └── proto_gen/          # gRPC 协议文件
│   ├── deploy/                 # 部署配置
│   │   ├── docker-compose.yml
│   │   ├── config.json         # 模型配置
│   │   └── .env.example
│   └── ARCHITECTURE.md         # 架构说明
├── CY_LLM_Training/            # 训练相关代码
│   ├── src/
│   │   ├── train_lora.py
│   │   ├── dataset_converter.py
│   │   └── inference.py
│   └── dialogue_gi/
├── scripts/                    # 辅助脚本
│   ├── convert_trt.py          # TRT 模型转换
│   └── diagnose_env.py         # 环境诊断
├── requirements*.txt           # Python 依赖
├── TESTING.md                  # 测试说明
├── QUICK_START.md              # 快速开始
├── CONTRIBUTING.md             # 贡献指南
└── LICENSE
```

## 快速开始

### 环境要求

| 组件 | 最低版本 |
|------|----------|
| Python | 3.10+ |
| CUDA | 12.0+ (NVIDIA) |
| CANN | 8.0+ (Ascend) |
| Redis | 可选 |

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/Baijin64/CY-LLM-Engine.git
cd CY-LLM-Engine

# 2. 初始化环境 (选择引擎)
./cy-llm setup --engine cuda-vllm    # NVIDIA GPU
./cy-llm setup --engine cuda-trt     # TensorRT-LLM
./cy-llm setup --engine ascend-vllm  # Ascend NPU

# 3. 安装 Lite 依赖
conda run -n ${CY_LLM_CONDA_ENV:-vllm} pip install -r CY_LLM_Backend/gateway_lite/requirements.txt
conda run -n ${CY_LLM_CONDA_ENV:-vllm} pip install -r CY_LLM_Backend/coordinator_lite/requirements.txt

# 4. 启动 Lite 服务
./cy-llm lite --model <model-id>

# 5. 验证部署
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"<model-id>","messages":[{"role":"user","content":"你好"}]}'
```

### Docker 部署 (Lite)

```bash
# 说明: Lite Docker Compose 将在后续补充
# 目前可使用 CLI 启动进行本地联调
```

## 引擎选择指南

| 引擎 | 硬件 | 特点 | 适用场景 |
|------|------|------|----------|
| `cuda-vllm` | NVIDIA GPU | PagedAttention, 高吞吐 | 通用推荐 |
| `cuda-trt` | NVIDIA GPU | 极致性能, 需预编译 | 固定模型生产 |
| `ascend-vllm` | 华为 NPU | 兼容 vLLM API | Ascend 环境 |
| `ascend-mindie` | 华为 NPU | 官方优化 | Ascend 高性能 |

## 主要组件 (Lite)

### Gateway Lite (端口 8000)

- OpenAI 兼容接口（/v1/chat/completions）
- 简单鉴权（Token）
- gRPC 透传到 Coordinator Lite

### Coordinator Lite (端口 50051)

- gRPC 透传到 Worker
- 简化调度（可扩展负载策略）

### Worker (端口 50052)

- 推理引擎：vLLM / TensorRT-LLM / MindIE
- 训练能力保留（按需使用）

## 监控与指标

Lite 版本暂不内置完整监控指标；可根据需要后续扩展。

## 文档导航

| 文档 | 说明 |
|------|------|
| [INSTALL.md](./INSTALL.md) | 详细安装与配置指南 |
| [ARCHITECTURE.md](./ARCHITECTURE.md) | 架构设计与数据流 |
| [API.md](./API.md) | REST API 与 gRPC 接口定义 |
| [CONTRIBUTING.md](./CONTRIBUTING.md) | 贡献者规范与开发指南 |
| [TESTING.md](./TESTING.md) | 测试说明与 CI 配置 |
| [FAQ.md](./FAQ.md) | 常见问题解答 |
| [TRT_GUIDE.md](./TRT_GUIDE.md) | TensorRT-LLM 专用指南 |

## 许可证

本项目采用 MIT 许可证，详见 [LICENSE](../LICENSE) 文件。

## 贡献

欢迎提交 Issue 或 Pull Request！请先阅读 [CONTRIBUTING.md](./CONTRIBUTING.md) 了解贡献规范。

## 联系

- 项目地址: https://github.com/Baijin64/CY-LLM-Engine
- 问题反馈: https://github.com/Baijin64/CY-LLM-Engine/issues
