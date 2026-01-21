# CY-LLM Engine

> 🚀 **高性能** · **使用简洁** · **高度自定义** 的完整 AI 服务系统

一个支持多种推理引擎（vLLM / TensorRT-LLM / MindIE）和多种硬件平台（NVIDIA GPU / 华为 Ascend NPU）的统一 AI 推理后端。

---

## 📚 文档导航

> **🔗 完整文档在线阅读**: [https://zread.ai/Baijin64/CY-LLM-Engine](https://zread.ai/Baijin64/CY-LLM-Engine)

### 核心文档

| 文档 | 描述 |
|------|------|
| [docs/README.md](./docs/README.md) | 项目详细介绍与快速入门 |
| [docs/INSTALL.md](./docs/INSTALL.md) | 详细安装与配置指南 |
| [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) | 架构设计与数据流详解 |
| [docs/API.md](./docs/API.md) | REST API 与 gRPC 接口文档 |
| [docs/CONTRIBUTING.md](./docs/CONTRIBUTING.md) | 贡献者规范与开发指南 |
| [docs/TESTING.md](./docs/TESTING.md) | 测试指南与 CI 配置 |
| [docs/FAQ.md](./docs/FAQ.md) | 常见问题解答 |
| [docs/TRT_GUIDE.md](./docs/TRT_GUIDE.md) | TensorRT-LLM 专用指南 |

### 快速指南

| 文件 | 描述 |
|------|------|
| [QUICK_START.md](./QUICK_START.md) | 快速开始指南 (VRAM 优化、TRT 转换) |

### 项目历史

| 文件 | 描述 |
|------|------|
| [docs/HISTORY/MIGRATION_SUMMARY.md](./docs/HISTORY/MIGRATION_SUMMARY.md) | EW_AI_Backend → CY_LLM_Backend 迁移总结 |
| [docs/HISTORY/PHASE2_3_UPGRADE_REPORT.md](./docs/HISTORY/PHASE2_3_UPGRADE_REPORT.md) | Phase 2 & 3 优化升级报告 |

---

## ✨ 核心特性

| 特性 | 描述 |
|------|------|
| **多引擎支持** | vLLM (CUDA/Ascend)、TensorRT-LLM、MindIE |
| **多硬件平台** | NVIDIA GPU、华为 Ascend NPU |
| **统一 CLI** | `./cy` / `./cy-llm` 一键部署和管理 |
| **流式推理** | SSE 实时流式输出 |
| **企业级网关** | Kotlin + Spring WebFlux 响应式架构 |
| **弹性伸缩** | 支持多 Worker 实例负载均衡 |
| **完整训练** | LoRA/PEFT 微调支持 |
| **显存优化** | VRAM 预估与 OOM 自动重试 |

---

## 🚀 快速开始

### 方式一：CLI 一键启动 (推荐)

```bash
# 1. 初始化环境
./cy-llm setup --engine cuda-vllm

# 2. 启动服务
./cy-llm start --model qwen2.5-7b

# 3. 验证部署
curl http://localhost:8080/api/v1/health
```

### 方式二：Docker 部署 (推荐生产)

```bash
# 配置环境变量
cd CY_LLM_Backend/deploy
cp .env.example .env
vim .env

# 启动服务
docker compose up -d
```

### 方式三：手动启动

```bash
# 1. 启动 Redis
docker run -d --name redis -p 6379:6379 redis:7-alpine

# 2. 启动 Coordinator (Java 21)
cd CY_LLM_Backend/coordinator
./gradlew bootRun

# 3. 启动 Worker (Python)
cd CY_LLM_Backend/worker
source .venv/bin/activate
python -m worker.main --serve --port 50051

# 4. 启动 Gateway (Java 21)
cd CY_LLM_Backend/gateway
./gradlew bootRun
```

服务将在 `http://localhost:8080` 启动。

---

## 🎯 引擎选择指南

| 引擎 | 硬件 | 特点 | 适用场景 |
|------|------|------|----------|
| `cuda-vllm` | NVIDIA GPU | PagedAttention, 高吞吐 | 通用推荐 |
| `cuda-trt` | NVIDIA GPU | 极致性能, 需预编译 | 固定模型生产 |
| `ascend-vllm` | 华为 NPU | 兼容 vLLM API | Ascend 环境 |
| `ascend-mindie` | 华为 NPU | 官方优化 | Ascend 高性能 |

```bash
# 使用 vLLM (默认)
./cy-llm start --engine cuda-vllm

# 使用 TensorRT-LLM
./cy-llm start --engine cuda-trt

# 使用华为 Ascend vLLM
./cy-llm start --engine ascend-vllm

# 使用华为 Ascend MindIE
./cy-llm start --engine ascend-mindie
```

---

## 📖 CLI 命令参考

```bash
./cy-llm <command> [options]

Commands:
  setup       初始化环境 (Conda + 依赖 + Gateway)
  start       启动完整服务 (Gateway + Worker)
  worker      仅启动 Worker
  stop        停止所有服务
  status      查看服务状态
  docker      Docker Compose 部署
  test        运行测试
  models      模型管理
  convert-trt 转换模型为 TensorRT-LLM 引擎
  help        显示帮助

Options:
  --engine TYPE     推理引擎 (cuda-vllm/cuda-trt/ascend-vllm/ascend-mindie)
  --model ID        模型 ID
  --port PORT       Gateway 端口 (默认: 8080)
  -d, --daemon      后台运行

Examples:
  ./cy-llm setup --engine cuda-vllm       # 初始化
  ./cy start --model qwen2.5-72b          # 启动指定模型
  ./cy-llm start -d                       # 后台启动
  ./cy-llm docker up --scale 2            # Docker 双 Worker
  ./cy-llm status                         # 查看状态
  ./cy-llm convert-trt --model Qwen/Qwen2.5-7B --output /models/trt  # 转换 TRT 模型
```

---

## 🛠 推理接口示例

### 流式推理 (SSE)

```bash
curl -N -X POST http://localhost:8080/api/v1/inference/stream \
  -H "Content-Type: application/json" \
  -d '{"modelId": "qwen2.5-7b", "prompt": "请描述一下未来 AI 的样子"}'
```

### 非流式推理

```bash
curl -X POST http://localhost:8080/api/v1/inference \
  -H "Content-Type: application/json" \
  -d '{"modelId": "qwen2.5-7b", "prompt": "你好，请介绍一下自己"}'
```

### 训练接口

```bash
curl -X POST http://localhost:8080/api/v1/training/start \
  -H "Content-Type: application/json" \
  -d '{
    "baseModel": "deepseek-ai/deepseek-llm-7b-chat",
    "outputDir": "/checkpoints/my_lora",
    "datasetPath": "/data/train.json"
  }'
```

---

## 🧪 测试

```bash
# 运行集成测试
./cy-llm test integration

# 运行单元测试
./cy-llm test unit

# 运行所有测试
./cy-llm test all
```

---

## 🏗 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client                                   │
│                  (Browser / API Client)                         │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP / SSE / WebSocket
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Gateway (Kotlin)                              │
│              Spring WebFlux + gRPC Client                        │
│  端口: 8080                                                      │
└───────────────────────────┬─────────────────────────────────────┘
                            │ gRPC (:50050)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Coordinator (Kotlin)                           │
│              Spring Boot + Redis + gRPC Server                   │
│  端口: 50050                                                     │
│  • TaskQueueService (Redis ZSET 优先级队列)                     │
│  • PromptCacheService (Redis TTL 缓存)                          │
│  • WorkerPoolManager (健康检查 + 负载均衡)                      │
└───────────────────────────┬─────────────────────────────────────┘
                            │ gRPC (:50051)
          ┌─────────────────┴─────────────────┐
          ▼                                   ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│      Worker (Python)    │     │      Worker (Python)    │
│      NVIDIA GPU         │     │      Ascend NPU         │
│  ┌───────────────────┐  │     │  ┌───────────────────┐  │
│  │ InferenceEngine   │  │     │  │ InferenceEngine   │  │
│  │  └─ vLLM/TensorRT │  │     │  │  └─ MindIE/vLLM   │  │
│  └───────────────────┘  │     │  └───────────────────┘  │
│  ┌───────────────────┐  │     │  ┌───────────────────┐  │
│  │ TrainingEngine    │  │     │  │ TrainingEngine    │  │
│  │  └─ LoRA/PEFT     │  │     │  │  └─ LoRA/PEFT     │  │
│  └───────────────────┘  │     │  └───────────────────┘  │
└─────────────────────────┘     └─────────────────────────┘
          │                                   │
          └─────────────────┬─────────────────┘
                            ▼
                    ┌───────────────────┐
                    │      Redis        │
                    │   (任务队列+缓存)  │
                    │      :6379        │
                    └───────────────────┘
```

详细架构设计请参考 [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md)。

---

## 📁 项目结构

```
CY-LLM-Engine/
├── cy                           # 主 CLI 工具 (Shell 脚本)
├── cy-llm                       # CLI 别名
├── CY_LLM_Backend/
│   ├── gateway/                # Kotlin Gateway 服务 (Spring WebFlux)
│   ├── coordinator/            # Kotlin Coordinator 服务 (Spring Boot)
│   ├── worker/                 # Python Worker 服务
│   │   ├── main.py             # 入口点
│   │   ├── core/               # 核心组件 (server, scheduler, memory, telemetry)
│   │   ├── engines/            # 推理引擎 (vLLM, TRT, MindIE, Ascend)
│   │   ├── training/           # 训练引擎 (LoRA/PEFT)
│   │   ├── config/             # 配置管理
│   │   ├── cache/              # 缓存服务
│   │   ├── utils/              # 工具函数
│   │   └── tests/              # 单元测试
│   ├── deploy/                 # 部署配置 (docker-compose, config.json)
│   └── proto/                  # gRPC 协议定义
├── CY_LLM_Training/            # 训练相关代码
├── scripts/                    # 辅助脚本
│   ├── convert_trt.py          # TRT 模型转换工具
│   └── diagnose_env.py         # 环境诊断
├── docs/                       # 项目文档
│   ├── HISTORY/                # 项目历史文档
│   ├── README.md               # 项目详细介绍
│   ├── INSTALL.md              # 安装指南
│   ├── ARCHITECTURE.md         # 架构设计
│   ├── API.md                  # API 文档
│   ├── CONTRIBUTING.md         # 贡献规范
│   ├── TESTING.md              # 测试指南
│   ├── FAQ.md                  # 常见问题
│   └── TRT_GUIDE.md            # TRT 使用指南
├── QUICK_START.md              # 快速开始指南
├── requirements*.txt           # Python 依赖
└── LICENSE
```

---

## ⚙️ 环境要求

| 组件 | 最低版本 | 推荐版本 |
|------|----------|----------|
| Python | 3.10+ | 3.11 |
| Java | 21+ | 21 (LTS) |
| CUDA | 12.0 | 12.4 |
| CANN | 8.0+ | 8.0.RC1 |
| Redis | 7.0 | 7.2 |
| Docker | 24.0 | 25.0 |

---

## 🏷 版本号规范

本项目采用 **四段式版本号**: `[major.minor.patch.build-SUFFIX]`

| 段 | 含义 | 说明 |
|----|------|------|
| major | 重大不兼容变化 | 破坏性重构 |
| minor | 向后兼容新功能 | feature |
| patch | Bug 修复与优化 | bugfix |
| build | 构建/测试次数 | 递增 |
| SUFFIX | 稳定性后缀 | PreAlpha/Alpha/Beta/RC/Release |

提交消息格式：
```
[2.1.1.2-Alpha] feat(worker): add async inference support

Add async inference support for vLLM engine.

【中文】worker：添加异步推理支持
```

详细规范请参考 [docs/CONTRIBUTING.md](./docs/CONTRIBUTING.md)。

---

## 📝 版本历史

- **[1.5.2.0]** - 简化部署流程，统一 CLI 工具，支持四种推理引擎
- **[1.5.1.3-alpha]** - C++ 入口点，四引擎架构实现
- **[1.5.0.0-alpha]** - VRAM 优化系统、TRT 真流式输出
- **[1.0.0-alpha]** - 初始版本，Gateway + Worker 基础架构

---

## 🤝 贡献

欢迎提交 Issue 或 Pull Request！

详细贡献指南请参考 [docs/CONTRIBUTING.md](./docs/CONTRIBUTING.md)。

---

## 📄 许可证

本项目采用 MIT 许可证，详见 [LICENSE](./LICENSE) 文件。

---

## 🔗 相关链接

- **在线文档**: [https://zread.ai/Baijin64/CY-LLM-Engine](https://zread.ai/Baijin64/CY-LLM-Engine)
- **项目仓库**: [https://github.com/Baijin64/CY-LLM-Engine](https://github.com/Baijin64/CY-LLM-Engine)
- **问题反馈**: [GitHub Issues](https://github.com/Baijin64/CY-LLM-Engine/issues)
