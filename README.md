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
| [QUICK_START.md](./QUICK_START.md) | 快速开始指令速查 |

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
| **基础推理** | OpenAI 兼容非流式输出 |
| **轻量化网关** | Python + FastAPI 轻量版 |
| **弹性伸缩** | 轻量版可扩展多 Worker 实例 |
| **完整训练** | LoRA/PEFT 微调支持 |
| **显存优化** | VRAM 预估与 OOM 自动重试 |

---

## 🚀 快速开始 (Community Lite)

### 本地启动

```bash
# 1. 初始化环境
./cy-llm setup --engine cuda-vllm

# 2. 安装 Lite 依赖
conda run -n ${CY_LLM_CONDA_ENV:-vllm} pip install -r CY_LLM_Backend/gateway_lite/requirements.txt
conda run -n ${CY_LLM_CONDA_ENV:-vllm} pip install -r CY_LLM_Backend/coordinator_lite/requirements.txt

# 3. 一键启动 (Lite Gateway + Lite Coordinator + Worker)
./cy-llm lite --engine cuda-vllm --model qwen2.5-7b

# 4. 测试 (OpenAI 兼容)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-7b","messages":[{"role":"user","content":"你好"}]}'
```

> Lite 版本默认端口为 8000（Gateway），Coordinator 为 50051，Worker 为 50052。

### Docker Compose 启动

```bash
# 启动
docker compose -f docker-compose.community.yml up -d

# 查看状态
docker compose -f docker-compose.community.yml ps

# 停止
docker compose -f docker-compose.community.yml down
```

### 使用 VRAM 预估和优化

```bash
# 诊断环境与模型显存需求
./cy-llm diagnose qwen2.5-7b

# 转换模型为 TensorRT-LLM 引擎（可选，提升性能）
./cy-llm convert-trt --model Qwen/Qwen2.5-7B-Instruct --output /models/qwen2.5-7b-trt

# 使用 TRT 引擎启动
./cy-llm lite --engine cuda-trt --model qwen2.5-7b-trt
```

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
./cy-llm lite --engine cuda-vllm

# 使用 TensorRT-LLM
./cy-llm lite --engine cuda-trt

# 使用华为 Ascend vLLM
./cy-llm lite --engine ascend-vllm

# 使用华为 Ascend MindIE
./cy-llm lite --engine ascend-mindie
```

---

## 📖 CLI 命令参考

```bash
./cy-llm <command> [options]

Commands:
  setup       初始化环境 (Conda + 依赖)
  lite        启动轻量版服务 (Gateway Lite + Coordinator Lite + Worker)
  worker      仅启动 Worker
  stop        停止所有服务
  status      查看服务状态
  docker      Docker Compose 部署 (后续补充 Lite)
  test        运行测试
  models      模型管理
  convert-trt 转换模型为 TensorRT-LLM 引擎
  prepare     数据预处理
  train       LoRA 微调训练
  chat        交互式 LoRA 推理
  help        显示帮助

Options:
  --engine TYPE     推理引擎 (cuda-vllm/cuda-trt/ascend-vllm/ascend-mindie)
  --model ID        模型 ID
  --port PORT       Lite Gateway 端口 (默认: 8000)
  -d, --daemon      后台运行

Examples:
  ./cy-llm setup --engine cuda-vllm       # 初始化
  ./cy-llm lite --engine cuda-vllm --model qwen2.5-7b  # Lite 一键启动
  ./cy-llm status                         # 查看状态
  ./cy-llm convert-trt --model Qwen/Qwen2.5-7B --output /models/trt  # 转换 TRT 模型
  ./cy-llm prepare --raw /data/raw --out /data/train.jsonl           # 预处理数据
  ./cy-llm train --dataset /data/train.jsonl --output /models/lora   # 启动训练
  ./cy-llm chat --model qwen2.5-7b --lora /models/lora               # 加载 LoRA 对话
```

---

## 🛠 推理接口示例 (Lite)

### 非流式推理 (OpenAI 兼容)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-7b","messages":[{"role":"user","content":"你好，请介绍一下自己"}]}'
```

---

## 🧪 测试

```bash
 # 运行集成测试 (默认运行核心集成测试)
 ./cy-llm test integration
 
+# 运行特定集成测试 (例如: engine, memory, scheduler, stream, telemetry, all)
+./cy-llm test integration all
+
# 运行所有测试
./cy-llm test all
```

---

## 🏗 系统架构 (Lite)

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
│  端口: 8000                                                      │
└───────────────────────────┬─────────────────────────────────────┘
                            │ gRPC (:50051)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│               Coordinator Lite (Python)                          │
│                 gRPC Proxy + 简化调度                            │
│  端口: 50051                                                     │
└───────────────────────────┬─────────────────────────────────────┘
                            │ gRPC (:50052)
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
                    │   (可选) Redis    │
                    │   (后续扩展)      │
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
|   ├── gateway_lite/           # Python Gateway Lite (FastAPI)
|   ├── coordinator_lite/       # Python Coordinator Lite (gRPC Proxy)
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

## ⚙️ 配置说明

### 关键环境变量

#### Gateway Lite
- `COORDINATOR_GRPC_ADDR`：Coordinator gRPC 地址（默认 `127.0.0.1:50051`）
- `GATEWAY_API_TOKEN`：可选的静态鉴权 Token
- `GATEWAY_REQUEST_TIMEOUT`：请求超时（秒，默认 60）

#### Coordinator Lite
- `COORDINATOR_GRPC_BIND`：Coordinator 监听地址（默认 `0.0.0.0:50051`）
- `WORKER_GRPC_ADDRS`：Worker 列表（用逗号分隔）
- `COORDINATOR_CONFIG`：可选配置文件路径（JSON）

示例配置：
```json
{
  "workers": ["worker-1:50052", "worker-2:50052"]
}
```

#### Worker
- `CY_LLM_ENGINE`：引擎类型（如 `cuda-vllm`）
- `CY_LLM_DEFAULT_MODEL`：默认模型 ID 或本地路径
- `CY_LLM_DEFAULT_ADAPTER`：LoRA 适配器路径（可选）
- `CY_LLM_MODEL_REGISTRY`：模型注册表 JSON（可选，字符串）
- `CY_LLM_MODEL_REGISTRY_PATH`：模型注册表路径（可选）
- `CY_LLM_HEALTH_PORT`：健康检查端口（默认 `9090`）
- `VLLM_GPU_MEMORY_UTILIZATION`：GPU 显存利用率（默认 `0.8`）

### 模型配置文件

模型配置文件位于 `CY_LLM_Backend/deploy/models.json`：

```json
{
  "qwen2.5-7b": {
    "model_path": "Qwen/Qwen2.5-7B-Instruct",
    "engine": "cuda-vllm",
    "gpu_memory_utilization": 0.85,
    "max_model_len": 4096,
    "quantization": null
  }
}
```

---

## ⚙️ 环境要求

| 组件 | 最低版本 | 推荐版本 |
|------|----------|----------|
| Python | 3.10+ | 3.11 |
| CUDA | 12.0 | 12.4 |
| CANN | 8.0+ | 8.0.RC1 |
| Redis | 可选 | 可选 |
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
