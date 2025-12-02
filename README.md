# CY-LLM Engine

> 🚀 **高性能** · **使用简洁** · **高度自定义** 的完整 AI 服务系统

一个支持多种推理引擎（vLLM (vLLM for Ascend) / TensorRT-LLM / MindIE）、多种硬件平台（NVIDIA GPU / 华为 Ascend NPU）的统一 AI 推理后端。

## ✨ 特性

- **四种推理引擎**: `cuda-vllm` / `cuda-trt` / `ascend-vllm` / `ascend-mindie`
- **一键部署**: 统一的 `./ew` 命令行工具
- **流式推理**: SSE 实时流式返回
- **企业级网关**: Kotlin + Spring WebFlux 响应式架构
- **弹性伸缩**: 支持多 Worker 实例
- **双平台支持**: NVIDIA CUDA 与 华为 Ascend NPU

## 🚀 快速开始

### 30 秒启动

```bash
# 1. 初始化环境
./ew setup --engine cuda-vllm

# 2. 启动服务
./ew start --model deepseek-v3
```

服务将在 `http://localhost:8080` 启动。

### 测试推理

```bash
curl -X POST http://localhost:8080/api/v1/inference/stream \
  -H "Content-Type: application/json" \
  -d '{"modelId": "deepseek-v3", "prompt": "你好"}' \
  --no-buffer
```

## 📦 安装

### 环境要求

| 组件 | 最低版本 |
|------|----------|
| Python | 3.10+ |
| Java | 21+ |
| CUDA | 12.0+ (NVIDIA) |
| CANN | 8.0+ (Ascend) |

### 方式一：本地部署 (推荐开发)

```bash
# 克隆仓库
git clone https://github.com/Baijin64/CY-LLM-Engine.git
cd CY-LLM-Engine

# 初始化环境
./ew setup

./ew setup --engine cuda-vllm
./ew start
```

### 方式二：Docker 部署 (推荐生产)
./ew start --model deepseek-v3
```bash
cd CY-LLM-Engine

# 配置环境变量
cp .env.example .env
vim .env  # 编辑配置

# 启动服务
./ew docker up
```

## 🎯 引擎选择指南

| 引擎 | 硬件 | 特点 | 适用场景 |
|------|------|------|----------|
| `cuda-vllm` | NVIDIA GPU | PagedAttention, 高吞吐 | 通用推荐 |
| `cuda-trt` | NVIDIA GPU | 极致性能, 需预编译 | 固定模型生产 |
| `ascend-vllm` | 华为 NPU | 兼容 vLLM API | Ascend 环境 |
| `ascend-mindie` | 华为 NPU | 官方优化 | Ascend 高性能 |

```bash
# 使用 vLLM (默认)
./ew start --engine cuda-vllm

# 使用 TensorRT-LLM
./ew start --engine cuda-trt

# 使用华为 Ascend
./ew start --engine ascend-vllm
```

## 📖 CLI 命令参考

```bash
./ew <command> [options]

git clone https://github.com/Baijin64/CY-LLM-Engine.git
  setup       初始化环境 (Conda + 依赖 + Gateway)
  start       启动完整服务 (Gateway + Worker)
  worker      仅启动 Worker
  stop        停止所有服务
  status      查看服务状态
  docker      Docker Compose 部署
  test        运行测试
  models      模型管理
  help        显示帮助

常用选项:
  --engine TYPE     推理引擎 (cuda-vllm/cuda-trt/ascend-vllm/ascend-mindie)
  --model ID        模型 ID
  --port PORT       Gateway 端口 (默认: 8080)
  -d, --daemon      后台运行

示例:
  ./ew setup --engine cuda-vllm       # 初始化
  ./ew start --model qwen2.5-72b      # 启动指定模型
  ./ew start -d                       # 后台启动
  ./ew docker up --scale 2            # Docker 双 Worker
  ./ew status                         # 查看状态
```

## 🛠 使用方法（详细）

下面提供开发、Docker 部署与 API 使用的分步说明，便于快速上手与调试。

### 1) 本地开发（推荐）

1. 准备 Python 环境并安装依赖：
```bash
cd CY_LLM_Backend/worker
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. 启动 Redis（如果需要）：
```bash
docker run -d --name redis -p 6379:6379 redis:7-alpine
```

3. 启动 Coordinator（JDK 21）：
```bash
cd CY_LLM_Backend/coordinator
JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 ./gradlew bootRun
```

4. 启动 Worker：
```bash
cd CY_LLM_Backend/worker
python -m worker.main --serve --port 50051
```

5. 启动 Gateway：
```bash
cd CY_LLM_Backend/gateway
./gradlew bootRun
```

6. 使用 `curl` 验证接口（示例）：
```bash
curl -X POST http://localhost:8080/api/v1/inference/stream \
  -H "Content-Type: application/json" \
  -d '{"modelId": "furina", "prompt": "你好"}' --no-buffer
```

### 2) Docker / Docker Compose（快速生产/测试）

```bash
cd CY_LLM_Backend/deploy
cp .env.example .env
# 编辑 .env 填写 API Key / Coordinator / Worker 配置
vim .env

# 构建镜像并启动（默认包含 Gateway + Coordinator + Worker）
docker compose up -d --build

# 查看日志
docker compose logs -f gateway
```

### 3) 使用 CLI（`./ew`）
# 核心配置（使用 CY_LLM_*）
CLI `ew` 提供多种便捷操作：
```bash
./ew start --engine cuda-vllm --model furina # 启动服务
CY_LLM_MODEL=deepseek-v3     # 默认模型（优先）
./ew stop                                     # 停止所有服务
./ew test unit                                # 运行单元测试
```

### 4) 模型管理（添加 / 更新）

编辑 `CY_LLM_Backend/deploy/config.json`：

```json
{
  "models": {
    "furina": {
      "engine": "cuda-vllm",
      "model_path": "deepseek-ai/deepseek-llm-7b-chat",
      "adapter_path": "/checkpoints/furina_lora",
      "max_model_len": 8192
    }
  }
}
```

编辑后重新加载或重启 Worker。

### 5) 推理接口示例（SSE 实时流）

```bash
curl -N -H "Content-Type: application/json" -X POST \
  http://localhost:8080/api/v1/inference/stream \
  -d '{"modelId": "furina", "prompt": "请描述一下未来 AI 的样子"}'
```

### 6) 训练接口示例

```bash
curl -H "Content-Type: application/json" -X POST http://localhost:8080/api/v1/training/start \
  -d '{"baseModel": "deepseek-ai/deepseek-llm-7b-chat", "outputDir": "/checkpoints/furina_lora", "datasetPath": "/data/train.json"}'
```

### 7) 运行测试

Python Worker 单元测试：
```bash
cd CY_LLM_Backend/worker
pytest tests/ -q
```

Kotlin Gateway / Coordinator 测试：
```bash
cd CY_LLM_Backend/gateway
./gradlew test

cd CY_LLM_Backend/coordinator
> 说明：我们使用 `CY_LLM_*` 前缀的环境变量（例如 `CY_LLM_PORT`, `CY_LLM_INTERNAL_TOKEN` 等）。请使用 `CY_LLM_*` 变量。
```

更多高级用法见 `TESTING.md` 或 `CONTRIBUTING.md`。

## �� 架构

```
┌─────────────┐      HTTP/SSE      ┌─────────────────┐
│   Client    │  ───────────────▶  │     Gateway     │
│  (Browser)  │                    │  (Kotlin/Spring)│
└─────────────┘                    └────────┬────────┘
                                            │ gRPC
                                            ▼
                                   ┌─────────────────┐
                                   │     Worker      │
                                   │    (Python)     │
                                   └────────┬────────┘
                                            │
                    ┌───────────────────────┼───────────────────────┐
                    │                       │                       │
              ┌─────▼─────┐          ┌──────▼──────┐         ┌──────▼──────┐
              │ cuda-vllm │          │  cuda-trt   │         │ascend-vllm  │
              │   (vLLM)  │          │(TensorRT)   │         │(vLLM-Ascend)│
              └───────────┘          └─────────────┘         └─────────────┘
```

## �� 项目结构

```
CY-LLM-Engine/
├── ew                          # 🔧 统一 CLI 工具
├── EW_AI_Backend/
│   ├── gateway/                # Kotlin Gateway 服务
│   │   └── src/main/kotlin/    # Spring WebFlux + gRPC
│   │   │   ├── vllm_ascend_engine.py
│   │   │   └── mindie_engine.py
│   │   └── core/               # 核心组件
│   ├── deploy/                 # 部署配置
│   │   ├── docker-compose.yml
│   │   ├── config.json         # 模型配置
│   │   └── .env.example        # 环境变量模板
│   └── proto/                  # gRPC 协议定义
└── CY_LLM_Training/             # 训练相关 (可选)
```

## ⚙️ 配置

### 环境变量

```bash
# 核心配置（使用 CY_LLM_*）
CY_LLM_ENGINE=cuda-vllm      # 推理引擎（优先）
CY_LLM_PORT=8080             # Gateway 端口（优先）
VLLM_TP=1                    # 张量并行度
VLLM_GPU_MEM=0.9             # GPU 显存使用率


### 模型配置

编辑 `CY_LLM_Backend/deploy/config.json`:

```json
{
  "models": {
    "my-model": {
      "engine": "cuda-vllm",
    }
  }
}
```

## 🧪 测试

```bash
# 运行集成测试
./ew test integration

# 运行单元测试
./ew test unit

# 运行所有测试
./ew test all
```

## 📚 文档与设计

本仓库包含以下关键文档：
-- `CY_LLM_Backend/ARCHITECTURE.md` - 架构说明（Gateway / Coordinator / Worker）
-- `CY_LLM_Backend/DEPLOY.md` - 部署与 Docker Compose 说明
- `TESTING.md` - 测试说明（本地与 CI）
- `CONTRIBUTING.md` - 提交与版本管理规则（四段式版本号 + 后缀）

建议在提交前运行以下命令以确保本地环境一致：

```bash
# Worker 单元测试
cd CY_LLM_Backend/worker
pytest tests/ -q

# Gateway 单元测试（Gradle）
cd ../gateway
./gradlew test

# Coordinator 单元测试（Gradle）
cd ../coordinator
./gradlew test
```


## 🏷 版本号规范与提交格式

本项目采用 **四段式版本号** 并配合后缀来指示稳定性，例子： `[x.y.z.n-Alpha]`。

- 第一段（x）：重大、**不兼容**变化（破坏性重构）
- 第二段（y）：向后兼容的新功能（feature）
- 第三段（z）：Bug 修复与优化
- 第四段（n）：构建/测试次数（递增）

后缀说明：
- `PreAlpha`：功能不完整、仍处于设计早期
- `Alpha`：大部分功能可用，开始第一次测试
- `Beta`：功能实现完整，展开更广泛测试
- `RC` / `Release`：可用于生产或候选发布

提交消息与版本号格式（示例）：

`[2.1.1.2-Alpha] refactor(worker): API, async, telemetry, security and performance improvements`

请在提交中包含英文与中文说明（英文在前，空一行，随后中文），并将版本号用方括号完整包围在一行开头，标题与版本号同一行（版本号在前）。

更多贡献规范见 `CONTRIBUTING.md`。

## 📝 版本历史

- **[1.5.2.0]** - 简化部署流程，统一 CLI 工具，支持四种推理引擎
- **[1.5.1.3-alpha]** - C++ 入口点，四引擎架构实现
- **[1.0.0-alpha]** - 初始版本，Gateway + Worker 基础架构

## 🤝 贡献

欢迎提交 Issue 或 Pull Request。