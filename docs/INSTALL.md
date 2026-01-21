# 安装与运行指南

本指南详细说明 CY-LLM Engine 的安装、配置和运行方法。

## 环境要求

### 硬件要求

| 平台 | 要求 |
|------|------|
| NVIDIA GPU | CUDA 12.0+, 显存 >= 16GB |
| 华为 Ascend | CANN 8.0+, 显存 >= 32GB |
| 通用 | 内存 >= 32GB, 磁盘 >= 100GB SSD |

### 软件要求

| 组件 | 最低版本 | 推荐版本 |
|------|----------|----------|
| Python | 3.10+ | 3.11 |
| Java | 21+ | 21 (LTS) |
| CUDA | 12.0 | 12.4 |
| cuDNN | 8.9 | 9.0 |
| Redis | 7.0 | 7.2 |
| Docker | 24.0 | 25.0 |
| Docker Compose | 2.20 | 2.24 |

## 安装步骤

### 方式一：本地开发部署（推荐）

#### 1. 克隆仓库

```bash
git clone https://github.com/Baijin64/CY-LLM-Engine.git
cd CY-LLM-Engine
```

#### 2. 安装 Python 依赖

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# 或
.venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt

# 安装 CUDA 相关依赖 (如使用 NVIDIA GPU)
pip install -r requirements-nvidia.txt

# 安装 TensorRT-LLM 依赖 (如需使用 TRT 引擎)
pip install -r requirements-trt.txt

# 安装 vLLM 依赖
pip install -r requirements-vllm.txt
```

#### 3. 安装 Java 21 (用于 Gateway/Coordinator)

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y openjdk-21-jdk

# macOS (使用 Homebrew)
brew install openjdk@21

# 验证安装
java -version
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
```

#### 4. 安装 Redis

```bash
# Ubuntu/Debian
sudo apt-get install -y redis-server

# macOS
brew install redis

# 启动 Redis
redis-server --daemonize yes

# 验证
redis-cli ping
# 返回 PONG 表示成功
```

#### 5. 初始化环境

```bash
# 初始化 vLLM 环境 (NVIDIA GPU)
./cy-llm setup --engine cuda-vllm

# 或初始化 TensorRT-LLM 环境
./cy-llm setup --engine cuda-trt

# 或初始化 Ascend 环境
./cy-llm setup --engine ascend-vllm
```

### 方式二：Docker 部署（推荐生产环境）

#### 1. 配置环境变量

```bash
cd CY_LLM_Backend/deploy
cp .env.example .env
vim .env
```

配置示例：

```bash
# Gateway 配置
CY_LLM_PORT=8080
CY_LLM_INTERNAL_TOKEN=your-secure-token

# Coordinator 配置
CY_LLM_COORDINATOR_HOST=coordinator
CY_LLM_COORDINATOR_PORT=50050

# Worker 配置
CY_LLM_ENGINE=cuda-vllm
CY_LLM_MODEL_REGISTRY_PATH=/app/models/config.json

# Redis 配置
REDIS_HOST=redis
REDIS_PORT=6379
```

#### 2. 配置模型

编辑 `CY_LLM_Backend/deploy/config.json`：

```json
{
  "models": {
    "qwen2.5-7b": {
      "engine": "cuda-vllm",
      "model_path": "Qwen/Qwen2.5-7B-Instruct",
      "max_model_len": 8192,
      "gpu_memory_utilization": 0.9
    },
    "deepseek-v3": {
      "engine": "cuda-vllm",
      "model_path": "deepseek-ai/DeepSeek-V3",
      "adapter_path": "/checkpoints/my_lora",
      "max_model_len": 16384
    },
    "llama3-trt": {
      "engine": "cuda-trt",
      "model_path": "/models/llama3-8b-trt",
      "max_input_len": 4096,
      "max_output_len": 2048
    }
  }
}
```

#### 3. 启动服务

```bash
# 构建并启动所有服务
docker compose up -d --build

# 查看日志
docker compose logs -f

# 查看服务状态
docker compose ps
```

### 方式三：手动启动各组件

#### 1. 启动 Redis

```bash
docker run -d --name redis \
  -p 6379:6379 \
  -v redis_data:/data \
  redis:7-alpine \
  redis-server --appendonly yes
```

#### 2. 启动 Coordinator

```bash
cd CY_LLM_Backend/coordinator
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
./gradlew bootRun
```

#### 3. 启动 Worker

```bash
cd CY_LLM_Backend/worker
source .venv/bin/activate

# 方式一：直接启动
python -m worker.main --serve --port 50051

# 方式二：使用 CLI
./cy-llm worker --port 50051
```

#### 4. 启动 Gateway

```bash
cd CY_LLM_Backend/gateway
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
./gradlew bootRun
```

## 验证部署

### 健康检查

```bash
# Gateway 健康检查
curl http://localhost:8080/api/v1/health

# Coordinator 健康检查
curl http://localhost:50050/actuator/health

# Worker 健康检查
curl http://localhost:50051/health
```

### 推理测试

```bash
# 非流式推理
curl -X POST http://localhost:8080/api/v1/inference \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "modelId": "qwen2.5-7b",
    "prompt": "你好，请介绍一下自己"
  }'

# 流式推理 (SSE)
curl -N -X POST http://localhost:8080/api/v1/inference/stream \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "modelId": "qwen2.5-7b",
    "prompt": "请讲述一个关于人工智能的短故事"
  }'
```

### 训练测试

```bash
# 启动训练任务
curl -X POST http://localhost:8080/api/v1/training/start \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "baseModel": "Qwen/Qwen2.5-7B-Instruct",
    "outputDir": "/checkpoints/my_lora",
    "datasetPath": "/data/train.json",
    "hyperParams": {
      "batchSize": 4,
      "learningRate": 2e-4,
      "epochs": 3,
      "loraRank": 64,
      "loraAlpha": 16
    }
  }'

# 查询训练状态
curl http://localhost:8080/api/v1/training/status/{jobId}
```

## 常用命令

### CLI 命令

```bash
# 初始化环境
./cy-llm setup --engine cuda-vllm

# 启动服务
./cy-llm start --model qwen2.5-7b
./cy-llm start --engine cuda-trt --model llama3-trt

# 停止服务
./cy-llm stop

# 查看状态
./cy-llm status

# 查看日志
./cy-llm logs

# 运行测试
./cy-llm test unit
./cy-llm test integration

# 模型管理
./cy-llm models list
./cy-llm models add qwen2.5-7b Qwen/Qwen2.5-7B-Instruct

# 环境诊断
./cy-llm doctor
```

### Docker 命令

```bash
# 启动
docker compose up -d

# 停止
docker compose down

# 重启
docker compose restart

# 查看日志
docker compose logs -f gateway
docker compose logs -f coordinator
docker compose logs -f worker

# 扩展 Worker
docker compose up -d --scale worker=2

# 清理
docker compose down -v
docker system prune -a
```

## 配置说明

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CY_LLM_PORT` | 8080 | Gateway 端口 |
| `CY_LLM_ENGINE` | cuda-vllm | 默认推理引擎 |
| `CY_LLM_MODEL` | - | 默认模型 ID |
| `CY_LLM_COORDINATOR_HOST` | localhost | Coordinator 地址 |
| `CY_LLM_COORDINATOR_PORT` | 50050 | Coordinator 端口 |
| `CY_LLM_INTERNAL_TOKEN` | - | 内部通信 Token |
| `REDIS_HOST` | localhost | Redis 地址 |
| `REDIS_PORT` | 6379 | Redis 端口 |
| `VLLM_TP` | 1 | 张量并行度 |
| `VLLM_GPU_MEM` | 0.9 | GPU 显存使用率 |

### 模型配置

```json
{
  "models": {
    "<model-id>": {
      "engine": "cuda-vllm",
      "model_path": "<huggingface-path or local-path>",
      "adapter_path": "<lora-adapter-path>",
      "max_model_len": 8192,
      "gpu_memory_utilization": 0.9,
      "use_4bit": false,
      "quantization": null
    }
  }
}
```

## 故障排除

### 问题 1: GPU 显存不足

**解决方案**：
```bash
# 降低显存使用率
./cy-llm start --model qwen2.5-7b --gpu-mem 0.5

# 或使用量化
./cy-llm start --model qwen2.5-7b --quantization awq
```

### 问题 2: 端口被占用

```bash
# 查看占用端口的进程
lsof -i :8080

# 杀掉进程
kill <PID>

# 或使用其他端口
./cy-llm start --port 8081
```

### 问题 3: gRPC 连接失败

```bash
# 检查 Coordinator 是否启动
curl http://localhost:50050/actuator/health

# 检查 Worker 是否启动
curl http://localhost:50051/health

# 查看日志
./cy-llm logs worker
```

### 问题 4: 依赖冲突

```bash
# 创建新的 Conda 环境
conda create -n cy-llm python=3.11
conda activate cy-llm
pip install -r requirements.txt

# 或使用 Docker 部署
docker compose up -d
```

### 问题 5: 模型加载失败

```bash
# 手动下载模型
huggingface-cli download Qwen/Qwen2.5-7B-Instruct

# 或设置镜像源
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen2.5-7B-Instruct
```

## 性能调优

### GPU 配置

```bash
# 设置 GPU 显存使用率
export VLLM_GPU_MEMORY_UTILIZATION=0.85

# 设置张量并行度 (多 GPU)
export VLLM_TENSOR_PARALLEL_SIZE=2

# 设置最大序列长度
export VLLM_MAX_MODEL_LEN=16384
```

### 批处理配置

```json
{
  "models": {
    "qwen2.5-7b": {
      "engine": "cuda-vllm",
      "model_path": "Qwen/Qwen2.5-7B-Instruct",
      "max_num_batched_tokens": 65536,
      "max_num_seqs": 256,
      "gpu_memory_utilization": 0.9
    }
  }
}
```

## 卸载

### 本地部署卸载

```bash
# 停止服务
./cy-llm stop

# 删除虚拟环境
rm -rf .venv

# 删除缓存数据
rm -rf ~/.cache/huggingface
rm -rf ~/.cache/tensorrt_llm
```

### Docker 卸载

```bash
# 停止并删除容器
docker compose down -v

# 删除镜像
docker rmi cy-llm-gateway cy-llm-coordinator cy-llm-worker

# 删除数据卷
docker volume prune -f
```
