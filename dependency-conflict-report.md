# CY-LLM Engine 依赖冲突分析报告

**生成日期**: 2026-02-10
**环境**: cy-llm-refactor (Python 3.11.14)
**分析范围**: 所有 requirements 文件

---

## 1. 已知冲突

### 1.1 Protobuf 版本冲突 (严重)

**位置**: `requirements-vllm.txt`
```protobuf==6.33.4```

**问题**:
- vLLM 0.12.0 要求 `protobuf<6.0.0`
- 当前文件指定 `protobuf==6.33.4` (6.x 版本)
- 这是版本不兼容

**影响**:
- gRPC 通信可能失败
- vLLM 推理服务无法正常启动

**建议解决方案**:
```protobuf==5.28.3  # 兼容 vLLM 0.12.0```

---

### 1.2 CUDA 版本不匹配 (严重)

**问题描述**:
| 文件 | CUDA 版本 | PyTorch 版本 |
|------|-----------|--------------|
| `requirements-nvidia.txt` | cu118 (11.8) | torch>=2.1.0 |
| `requirements-trt.txt` | cu121 (12.1) | torch==2.4.0 |
| `requirements-vllm.txt` | cu124 (12.4) | torch==2.9.0 |

**影响**:
- 无法在同一环境中同时使用多个引擎
- 依赖冲突导致安装失败

**建议**:
- 创建独立的 conda 环境用于不同的推理引擎
- 或者使用 CPU-only 模式进行开发

---

### 1.3 NumPy 版本冲突

**冲突点**:
- `requirements-base.txt`: `numpy>=1.24.0,<2.0.0`
- `requirements-vllm.txt`: `numpy==1.26.4`

**分析**:
- vLLM 强制使用 1.26.4
- base 允许 1.24.x - 1.26.x
- **可以接受**，但版本过旧

**建议**:
- 考虑升级到 numpy 2.x（需要全面测试）

---

## 2. 版本范围过宽的风险

### 2.1 grpcio 范围过大

```grpcio>=1.60.0,<2.0.0```

**问题**:
- 跨次要版本升级可能引入不兼容变更
- 与 vLLM 强制的 `grpcio==1.76.0` 冲突

**建议**:
```grpcio==1.76.0  # 固定版本以确保兼容性```

---

### 2.2 transformers 版本

```transformers>=4.36.0,<5.0.0```

**风险**:
- 4.36.0 到 4.45.x 之间的变更可能导致 API 变化
- 与 TRL/PEFT 版本可能不兼容

**建议**:
- 固定到已测试的版本，如 `==4.42.4`

---

## 3. 建议的修复方案

### 3.1 最小改动方案 (保守)

修改 `requirements-vllm.txt`:

```diff
- protobuf==6.33.4
+ protobuf==5.28.3
```

### 3.2 推荐方案 (全面)

创建统一的依赖配置:

```requirements-cpu.txt
-r requirements-base.txt

# CPU-only PyTorch (无 CUDA)
torch==2.4.0
torchvision==0.19.0
torchaudio==2.9.0

# 统一版本
grpcio==1.76.0
protobuf==5.28.3
transformers==4.42.4
accelerate==0.32.0
bitsandbytes==0.48.2
peft==0.11.1
trl==0.9.6
```

---

## 4. 环境隔离建议

对于当前无 GPU 的开发环境，建议:

1. **CPU-only 开发环境**: 使用通用依赖
2. **NVIDIA GPU 环境**: 使用 `requirements-nvidia.txt`
3. **vLLM 生产环境**: 使用 `requirements-vllm.txt` (修复 protobuf)
4. **TensorRT 环境**: 使用 `requirements-trt.txt`

---

## 5. 检测方法

使用以下命令验证冲突:

```bash
# 创建临时环境测试
conda create -n test-deps python=3.11 -y
conda activate test-deps
pip install -r requirements-vllm.txt  # 应该失败

# 详细冲突分析
pip install pip-tools
pip-compile --generate-hashes requirements-base.txt
```

---

## 6. 行动计划

| 优先级 | 问题 | 修复方案 | 预计时间 |
|--------|------|----------|----------|
| P0 | protobuf 版本冲突 | 修改为 5.28.3 | 5分钟 |
| P1 | CUDA 版本混乱 | 创建统一配置 | 30分钟 |
| P2 | grpcio 版本范围过大 | 固定版本 | 10分钟 |
| P3 | numpy 版本过旧 | 评估升级到 2.x | 2小时 |

---

**报告生成工具**: cy-llm-refactor 环境
**生成者**: Environment & DevOps Setup Engineer
