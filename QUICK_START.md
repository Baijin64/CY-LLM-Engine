# Phase 2 & 3 快速开始指南

## 🚀 快速使用

### 1. 使用 VRAM 预估器（自动）

模型加载时会自动运行 VRAM 预检查：

```python
from CY_LLM_Backend.worker.engines.vllm_cuda_engine import VllmCudaEngine

engine = VllmCudaEngine(
    max_model_len=2048,
    gpu_memory_utilization=0.90,
)

# 自动进行 VRAM 预检查，如果不安全会自动降低 gpu_memory_utilization
engine.load_model("Qwen/Qwen2.5-7B-Instruct")
```

**日志输出**:
```
INFO: VRAM 估算: ✅ 显存充足，可以加载
```

或：
```
WARNING: 自动调整 gpu_memory_utilization: 0.90 -> 0.65
```

### 2. 手动估算显存（可选）

```python
from CY_LLM_Backend.worker.utils.vram_optimizer import estimate_vram_requirements

estimate = estimate_vram_requirements(
    model_name_or_params="Qwen/Qwen2.5-7B-Instruct",
    max_model_len=2048,
    dtype="fp16",
    quantization=None,
    engine_type="vllm"
)

print(f"需要显存: {estimate.required_gb:.2f}GB")
print(f"可用显存: {estimate.available_gb:.2f}GB")
print(f"建议: {estimate.recommendation}")
```

### 3. 转换模型为 TRT 引擎

```bash
# 查看帮助
./cy convert-trt --help  # or ./cy-llm convert-trt --help

# 转换模型
./cy convert-trt \
  --model Qwen/Qwen2.5-7B-Instruct \
  --output /models/qwen2.5-7b-trt

# 使用自定义参数
./cy convert-trt \
  --model Qwen/Qwen2.5-7B-Instruct \
  --output /models/qwen2.5-7b-trt \
  --dtype float16 \
  --max-batch-size 64 \
  --max-input-len 4096 \
  --max-output-len 2048
```

### 4. 启动 TRT 服务

```bash
# 初始化环境
./cy setup --engine cuda-trt

# 启动服务
./cy start --engine cuda-trt --model qwen2.5-7b-trt

# 查看状态
./cy status
```

## 📊 新增特性对照表

| 特性 | 位置 | 触发方式 | 用户需要做什么 |
|------|------|--------|---------------|
| **VRAM 预检查** | vllm_cuda_engine.py | 自动 | 无，自动运行 |
| **VRAM 预估** | vram_optimizer.py | 手动或自动 | 可选手动调用 |
| **OOM 自动重试** | server.py | 自动 | 无，自动处理 |
| **TRT 真流式** | trt_engine.py | 自动 | 无，自动处理 |
| **TRT 转换工具** | scripts/convert_trt.py | 手动 | `./cy convert-trt ...` |
| **TRT 使用文档** | docs/TRT_GUIDE.md | 参考 | 查看文档 |

## 🔍 关键文件位置

### 代码文件

```
CY_LLM_Backend/worker/
├── utils/
│   └── vram_optimizer.py          # VRAM 预估和优化
├── engines/
│   ├── vllm_cuda_engine.py        # vLLM 引擎（已集成 VRAM 预检查）
│   └── trt_engine.py              # TRT 引擎（已改进流式输出）
└── core/
    └── server.py                  # 推理服务器（已集成 OOM 重试）

scripts/
└── convert_trt.py                 # TRT 模型转换工具
```

### 文档文件

```
docs/
└── TRT_GUIDE.md                   # TensorRT-LLM 完整使用指南

PHASE2_3_UPGRADE_REPORT.md         # 升级详细报告
```

### 脚本

```
cy / cy-llm                         # 主脚本
```

## ⚡ 常见命令速查

```bash
  # 诊断环境
  ./cy doctor

# 初始化 vLLM 环境
./cy setup --engine cuda-vllm

# 初始化 TRT 环境
./cy setup --engine cuda-trt

# 转换模型为 TRT
./cy convert-trt --model <model> --output <dir>

# 启动 vLLM 服务
./cy start --engine cuda-vllm --model <model>

# 启动 TRT 服务
./cy start --engine cuda-trt --model <model>

# 停止服务
./cy stop

# 查看状态
./cy status

# 帮助信息
./cy help
```

## 🐛 故障排除

### 问题 1: VRAM 预检查报告 "显存不足"

**解决**:
```python
# 方案 A: 跳过检查
engine.load_model(model_path, skip_vram_check=True)

# 方案 B: 手动调整配置
engine.gpu_memory_utilization = 0.65
engine.load_model(model_path)

# 方案 C: 使用量化
estimate = estimate_vram_requirements(
    "Qwen/Qwen2.5-7B",
    quantization="awq"  # 改为 4-bit 量化
)
```

### 问题 2: 模型加载 OOM

**自动处理**: 系统会自动重试 3 次，如果仍失败，查看日志：

```bash
tail -f logs/worker.log
```

**手动调整**:
```python
# 降低显存利用率
engine.gpu_memory_utilization = 0.50
engine.max_model_len = 2048
engine.load_model(model_path)
```

### 问题 3: TRT 转换失败

**检查**:
```bash
# 1. 验证依赖安装
python -c "import tensorrt_llm; print(tensorrt_llm.__version__)"

# 2. 检查模型格式
ls -la /path/to/model/

# 3. 查看详细错误
python scripts/convert_trt.py --model ... --output ... --verbose
```

## 📈 性能监控

```bash
# 实时显存使用
watch -n 1 nvidia-smi

# 查看日志
tail -f logs/worker.log

# 统计推理延迟
# 通过 API 响应头 X-Response-Time 查看
curl -i http://localhost:8080/api/v1/health
```

## 💡 最佳实践

1. **总是运行 doctor 命令**
   ```bash
  ./cy doctor
   ```

2. **转换 TRT 后测试**
   ```bash
   # 小批量测试
   curl -X POST http://localhost:8080/api/v1/inference \
     -d '{"modelId":"...","prompt":"test"}'
   ```

3. **为常用模型预编译 TRT**
   ```bash
   # 提前转换，避免首次启动慢
  ./cy convert-trt --model Qwen/Qwen2.5-7B --output /models/qwen2.5-7b-trt
   ```

4. **定期监控显存**
   ```bash
   # 长期运行时监控
   watch -n 5 'nvidia-smi | grep python'
   ```

## 📚 更多信息

- 详细升级报告: `PHASE2_3_UPGRADE_REPORT.md`
- TRT 完整指南: `docs/TRT_GUIDE.md`
- 源代码: `CY_LLM_Backend/worker/utils/vram_optimizer.py`
- 转换工具: `scripts/convert_trt.py`

---

**版本**: v3.5.0  
**完成时间**: 2025-12-03  
**状态**: ✅ 生产就绪
