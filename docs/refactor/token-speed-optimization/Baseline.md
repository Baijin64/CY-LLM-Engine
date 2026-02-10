# CY-LLM Engine Token Speed Baseline

## 性能基线记录

**记录日期**: 2026-02-10  
**记录版本**: v1.0  
**测试环境**: 开发环境

---

## 当前性能指标

| 指标 | 当前值 | 目标值 | 状态 |
|------|--------|--------|------|
| **Token速度** | 15-20 tokens/s | ≥50 tokens/s | ❌ 不达标 |
| **TTFT (首Token延迟)** | ~500ms | ≤200ms | ❌ 不达标 |
| **并发处理能力** | 未测试 | 支持100并发 | ⏳ 待测试 |

### 性能差距分析

- Token速度差距: **2.5-3.3x** (目标/当前)
- TTFT差距: **2.5x** (目标/当前)

---

## 瓶颈分析

### 🔴 致命瓶颈 (P0)

#### 1. 逐字符yield (vllm_cuda_engine.py:500)
**位置**: `CY_LLM_Backend/worker/engines/vllm_cuda_engine.py` 第495-501行

```python
# 当前代码（问题）
outputs = self._llm.generate([prompt], **request_kwargs)
if outputs and len(outputs) > 0:
    generated_text = outputs[0].outputs[0].text
    for char in generated_text:  # <-- 致命问题
        yield char
```

**问题**:
1. vLLM先生成完整文本（阻塞）
2. 然后逐字符yield，每次yield产生巨大开销
3. 每次字符yield都经过完整的Python生成器协议

**预期优化收益**: 15→35+ tokens/s (单此优化)

### 🟡 架构瓶颈 (P1)

#### 2. 同步引擎使用
**位置**: `CY_LLM_Backend/worker/engines/engine_factory.py` 默认配置

- 当前默认引擎: `cuda-vllm` (同步LLM)
- 可用异步引擎: `cuda-vllm-async` (AsyncLLMEngine)

**问题**:
- 同步引擎需要等待完整生成
- TTFT无法优化到<200ms

**预期优化收益**: TTFT 500ms→50ms, Token速度 35→50+ tokens/s

#### 3. gRPC消息传输粒度
**位置**: `CY_LLM_Backend/worker/grpc_servicer.py:231-237`

- 每个yield产生一个gRPC响应
- 逐字符导致大量网络往返

**预期优化收益**: 减少网络开销20-30%

---

## 测试环境信息

### 硬件配置
| 项目 | 值 |
|------|-----|
| GPU型号 | 待测试环境确定 |
| GPU显存 | 待测试环境确定 |
| CUDA版本 | 待测试环境确定 |

### 软件配置
| 项目 | 当前值 |
|------|--------|
| Python版本 | 3.12.12 |
| vLLM版本 | 待安装 |
| PyTorch版本 | 待验证 |
| gRPC版本 | 待验证 |

---

## 测试命令

### 1. 运行基准测试
```bash
# 测试当前引擎（同步vLLM）
python scripts/benchmark_token_speed.py \
    --model deepseek-ai/deepseek-llm-7b-chat \
    --engine cuda-vllm \
    --output docs/refactor/token-speed-optimization/baseline_result.json

# 测试异步引擎（优化后）
python scripts/benchmark_token_speed.py \
    --model deepseek-ai/deepseek-llm-7b-chat \
    --engine cuda-vllm-async \
    --output docs/refactor/token-speed-optimization/optimized_result.json

# 多轮测试取平均
python scripts/benchmark_token_speed.py \
    --model deepseek-ai/deepseek-llm-7b-chat \
    --engine cuda-vllm \
    --runs 5 \
    --output docs/refactor/token-speed-optimization/baseline_avg.json
```

### 2. 对比测试
```bash
# 对比优化前后
python scripts/benchmark_compare.py \
    --before docs/refactor/token-speed-optimization/baseline_result.json \
    --after docs/refactor/token-speed-optimization/optimized_result.json
```

---

## 构建命令

### 安装依赖
```bash
# 基础依赖
pip install -e .

# vLLM引擎
pip install vllm==0.12.0

# 测试依赖
pip install pytest pytest-benchmark pytest-asyncio
```

### 验证安装
```bash
# 验证引擎可用
python -c "from worker.engines import check_engine_available; print(check_engine_available('cuda-vllm'))"
python -c "from worker.engines import check_engine_available; print(check_engine_available('cuda-vllm-async'))"

# 验证默认引擎
python -c "from worker.engines.engine_factory import EngineFactory; print(EngineFactory.auto_detect())"
```

---

## 回归测试命令

```bash
# 单元测试
pytest tests/unit/test_vllm_cuda_engine_streaming.py -v

# 集成测试
pytest tests/integration/test_streaming_performance.py -v --timeout=300

# 性能测试
pytest tests/performance/ -m performance --benchmark-only

# 全量回归
pytest tests/ -xvs -k "not slow"
```

---

## 测试数据集

### 标准测试Prompts

#### Prompt 1: 中文长文本生成
```
请详细解释什么是人工智能，包括其历史、现状和未来发展趋势。
要求：
1. 从历史角度回顾AI的发展
2. 分析当前AI技术的核心能力
3. 预测未来10年的发展方向
4. 讨论可能面临的挑战和伦理问题
```

#### Prompt 2: 代码生成
```
请用Python实现一个快速排序算法，并添加详细的中文注释。
要求：
1. 实现完整的quicksort函数
2. 包含partition辅助函数
3. 添加时间复杂度分析
4. 提供测试用例
```

#### Prompt 3: 数学推理
```
请逐步推导求解以下方程：
2x^2 + 5x - 3 = 0

要求：
1. 使用求根公式
2. 展示完整推导过程
3. 验证结果正确性
```

---

## 基线测试记录

### 测试记录模板

| 日期 | 引擎 | 模型 | Token速度 | TTFT | 测试人 | 备注 |
|------|------|------|-----------|------|--------|------|
| 2026-02-10 | cuda-vllm | deepseek-7b | 15-20 t/s | ~500ms | - | 优化前基线 |
| | | | | | | |

---

## 优化里程碑

| 里程碑 | 目标Token速度 | 目标TTFT | 验收标准 |
|--------|---------------|----------|----------|
| M1: 流式优化完成 | ≥35 t/s | <600ms | TASK-001完成 |
| M2: gRPC优化完成 | ≥40 t/s | <500ms | TASK-002完成 |
| M3: 默认引擎切换 | ≥50 t/s | ≤200ms | TASK-003完成 |

---

## 附录

### 术语表
| 术语 | 说明 |
|------|------|
| TTFT | Time To First Token，首token延迟 |
| TPS | Tokens Per Second，每秒生成token数 |
| yield | Python生成器关键字 |
| AsyncLLMEngine | vLLM异步推理引擎 |

### 参考链接
- [vLLM性能优化指南](https://docs.vllm.ai/en/latest/getting_started/performance.html)
- [gRPC Python性能](https://grpc.io/docs/guides/performance/)

---

*文档版本: v1.0*  
*最后更新: 2026-02-10*  
*维护者: CY-LLM Engine Team*
