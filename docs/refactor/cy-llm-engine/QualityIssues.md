# CY-LLM Engine Refactor - Quality Issues Log

## 问题追踪

### 已发现问题

#### QI-001: Import路径错误
- **文件**: `src/cy_llm/worker/tests/test_abstract_engine.py`
- **问题**: Import "worker.engines.abstract_engine" could not be resolved
- **原因**: 测试文件使用了错误的相对导入路径
- **建议修复**: 改为 `from cy_llm.worker.engines import abstract_engine`
- **状态**: 🔵 待修复（Phase 1 目录合并时处理）
- **优先级**: P1

#### QI-002: Protobuf版本冲突 ✅ FIXED
- **文件**: `requirements-vllm.txt` vs `requirements-base.txt`
- **问题**: protobuf==6.33.4 与 protobuf<6.0.0 冲突
- **影响**: vLLM无法正常运行
- **修复**: 统一使用 protobuf==4.25.3
- **状态**: ✅ 已修复（2026-02-10）
- **验证**: `grep "protobuf" requirements-vllm.txt` → "protobuf==4.25.3"
- **优先级**: P0

#### QI-003: CUDA版本不匹配 ✅ FIXED
- **文件**: `requirements-nvidia.txt` (cu118) vs `requirements-vllm.txt` (cu124)
- **问题**: PyTorch CUDA版本不一致
- **影响**: 运行时CUDA库错误（libcudart.so.11.0 not found）
- **修复**: 统一使用 cu124，torch==2.9.0
- **状态**: ✅ 已修复（2026-02-10）
- **验证**: `grep "cu124" requirements-nvidia.txt` → "cu124"
- **优先级**: P0

#### QI-004: 目录重复 ✅ FIXED
- **问题**: `src/cy_llm/` 和 `CY_LLM_Backend/` 高度重复
- **影响**: 维护困难，代码不一致风险
- **修复**: 删除 src/cy_llm/，保留 CY_LLM_Backend/
- **状态**: ✅ 已修复（2026-02-10）
- **验证**: `ls src/cy_llm` → "No such file or directory"
- **优先级**: P0

#### QI-005: 模型推理异常
- **问题**: Token生成速度过快（6187 tokens/s）、内容重复
- **原因**: 可能是参数配置问题或模型未正确加载
- **建议修复**: 检查sampling参数，添加健康检查
- **状态**: 🔵 待修复（Phase 3 引擎重构）
- **优先级**: P1

## 修复状态统计

| 状态 | 数量 |
|------|------|
| 🔴 Blocker | 0 |
| 🟡 Critical | 2 |
| 🔵 Pending | 3 |
| 🔄 In Progress | 1 |
| ✅ Fixed | 0 |

## 质量门禁

进入下一阶段前必须解决:
- [ ] QI-002 (P0 - protobuf冲突)
- [ ] QI-003 (P0 - CUDA版本)
- [ ] QI-004 (P0 - 目录合并)
