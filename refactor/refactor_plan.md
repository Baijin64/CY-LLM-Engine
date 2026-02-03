# 重构任务计划

**项目**: CY-LLM-Engine | **生成时间**: 2026-02-02
**总任务**: 6 项 | **预计耗时**: 14 小时

---

## 📋 任务概览

| 优先级 | 任务数 | 预计耗时 | 说明 |
|--------|--------|----------|------|
| **P0** | 2 | 1h | 必须修复（清理与基础设施） |
| **P1** | 2 | 6h | 架构升级（通信与结构） |
| **P2** | 2 | 7h | 质量保证（类型与测试） |

---

## 🚨 P0 任务（Critical）

### TASK-C001: 清理企业版遗留物
**类型**: Cleanup
**优先级**: P0
**预计耗时**: 0.5h

**问题描述**:
- 存在 Kotlin/Gradle 构建文件 (`settings.gradle.kts`, `gradle.properties`)。
- 遗留的兼容性入口 `CY_LLM_Backend/main.py`。
- 目录结构混合了旧版 Enterprise 代码。

**执行步骤**:
1. 删除 `CY_LLM_Backend/settings.gradle.kts`, `CY_LLM_Backend/gradle.properties`。
2. 删除 `CY_LLM_Backend/main.py`。
3. 检查并移除 `CY_LLM_Backend/deploy/grafana` 中可能存在的敏感企业配置。

**验收标准**:
- [ ] 根目录无 Gradle 相关文件。
- [ ] 纯 Python/Rust 环境。

**负责人**: @maintainer

---

### TASK-C002: 初始化 Rust Sidecar
**类型**: Scaffold
**优先级**: P0
**预计耗时**: 0.5h

**问题描述**:
- 缺失核心 Rust 组件 (`rust_core`)，导致架构评分低。
- 需要高性能 Sidecar 处理网格通信。

**执行步骤**:
1. 在根目录创建 Rust crate: `cargo new rust_core --lib`。
2. 配置 `Cargo.toml` (添加 `pyo3` 依赖以便 Python 调用)。
3. 创建基础 FFI 绑定入口。

**验收标准**:
- [ ] `rust_core` 目录存在。
- [ ] `cargo build` 成功。
- [ ] Python 可通过 maturin 或 setuptools-rust 引用。

**负责人**: @rust_engineer

---

## ⚠️ P1 任务（High）

### TASK-A001: 标准化 Python 项目结构
**类型**: Refactor
**优先级**: P1
**预计耗时**: 2h

**问题描述**:
- 当前代码散落在 `CY_LLM_Backend`，不符合 Python Packaging 标准。
- 缺乏统一的 `pyproject.toml` 管理。

**执行步骤**:
1. 创建 `src/cy_llm` 目录。
2. 将 `CY_LLM_Backend/worker`, `coordinator_lite`, `gateway_lite` 迁移至 `src/cy_llm` 下的子模块。
3. 创建根目录 `pyproject.toml` 替代分散的 `requirements.txt`。

**验收标准**:
- [ ] 遵循 `src/` 布局。
- [ ] 可通过 `pip install -e .` 安装。

**负责人**: @python_lead

---

### TASK-A002: Worker 通信架构迁移 (TCP -> UDS)
**类型**: Architecture
**优先级**: P1
**预计耗时**: 4h

**问题描述**:
- Worker 目前使用 gRPC over TCP，本地通信效率非最优。
- 目标架构要求使用 UDS (Unix Domain Sockets) 提升 Sidecar 与 Worker 间的吞吐量。
- *注：Windows 环境将回退到 Named Pipes 或 Local TCP Loopback。*

**执行步骤**:
1. 修改 `worker/grpc_servicer.py` 支持 UDS 监听 (`unix:///tmp/cy_worker.sock`)。
2. 更新 Coordinator/Sidecar 的连接逻辑以支持 UDS URI。
3. 编写通信适配层，自动判断 OS 选择 UDS 或 TCP。

**验收标准**:
- [ ] Linux 下使用 `.sock` 文件通信。
- [ ] 延迟降低 10%+。
- [ ] 兼容 Windows 开发环境。

**负责人**: @backend_architect

---

## 💡 P2 任务（Medium）

### TASK-Q001: 添加静态类型检查
**类型**: Quality
**优先级**: P2
**预计耗时**: 3h

**问题描述**:
- `worker/core` 等核心模块缺乏类型提示。
- `mypy` 检查目前未启用。

**执行步骤**:
1. 配置 `mypy.ini` (strict mode)。
2. 为 `worker/core/*.py` 添加 Type Hints。
3. 修复现有的类型错误。

**验收标准**:
- [ ] `mypy` 检查通过。
- [ ] 核心模块覆盖率 100%。

---

### TASK-Q002: 单元测试覆盖
**类型**: Quality
**优先级**: P2
**预计耗时**: 4h

**问题描述**:
- 核心业务逻辑测试覆盖率低。
- 缺乏针对 Sidecar 交互的测试。

**执行步骤**:
1. 配置 `pytest`。
2. 为 `worker/core/task_scheduler.py` 和 `memory_manager.py` 添加单元测试。
3. 模拟 Rust Sidecar 接口进行 Mock 测试。

**验收标准**:
- [ ] 测试覆盖率 > 80% (核心模块)。
- [ ] CI 自动运行测试。

---

## 📅 执行时间线

### 第 1 阶段 (Cleanup & Scaffold)
- [ ] TASK-C001 (0.5h) - 清理
- [ ] TASK-C002 (0.5h) - Rust 初始化

**里程碑**: 干净的混合语言项目结构

### 第 2 阶段 (Architecture)
- [ ] TASK-A001 (2h) - 目录重构
- [ ] TASK-A002 (4h) - UDS 迁移

**里程碑**: 符合开源架构标准

### 第 3 阶段 (Quality)
- [ ] TASK-Q001 (3h) - 类型增强
- [ ] TASK-Q002 (4h) - 测试补全

**里程碑**: 生产级代码质量

---

## ⚙️ 自动化执行

回复 "execute" 将自动开始执行 P0 任务。
