# CY-LLM Engine 项目基线快照

**生成日期**: 2026-02-10
**环境**: cy-llm-refactor (Python 3.11.14)
**用途**: 重构项目的基线参考

---

## 项目概览

```
CY-LLM-Engine/
├── 主要组件: 3 个核心服务
│   ├── CY_LLM_Backend (后端服务)
│   ├── gateway (API 网关)
│   └── rust_core (Rust 核心模块)
├── 训练模块: CY_LLM_Training
├── 测试目录: tests/
├── 配置目录: docs/
└── 脚本工具: scripts/
```

---

## 目录结构详情

### 1. CY_LLM_Backend/ (主后端服务)

```
CY_LLM_Backend/
├── coordinator_lite/     # 轻量级协调器
│   ├── app/
│   ├── Dockerfile.coordinator.lite
│   ├── __init__.py
│   └── requirements.txt
├── deploy/               # 部署配置
│   ├── grafana/          # Grafana 监控面板
│   ├── prometheus/       # Prometheus 指标收集
│   ├── Dockerfiles
│   ├── docker-compose.yml
│   └── config.json
├── gateway_lite/         # 轻量级网关
│   ├── app/
│   ├── Dockerfile.gateway.lite
│   └── requirements.txt
├── proto/                # gRPC 协议定义
│   └── ai_service.proto
├── scripts/              # 构建脚本
│   └── gen_proto.sh
├── tests/               # 集成测试
├── training/            # 训练引擎
│   ├── model/
│   ├── __init__.py
│   └── engine.py
├── worker/              # 工作节点 (核心)
│   ├── cache/
│   ├── config/
│   ├── core/
│   ├── engines/         # 推理引擎
│   │   ├── abstract_engine.py
│   │   ├── nvidia_engine.py
│   │   ├── vllm_engine.py
│   │   └── trt_engine.py
│   ├── health/
│   ├── proto_gen/
│   ├── training/       # 训练相关
│   ├── utils/
│   ├── Dockerfiles
│   ├── main.py         # 入口点
│   ├── requirements.txt
│   └── training_engine.py
└── 文档
    ├── ARCHITECTURE.md
    └── DEPLOY.md
```

**关键文件**:
- `worker/REFACTORING.py` - 重构计划
- `worker/grpc_servicer.py` - gRPC 服务实现
- `worker/training_engine.py` - 训练引擎

---

### 2. gateway/ (API 网关)

```
gateway/
├── gateway_lite/        # 轻量级网关实现
│   ├── app/
│   ├── Dockerfile.gateway.lite
│   └── requirements.txt
├── INTERFACE_CONTRACT.md
├── README.md
└── pyproject.toml      # PEP 518 配置
```

---

### 3. rust_core/ (Rust 核心模块)

```
rust_core/
├── src/
│   ├── bin/            # 二进制入口
│   ├── generated/      # 自动生成代码
│   ├── config.rs      # 配置管理
│   ├── errors.rs      # 错误处理
│   ├── health.rs      # 健康检查
│   ├── lib.rs         # 库主文件
│   ├── metering.rs   # 计量统计
│   ├── metrics.rs     # 指标收集
│   └── proxy.rs       # 代理逻辑
├── tests/
│   └── integration_test.rs
├── Cargo.toml         # Cargo 配置文件
├── Cargo.lock         # 依赖锁定
└── 文档
    ├── ARCHITECTURE.md
    ├── README.md
    └── SECURITY.md
```

**关键特性**:
- 独立的 Rust 二进制
- 完整的测试覆盖
- gRPC + REST 双协议支持

---

### 4. CY_LLM_Training/ (训练模块)

```
CY_LLM_Training/
├── dialogue_gi/       # 对话生成
│   ├── speaker/
│   └── README.md
└── src/
    ├── dataset_converter.py
    ├── inference.py
    └── train_lora.py
```

---

### 5. tests/ (测试套件)

```
tests/
├── integration/       # 集成测试
│   └── test_grpc_uds.py
├── unit/              # 单元测试
│   ├── test_memory_manager.py
│   └── test_task_scheduler.py
├── TEST_COVERAGE.md   # 覆盖率报告
└── __init__.py
```

---

### 6. docs/ (文档)

```
docs/
├── HISTORY/
│   ├── MIGRATION_SUMMARY.md
│   └── PHASE2_3_UPGRADE_REPORT.md
├── REFACTOR/
│   └── CY-LLM-Engine/
├── refactor/
│   └── cy-llm-engine/
├── API.md
├── ARCHITECTURE.md
├── CONTRIBUTING.md
├── FAQ.md
├── INSTALL.md
├── README.md
├── TESTING.md
└── TRT_GUIDE.md
```

---

### 7. scripts/ (工具脚本)

```
scripts/
├── benchmark.sh        # 性能测试
├── check-ci-refs.sh    # CI 检查
├── clean.sh            # 清理脚本
├── convert_trt.py       # TensorRT 转换
├── diagnose_env.py     # 环境诊断
├── e2e_lite.sh         # 端到端测试
└── verify-deploy.sh    # 部署验证
```

---

## 文件统计

| 类别 | 数量 |
|------|------|
| 目录 | 56 |
| 文件 | 117 |
| Python 文件 | ~80+ |
| 配置文件 | 15+ |
| 文档 | 20+ |

---

## 依赖配置

### 现有 requirements 文件

| 文件 | 用途 | Python 版本 |
|------|------|-------------|
| `requirements-base.txt` | 基础依赖 | 未指定 |
| `requirements-nvidia.txt` | NVIDIA GPU | torch 2.1-2.4 |
| `requirements-vllm.txt` | vLLM 推理 | torch 2.9 |
| `requirements-trt.txt` | TensorRT | torch 2.4 |

### 配置文件

- `pyproject.toml` - PEP 517/518 配置
- `pytest.ini` - pytest 配置
- `mypy.ini` - mypy 类型检查配置

---

## 技术栈概览

### 后端服务
- **语言**: Python 3.10+
- **框架**: gRPC + FastAPI
- **引擎**: Transformers, vLLM, TensorRT
- **部署**: Docker + Docker Compose

### 核心模块
- **语言**: Rust 1.70+
- **框架**: Tokio async runtime
- **构建**: Cargo

### 监控
- **指标**: Prometheus
- **面板**: Grafana
- **日志**: 结构化日志

---

## 关键配置路径

```
项目根目录: /home/baijin/Dev/CY-LLM-Engine
开发环境: /home/baijin/miniforge3/envs/cy-llm-refactor
Worker 配置: CY_LLM_Backend/worker/config/
网关配置: gateway/gateway_lite/app/
Rust 配置: rust_core/sidecar.toml
```

---

## 快照元数据

- **快照 ID**: CY-LLM-Engine-baseline-20260210
- **Python 环境**: cy-llm-refactor (3.11.14)
- **Conda 版本**: 25.9.1
- **开发工具**: pytest, black, ruff, mypy
- **依赖冲突**: 3 个已知问题 (见 dependency-conflict-report.md)

---

**生成工具**: Environment & DevOps Setup Engineer
**生成时间**: 2026-02-10 10:00:00 UTC+8
