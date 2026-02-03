# 项目目录结构说明

## 📁 当前目录结构（v3.0 Unified Architecture）

```
CY-LLM-Engine/
├── gateway/                    # 🔌 可替换的 API 接入层
│   ├── gateway_lite/           # Python FastAPI 网关（开源版）
│   ├── README.md               # Gateway 架构说明
│   ├── INTERFACE_CONTRACT.md   # gRPC 接口契约（企业版对接）
│   └── pyproject.toml          # 独立依赖管理
│
├── src/cy_llm/                 # 🧠 核心引擎（统一内核）
│   ├── worker/                 # Python 推理 Worker (vLLM/TensorRT)
│   └── coordinator_lite/       # Python 协调器（轮询调度）
│
├── rust_core/                  # ⚡ Rust Sidecar (高性能数据面)
│   ├── src/
│   └── Cargo.toml
│
├── CY_LLM_Backend/             # 🗂️ 遗留兼容目录（保留）
│   ├── worker/                 # [已迁移至 src/cy_llm/worker]
│   ├── coordinator_lite/       # [已迁移至 src/cy_llm/coordinator_lite]
│   └── deploy/                 # Docker 编排配置
│
├── docs/                       # 📚 项目文档
├── scripts/                    # 🛠️ 工具脚本
├── pyproject.toml              # Python 统一项目配置
├── CYLLM架构图.html             # 🎨 可视化架构图
└── README.md                   # 项目主文档
```

---

## 🔄 模块化设计理念

### 1️⃣ **Gateway 层（可插拔）**

**开源版**：`gateway/gateway_lite/`（Python FastAPI）
- ✅ 轻量级 REST API
- ✅ 基础 API Key 认证
- ✅ OpenAI 兼容接口

**企业版**：Kotlin Backend（外部项目）
- ✅ OAuth2/JWT 认证
- ✅ 多租户配额管理
- ✅ PostgreSQL 审计日志

**关键设计**：
- 两者通过 **统一的 gRPC 接口** 连接后端
- 替换 Gateway **不影响** Worker 层
- 参考 `gateway/INTERFACE_CONTRACT.md` 查看接口契约

---

### 2️⃣ **核心引擎（src/cy_llm/）**

**Worker**：推理引擎容器
- 支持 vLLM、TensorRT-LLM、MindIE
- 通过 UDS (Unix Domain Socket) 连接 Rust Sidecar
- 硬件无关抽象层（HAL）

**Coordinator**：请求调度器
- 开源版：简单轮询（Round-Robin）
- 企业版：可加载 Rust 拓扑感知插件

---

### 3️⃣ **Rust Sidecar（rust_core/）**

**功能**：
- 高性能流量转发（ZeroCopy）
- 替代 Python GIL 瓶颈
- 熔断、计数、遥测

**通信**：
- 与 Worker：UDS (`/tmp/cy_worker.sock`)
- 与 Coordinator：gRPC over UDS

---

## 🚀 部署模式对比

### 模式 A：开源版（标准链路）

```
HTTP → Python Gateway → Coordinator → Rust Sidecar → Worker
       (FastAPI :8000)  (UDS)         (UDS)
```

**适用场景**：
- 个人开发者、学术研究
- 单机或小规模部署
- 无需多租户管理

---

### 模式 B：企业版（直连 Sidecar）

```
HTTP → Kotlin Backend → Rust Sidecar → Worker
       (Spring :8080)   (gRPC :50050)
```

**适用场景**：
- 生产环境、商业项目
- 多租户 SaaS 平台
- 需要审计日志、配额管理

---

## 📦 依赖管理

### 根项目 (pyproject.toml)

```bash
# 安装核心引擎
pip install -e .

# 安装 NVIDIA 支持
pip install -e .[nvidia]

# 安装开发工具
pip install -e .[dev]
```

### Gateway (gateway/pyproject.toml)

```bash
# 独立安装 Gateway
cd gateway
pip install -e .
```

---

## 🔧 环境变量规范

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `WORKER_UDS_PATH` | `/tmp/cy_worker.sock` | Worker 监听地址 |
| `COORDINATOR_UDS_PATH` | `/tmp/cy_coordinator.sock` | Coordinator 监听地址 |
| `GATEWAY_API_TOKEN` | 空 | 开源版 API Key |

**企业版额外变量**：
- `OAUTH2_ISSUER`：OAuth2 认证服务器
- `DB_URL`：PostgreSQL 连接串
- `SIDECAR_GRPC_ADDR`：直连 Sidecar 地址

---

## 📝 升级路径

### 从旧版本迁移

1. **保留旧目录**（`CY_LLM_Backend/`）作为兼容层
2. **新代码优先使用** `src/cy_llm/` 和 `gateway/`
3. **逐步清理** `CY_LLM_Backend/` 中的冗余文件

### 从开源版升级到企业版

1. **替换 Gateway**：
   ```bash
   # 停用 Python Gateway
   docker stop cy-llm-gateway
   
   # 启动 Kotlin Backend
   docker run -d kotlin-backend:latest
   ```

2. **无需修改**：
   - Worker 层配置保持不变
   - Rust Sidecar 无需重新编译
   - 仅需修改环境变量指向新 Gateway

---

## 🎯 设计哲学

- **接口标准化 > 实现语言**
- **可替换组件 > 单体架构**
- **统一内核 + 插件增强**

---

**更多细节请参考**：
- 架构图：`CYLLM架构图.html`
- Gateway 对接：`gateway/INTERFACE_CONTRACT.md`
- 核心文档：`docs/ARCHITECTURE.md`
