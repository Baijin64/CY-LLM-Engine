# Rust Sidecar 完整测试指南

## ✅ 当前状态

- **编译**: ✅ 成功 (Release 优化模式)
- **单元测试**: ✅ 8/8 通过
- **集成测试**: ✅ 5/5 通过  
- **gRPC 端口**: ✅ 50051 监听正常 (已实现 `AiInference` 服务)
- **Metrics 端口**: ✅ 9090 监听正常 (Prometheus 指标导出)
- **Proto 文件**: ✅ 复用 `CY_LLM_Backend/proto/ai_service.proto`
- **核心逻辑**: ✅ 实现 `StreamPredict` 转发与 UDS 连接

---

## 🧪 测试方法

### 1. 单独测试 Rust Sidecar（无需 Worker）

```bash
cd /home/wtx/ElementWarfare/CY-LLM-Engine/rust_core

# 编译
cargo build --release --no-default-features

# 运行测试脚本 (包含所有验证项)
./test_sidecar.sh

# 启动 Sidecar
./target/release/sidecar
```

**验证端口监听：**
```bash
# 另一个终端
ss -ltn | grep -E ':(50051|9090)'

# 测试 Metrics (应返回 prometheus 格式数据)
curl http://localhost:9090/metrics

# 测试 Health（检查 Sidecar 自身 gRPC 响应）
grpcurl -plaintext localhost:50051 cy.llm.AiInference/Health
```

---

### 2. 完整集成测试（Sidecar + Worker）

#### 步骤 1：启动 Python Worker

```bash
cd /home/wtx/ElementWarfare/CY-LLM-Engine
export CY_LLM_HEALTH_PORT=9091

# 启动并监听 UDS (Unix Domain Socket)
python -m CY_LLM_Backend.worker.main \
    --serve \
    --uds-path /tmp/cy_worker.sock \
    --model default
```

**日志检查点：**
- `[INFO] Worker socket created at /tmp/cy_worker.sock`

#### 步骤 2：启动 Rust Sidecar

```bash
# 新终端
cd /home/wtx/ElementWarfare/CY-LLM-Engine/rust_core
./target/release/sidecar
```

**日志检查点：**
- `INFO: gRPC server binding to 0.0.0.0:50051`
- `INFO: Metrics server listening on http://0.0.0.0:9090/metrics`
- 成功连接后，警告 `Worker socket not found` 将消失。

#### 步骤 3：发送真实推理请求

```bash
# 新终端
grpcurl -plaintext \
    -d '{"model_id":"default","prompt":"Hello, world!"}' \
    localhost:50051 \
    cy.llm.AiInference/StreamPredict
```

---

## 🔍 调试技巧

### 查看详细日志
```bash
SIDECAR_LOG_LEVEL=debug ./target/release/sidecar
```

### 监控指标实时变化
```bash
watch -n 1 "curl -s http://localhost:9090/metrics | grep sidecar"
```

---

## ❌ 常见问题 (FAQ)

### 问题 1：Worker 启动提示 ModuleNotFoundError
**解决**：确保在工作区根目录下执行，并使用 `python -m CY_LLM_Backend.worker.main`，不要漏掉 `CY_LLM_Backend` 前缀。

### 问题 2：Cargo.lock 版本 4 错误
**解决**：运行 `rm rust_core/Cargo.lock && cd rust_core && cargo build`。

### 问题 3：端口 50051 拒绝连接
**解决**：检查 Sidecar 日志是否显示 `gRPC server binding...`。如果显示，检查防火墙或 `ss -ltn` 确认端口是否开启。

---

## 🚀 下一步开发

1. **多 Worker 负载均衡**
   - 当前 Sidecar 采用单 Worker 绑定模式 (1:1 Sidecar Pattern)。

2. **Docker 化集成**
   - 编写 `rust_core/Dockerfile` 并通过 `docker-compose.community.yml` 统一部署。

3. **命令集成**
   - 修改 `cy-llm` 启动脚本，支持 `./cy-llm lite --use-rust`。

---

## 📝 相关文档

- [rust_core/ARCHITECTURE.md](ARCHITECTURE.md)
- [rust_core/README.md](README.md)
- [CY_LLM_Backend/proto/ai_service.proto](../CY_LLM_Backend/proto/ai_service.proto)
