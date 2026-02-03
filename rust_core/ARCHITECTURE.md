# Rust Sidecar - Worker 数据面代理

## 🎯 架构定位

**NOT**: 中心化安全网关  
**YES**: Worker Pod 的伴生容器（Sidecar Pattern）

```
┌────────────────────────────────────────────────────────┐
│  GPU Node (同一台物理机)                                │
│                                                         │
│  ┌─────────────────┐         ┌──────────────────┐     │
│  │  Rust Sidecar   │◄───UDS──┤ Python Worker    │     │
│  │  (数据面代理)    │         │ (推理引擎)        │     │
│  │                 │         │                  │     │
│  │ • 协议卸载      │         │ • vLLM/TensorRT  │     │
│  │ • Token 计费    │         │ • 模型加载       │     │
│  │ • 熔断保护      │         │ • Token 生成     │     │
│  │ • 热更新支持    │         │                  │     │
│  └────────┬────────┘         └──────────────────┘     │
│           │                                            │
└───────────┼────────────────────────────────────────────┘
            │ gRPC/HTTP2 (网络)
            ▼
    ┌───────────────┐
    │  Gateway      │ (可以是 Python 或 Kotlin Backend)
    │  (控制面)      │
    └───────────────┘
```

---

## 核心职责

### 1️⃣ **协议卸载 (Protocol Offloading)**

**问题**：Python Worker 不应该关心复杂的网络协议。

**解决**：
- Rust 处理 gRPC 双向流、HTTP/2 多路复用
- Python Worker 只需简单的本地 socket，发 JSON/Protobuf
- 支持心跳维持、断线重连、背压控制

---

### 2️⃣ **精确计费 (Token Metering)**

**问题**：Token 是计费基础，必须 100% 准确。

**解决**：
```rust
// Rust Sidecar 作为"水表"
let mut token_counter = 0;
for chunk in python_worker.stream_tokens() {
    token_counter += chunk.len();
    gateway.send(chunk).await?;
}

// 异步发送计费数据
billing_system.record(session_id, token_counter).await;
```

**关键优势**：
- 即使 Gateway 断连，Sidecar 也能本地记录
- 定期批量上报，减少网络开销
- 支持离线重试队列

---

### 3️⃣ **故障熔断与优雅降级**

**场景**：Python Worker OOM 崩溃。

**传统方式**：
```
Gateway → (timeout 30s) → 503 Error
```

**Sidecar 方式**：
```rust
match python_worker.connect().await {
    Ok(_) => { /* 正常转发 */ },
    Err(ConnectionRefused) => {
        // 立即返回优雅错误
        return Response::new(StatusCode::SERVICE_UNAVAILABLE)
            .body("Worker is restarting, please retry in 5s");
    }
}
```

---

### 4️⃣ **无感热更新 (Zero-Downtime Reload)**

**场景**：加载新模型，需要重启 Python Worker。

**Sidecar 支持**：
```rust
// Python Worker 发送 "RELOADING" 信号
if worker_status == WorkerStatus::Reloading {
    // 保持 Gateway 连接，返回友好提示
    stream.send(Token {
        text: "[System] Loading new model, ETA 30s...",
        is_system_message: true,
    }).await?;
}

// Python Worker 重启完成后自动重连
```

**用户体验**：
```
User: "帮我写代码"
Bot:  "[System] Loading new model, ETA 30s..."
      (30秒后)
      "好的，我来帮你写..."
```

---

## 🚫 **不包含的功能**（留给 Kotlin Backend）

| 功能 | Rust Sidecar | Kotlin Backend |
|------|--------------|----------------|
| **JWT 验证** | ❌ | ✅ |
| **RBAC 权限** | ❌ | ✅ |
| **多租户隔离** | ❌ | ✅ |
| **审计日志存储** | ❌ | ✅ (PostgreSQL) |
| **计费账单生成** | ❌ | ✅ |
| **Token 计数** | ✅ (原始数据) | ✅ (聚合统计) |
| **协议转换** | ✅ | ❌ |
| **熔断保护** | ✅ | ❌ |
| **Worker 健康检查** | ✅ | ❌ |

---

## 📦 简化后的依赖

```toml
[dependencies]
# 核心异步运行时
tokio = { version = "1.35", features = ["full"] }

# gRPC（仅客户端/服务端，无 TLS）
tonic = "0.11"
prost = "0.12"

# 指标（本地导出）
prometheus = "0.13"

# 日志
tracing = "0.1"

# 序列化
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# 错误处理
anyhow = "1.0"
thiserror = "1.0"
```

**移除**：
- ~~rustls（TLS 由 Kotlin Backend 处理）~~
- ~~jsonwebtoken（JWT 由 Kotlin Backend 验证）~~
- ~~governor（限流由 Gateway 处理）~~

---

## 🎛️ 配置文件

```toml
# sidecar.toml
[server]
# 监听外部 gRPC 请求（来自 Gateway）
bind_addr = "0.0.0.0:50051"

# 连接本地 Python Worker
worker_uds = "unix:///tmp/cy_worker.sock"

[metering]
# Token 计费上报地址（可选，Kotlin Backend）
billing_endpoint = "http://billing-service:8080/v1/usage"

# 批量上报间隔
batch_interval_secs = 10

# 离线队列大小
offline_queue_size = 10000

[health]
# Worker 健康检查间隔
check_interval_secs = 5

# 重连等待时间
reconnect_delay_secs = 2

[observability]
# Prometheus 指标端口
metrics_port = 9090
```

---

## 🔧 部署方式

### Kubernetes Pod Spec

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: worker-gpu-001
spec:
  containers:
    # Sidecar: Rust Proxy
    - name: sidecar
      image: cy-llm-sidecar:latest
      ports:
        - containerPort: 50051  # gRPC
        - containerPort: 9090   # Metrics
      volumeMounts:
        - name: worker-socket
          mountPath: /tmp
      resources:
        limits:
          cpu: "500m"
          memory: "128Mi"

    # Main: Python Worker
    - name: worker
      image: cy-llm-worker:latest
      volumeMounts:
        - name: worker-socket
          mountPath: /tmp
      resources:
        limits:
          nvidia.com/gpu: 1
          memory: "32Gi"

  volumes:
    - name: worker-socket
      emptyDir: {}
```

---

## 📊 关键指标

```prometheus
# Token 计费
sidecar_tokens_processed_total{session_id, model_id}

# Worker 健康
sidecar_worker_connection_status{status="connected|disconnected|reloading"}

# 请求统计
sidecar_requests_total
sidecar_request_duration_seconds

# 错误统计
sidecar_worker_connection_errors_total
sidecar_gateway_send_errors_total
```

---

## 🚀 性能目标

| 指标 | 目标 | 说明 |
|------|------|------|
| **延迟开销** | < 0.1ms | UDS 通信几乎零开销 |
| **内存占用** | < 50MB | 轻量级代理 |
| **CPU 占用** | < 5% | 纯转发，无复杂计算 |
| **吞吐量** | > 10k req/s | 单 Sidecar 实例 |

---

## ✅ 总结

**Rust Sidecar 的核心价值**：

1. **数据面代理**：高效转发，协议卸载
2. **计费基础**：精确 Token 计数，离线容错
3. **稳定性保障**：熔断保护，优雅降级
4. **运维友好**：无感热更新，实时监控

**不做的事情**：

- ❌ 不做身份认证（交给 Kotlin Backend）
- ❌ 不做权限控制（交给 Gateway）
- ❌ 不做中心化调度（避免瓶颈）
- ❌ 不做业务逻辑（纯数据传输）

**这样设计的好处**：

- 🎯 **职责单一**：专注数据面，极致稳定
- ⚡ **性能极致**：Rust + UDS，接近原生速度
- 🔧 **易于扩展**：每个 Worker 独立，水平扩展
- 🛡️ **故障隔离**：单个 Worker 崩溃不影响全局
