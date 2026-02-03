# Rust Sidecar - Worker 数据面代理

[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

## 🎯 架构定位

**Rust Sidecar 是 Python Worker 的伴生容器（Sidecar Pattern），专注于数据面代理。**

```
┌────────────────────────────────────────┐
│  GPU Node (同一台物理机)               │
│                                         │
│  ┌─────────────┐    ┌──────────────┐   │
│  │ Rust Sidecar│◄───┤Python Worker │   │
│  │ (数据代理)  │UDS │(推理引擎)    │   │
│  └──────┬──────┘    └──────────────┘   │
└─────────┼──────────────────────────────┘
          │ gRPC/HTTP2
          ▼
    ┌────────────┐
    │  Gateway   │
    └────────────┘
```

## 🚀 核心功能

### 1. 协议卸载 (Protocol Offloading)
- 处理 gRPC 双向流、HTTP/2 多路复用
- Python Worker 仅需简单的 UDS 通信
- 支持心跳维持、断线重连、背压控制

### 2. 精确计费 (Token Metering)
- 像"水表"一样统计流经的 Token 数量
- 即使 Gateway 断连也能本地记录
- 定期批量上报，支持离线重试队列

### 3. 故障熔断与优雅降级
- Worker OOM 时立即返回优雅错误（而非超时）
- 实时健康检查，快速故障检测
- 自动重连机制

### 4. 无感热更新 (Zero-Downtime Reload)
- Worker 重启时保持 Gateway 连接
- 发送友好的 "Loading..." 消息
- 自动等待 Worker 恢复

## 📦 构建与运行

### 前置要求

- Rust 1.75+
- Protocol Buffers compiler (`protoc`)
- Linux/WSL (UDS 支持)

### 编译

```bash
cd rust_core

# 开发构建
cargo build

# 生产构建（优化）
cargo build --release --no-default-features
```

### 运行

```bash
# 使用默认配置
./target/release/sidecar

# 指定配置文件
SIDECAR_CONFIG=custom.toml ./target/release/sidecar

# 使用环境变量覆盖
SIDECAR_BIND_ADDR=0.0.0.0:8080 \
SIDECAR_WORKER_UDS=unix:///tmp/custom_worker.sock \
./target/release/sidecar
```

### 测试

```bash
# 运行所有测试
cargo test

# 运行特定测试
cargo test test_token_counting

# 集成测试
cargo test --test integration_test
```

## ⚙️ 配置

### 配置文件 (sidecar.toml)

```toml
[server]
bind_addr = "0.0.0.0:50051"
worker_uds = "unix:///tmp/cy_worker.sock"

[metering]
batch_interval_secs = 10
offline_queue_size = 10000

[health]
check_interval_secs = 5
reconnect_delay_secs = 2
max_reconnect_attempts = 5

[observability]
metrics_port = 9090
log_level = "info"
```

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `SIDECAR_CONFIG` | 配置文件路径 | `sidecar.toml` |
| `SIDECAR_BIND_ADDR` | gRPC 监听地址 | `0.0.0.0:50051` |
| `SIDECAR_WORKER_UDS` | Worker UDS 路径 | `unix:///tmp/cy_worker.sock` |
| `SIDECAR_LOG_LEVEL` | 日志级别 | `info` |

## 📊 监控指标

Sidecar 在 `:9090/metrics` 暴露 Prometheus 指标：

```prometheus
# Token 计费
sidecar_tokens_processed_total

# Worker 连接状态
sidecar_worker_connection_status

# 请求统计
sidecar_requests_total
sidecar_requests_success_total
sidecar_requests_failed_total
sidecar_request_duration_seconds

# 活跃连接
sidecar_active_connections

# 错误统计
sidecar_worker_connection_errors_total
sidecar_gateway_send_errors_total
```

## 🔧 部署

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

### Docker Compose

```yaml
version: '3.8'
services:
  worker:
    image: cy-llm-worker:latest
    command: python -m src.cy_llm.worker.main --serve --uds-path /tmp/cy_worker.sock
    volumes:
      - worker-socket:/tmp
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  sidecar:
    image: cy-llm-sidecar:latest
    ports:
      - "50051:50051"
      - "9090:9090"
    volumes:
      - worker-socket:/tmp
    depends_on:
      - worker

volumes:
  worker-socket:
```

## 🛡️ 性能目标

| 指标 | 目标 | 说明 |
|------|------|------|
| **延迟开销** | < 0.1ms | UDS 通信几乎零开销 |
| **内存占用** | < 50MB | 轻量级代理 |
| **CPU 占用** | < 5% | 纯转发，无复杂计算 |
| **吞吐量** | > 10k req/s | 单 Sidecar 实例 |

## 🚫 不包含的功能

以下功能由 **Kotlin Backend** 或 **Gateway** 处理：

- ❌ JWT 验证
- ❌ RBAC 权限控制
- ❌ 多租户隔离
- ❌ TLS/mTLS 加密
- ❌ 审计日志存储
- ❌ 计费账单生成（仅提供原始 Token 计数）

## 📁 项目结构

```
rust_core/
├── Cargo.toml              # 依赖配置
├── build.rs                # Protobuf 编译脚本
├── sidecar.toml            # 默认配置文件
├── ARCHITECTURE.md         # 架构文档
├── src/
│   ├── lib.rs              # 库入口
│   ├── config.rs           # 配置管理
│   ├── errors.rs           # 错误定义
│   ├── health.rs           # 健康检查
│   ├── metering.rs         # Token 计费
│   ├── metrics.rs          # Prometheus 指标
│   ├── proxy.rs            # gRPC 代理核心
│   └── bin/
│       └── sidecar.rs      # 主程序入口
└── tests/
    └── integration_test.rs # 集成测试
```

## 🤝 开发指南

### 添加新功能

1. 在 `src/` 中创建新模块
2. 在 `src/lib.rs` 中导出
3. 编写单元测试（`#[cfg(test)]`）
4. 更新文档

### 代码风格

```bash
# 格式化代码
cargo fmt

# 静态检查
cargo clippy -- -D warnings

# 文档检查
cargo doc --no-deps
```

## 📄 许可证

Apache 2.0 - 详见 [LICENSE](../LICENSE)

## 🔗 相关文档

- [ARCHITECTURE.md](ARCHITECTURE.md) - 详细架构设计
- [../PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md) - 项目整体结构
- [../gateway/INTERFACE_CONTRACT.md](../gateway/INTERFACE_CONTRACT.md) - gRPC 接口契约
