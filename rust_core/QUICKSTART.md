# Rust Sidecar 快速开始

## 🛠️ 环境准备

### 1. 安装依赖

#### Ubuntu/Debian (WSL)
```bash
# 安装 Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# 安装 protobuf 编译器
sudo apt update
sudo apt install -y protobuf-compiler build-essential

# 验证安装
protoc --version  # 应显示版本号，如 libprotoc 3.12.4
```

#### macOS
```bash
# 安装 Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 安装 protobuf 编译器
brew install protobuf

# 验证安装
protoc --version
```

### 2. 编译 Rust Sidecar

```bash
cd rust_core

# 首次编译（会下载依赖，较慢）
cargo build --release --no-default-features

# 成功后可执行文件位于
# ./target/release/sidecar
```

### 3. 运行测试

```bash
# 运行单元测试
cargo test --lib

# 运行集成测试
cargo test --test integration_test

# 查看测试覆盖率
cargo test -- --nocapture
```

---

## 🚀 启动 Sidecar

### 方式 1：使用默认配置

```bash
# 确保 Python Worker 已启动
python -m src.cy_llm.worker.main --serve --uds-path /tmp/cy_worker.sock

# 在另一个终端启动 Sidecar
cd rust_core
./target/release/sidecar
```

**预期输出：**
```
🚀 Starting Rust Sidecar - Worker Data Plane Proxy
Version: 0.1.0
Loading config from sidecar.toml
Configuration:
  - Bind address: 0.0.0.0:50051
  - Worker UDS: unix:///tmp/cy_worker.sock
  - Metrics port: 9090
Connecting to Python Worker...
✅ Connected to Worker successfully
✅ Background health check started
✅ Metrics server started on 0.0.0.0:9090
🎯 Starting gRPC server on 0.0.0.0:50051
```

### 方式 2：自定义配置

```bash
# 创建自定义配置
cat > custom.toml <<EOF
[server]
bind_addr = "0.0.0.0:8080"
worker_uds = "unix:///tmp/custom_worker.sock"

[health]
check_interval_secs = 10
reconnect_delay_secs = 5

[observability]
metrics_port = 9091
log_level = "debug"
EOF

# 使用自定义配置启动
SIDECAR_CONFIG=custom.toml ./target/release/sidecar
```

### 方式 3：环境变量覆盖

```bash
SIDECAR_BIND_ADDR=0.0.0.0:9000 \
SIDECAR_WORKER_UDS=unix:///tmp/my_worker.sock \
SIDECAR_LOG_LEVEL=debug \
./target/release/sidecar
```

---

## 🧪 验证运行状态

### 1. 检查健康状态

```bash
# 使用 grpcurl（需先安装）
grpcurl -plaintext localhost:50051 cy.llm.AiInference/Health

# 或使用 curl 检查指标
curl http://localhost:9090/metrics
```

**预期输出：**
```
# HELP sidecar_requests_total Total number of requests
# TYPE sidecar_requests_total counter
sidecar_requests_total 0
# HELP sidecar_worker_connection_status Worker connection status (1=connected, 0=disconnected, 2=reloading)
# TYPE sidecar_worker_connection_status gauge
sidecar_worker_connection_status 1
```

### 2. 测试推理请求

```bash
# 使用 grpcurl 发送测试请求
grpcurl -plaintext -d '{
  "model_id": "qwen-7b",
  "prompt": "你好",
  "generation": {
    "max_new_tokens": 100,
    "temperature": 0.7
  }
}' localhost:50051 cy.llm.AiInference/StreamPredict
```

---

## 🐛 常见问题

### 问题 1: `protoc` 未找到

**错误信息：**
```
Could not find `protoc`. If `protoc` is installed, try setting the `PROTOC` environment variable
```

**解决方法：**
```bash
# Ubuntu/Debian
sudo apt install protobuf-compiler

# macOS
brew install protobuf

# 或手动指定 protoc 路径
PROTOC=/usr/local/bin/protoc cargo build --release
```

### 问题 2: Worker UDS 连接失败

**错误信息：**
```
❌ Failed to connect to Worker: Worker is not available
```

**解决方法：**
1. 检查 Python Worker 是否运行：
   ```bash
   ps aux | grep "cy_llm.worker.main"
   ```

2. 检查 UDS socket 是否存在：
   ```bash
   ls -l /tmp/cy_worker.sock
   ```

3. 确保路径一致：
   - Python Worker: `--uds-path /tmp/cy_worker.sock`
   - Rust Sidecar: `worker_uds = "unix:///tmp/cy_worker.sock"`

### 问题 3: 端口已被占用

**错误信息：**
```
Error: Os { code: 98, kind: AddrInUse, message: "Address already in use" }
```

**解决方法：**
```bash
# 查找占用端口的进程
lsof -i :50051

# 杀死进程或更改端口
SIDECAR_BIND_ADDR=0.0.0.0:50052 ./target/release/sidecar
```

### 问题 4: 权限问题（UDS）

**错误信息：**
```
Permission denied (os error 13)
```

**解决方法：**
```bash
# 确保 /tmp 目录有写权限
chmod 1777 /tmp

# 或使用用户自己的目录
mkdir -p ~/.cy_llm/sockets
SIDECAR_WORKER_UDS=unix://$HOME/.cy_llm/sockets/worker.sock ./target/release/sidecar
```

---

## 📊 性能调优

### 1. 生产环境编译优化

```bash
# 使用 LTO (Link-Time Optimization)
RUSTFLAGS="-C target-cpu=native" cargo build --release --no-default-features

# 结果二进制大小会减小，性能提升 10-20%
```

### 2. 调整 Tokio 线程池

```bash
# 设置 Worker 线程数（默认为 CPU 核心数）
TOKIO_WORKER_THREADS=8 ./target/release/sidecar
```

### 3. 日志级别调整

```toml
# sidecar.toml
[observability]
log_level = "warn"  # 生产环境建议使用 warn 或 error
```

---

## 🐳 Docker 部署

### 构建镜像

```bash
# 在项目根目录创建 Dockerfile
cd rust_core

# 构建
docker build -t cy-llm-sidecar:latest .

# 运行
docker run -d \
  --name sidecar \
  -p 50051:50051 \
  -p 9090:9090 \
  -v /tmp:/tmp \
  cy-llm-sidecar:latest
```

### Docker Compose 部署

```bash
# 使用项目根目录的 docker-compose.yml
docker-compose up -d sidecar
```

---

## 📈 监控集成

### Prometheus 配置

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'cy-llm-sidecar'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

### Grafana 仪表板

关键指标查询：

```promql
# Token 处理速率
rate(sidecar_tokens_processed_total[1m])

# 请求成功率
sum(rate(sidecar_requests_success_total[1m])) / sum(rate(sidecar_requests_total[1m]))

# Worker 连接状态
sidecar_worker_connection_status

# 平均请求延迟
rate(sidecar_request_duration_seconds_sum[1m]) / rate(sidecar_request_duration_seconds_count[1m])
```

---

## ✅ 下一步

1. **性能基准测试**
   ```bash
   cargo bench
   ```

2. **生产环境部署**
   - 参考 `k8s/deployment.yaml`
   - 配置监控告警

3. **日志聚合**
   - 集成 ELK Stack
   - 配置日志轮转

4. **安全加固**
   - 在外层 Gateway/Kotlin Backend 配置 TLS
   - 配置网络隔离策略

---

**问题反馈**: 如遇到问题，请检查：
1. `rust_core/IMPLEMENTATION_SUMMARY.md` - 实现总结
2. `rust_core/ARCHITECTURE.md` - 架构设计
3. `rust_core/README.md` - 详细文档
