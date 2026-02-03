# Rust Sidecar - 企业级安全架构

## 🔒 安全特性概览

### 1. **传输层安全 (TLS/mTLS)**

#### 单向 TLS
```toml
[security]
enable_tls = true
cert_path = "/etc/cy-llm/certs/server.crt"
key_path = "/etc/cy-llm/certs/server.key"
require_client_cert = false
```

#### 双向 mTLS（推荐生产环境）
```toml
[security]
enable_tls = true
cert_path = "/etc/cy-llm/certs/server.crt"
key_path = "/etc/cy-llm/certs/server.key"
ca_cert_path = "/etc/cy-llm/certs/ca.crt"
require_client_cert = true
```

**生成测试证书**：
```bash
# 生成 CA 私钥和证书
openssl genrsa -out ca.key 4096
openssl req -new -x509 -days 3650 -key ca.key -out ca.crt \
  -subj "/CN=CY-LLM CA"

# 生成服务端证书
openssl genrsa -out server.key 4096
openssl req -new -key server.key -out server.csr \
  -subj "/CN=cy-llm-sidecar"
openssl x509 -req -days 365 -in server.csr -CA ca.crt -CAkey ca.key \
  -CAcreateserial -out server.crt

# 生成客户端证书（mTLS）
openssl genrsa -out client.key 4096
openssl req -new -key client.key -out client.csr \
  -subj "/CN=cy-llm-client"
openssl x509 -req -days 365 -in client.csr -CA ca.crt -CAkey ca.key \
  -CAcreateserial -out client.crt
```

---

### 2. **身份验证与授权 (JWT + RBAC)**

#### JWT 令牌生成示例
```rust
// 在 Rust 中生成（测试用）
let validator = JwtValidator::new(b"your-secret-key", "cy-llm");
let token = validator.generate("user123", vec!["user".to_string()], 3600)?;
```

#### RBAC 策略
默认策略：
- **admin**：可访问所有资源 (`/`)
- **user**：可访问推理接口 (`/v1/chat`, `/v1/completions`)
- **guest**：拒绝访问

**自定义策略**：
```rust
let mut policy = RbacPolicy::new();
policy.add_rule("/admin", vec!["admin".to_string()]);
policy.add_rule("/v1/chat", vec!["user".to_string(), "premium".to_string()]);
```

---

### 3. **请求限流 (Rate Limiting)**

配置示例：
```toml
[rate_limit]
requests_per_second = 1000  # 每秒最大请求数
burst_size = 100             # 突发流量缓冲
```

**算法**：Token Bucket
- 平滑限流，允许短时突发
- 超过限制返回 `429 Too Many Requests`

---

### 4. **熔断器 (Circuit Breaker)**

配置示例：
```toml
[circuit_breaker]
failure_threshold = 5             # 连续失败 5 次触发熔断
success_threshold = 2             # 半开状态成功 2 次恢复
timeout_seconds = 30              # 请求超时时间
recovery_timeout_seconds = 60     # 熔断打开后多久尝试恢复
```

**状态机**：
```
[Closed] --5 failures--> [Open] --60s--> [HalfOpen] --2 success--> [Closed]
                                              |
                                          1 failure
                                              ↓
                                           [Open]
```

---

### 5. **审计日志与遥测**

#### Prometheus 指标

访问 `http://localhost:9090/metrics` 查看：

```prometheus
# 请求总数
sidecar_requests_total
sidecar_requests_success_total
sidecar_requests_failed_total

# 认证
sidecar_auth_success_total
sidecar_auth_failed_total

# 限流与熔断
sidecar_rate_limited_total
sidecar_circuit_breaker_open_total

# 延迟直方图
sidecar_request_duration_seconds_bucket
sidecar_request_duration_seconds_sum
sidecar_request_duration_seconds_count

# 活跃连接数
sidecar_active_connections
```

#### OpenTelemetry 集成（可选）

```toml
[observability]
enable_otel = true
otlp_endpoint = "http://localhost:4317"
```

支持导出到：
- **Jaeger**（分布式追踪）
- **Prometheus**（指标）
- **Grafana Loki**（日志）

---

## 🛡️ 安全加固建议

### 生产环境检查清单

#### ✅ 必须项
- [ ] 修改默认 JWT Secret（最少 32 字节）
- [ ] 启用 TLS/mTLS
- [ ] 配置防火墙规则（仅允许内网访问）
- [ ] 设置文件权限（`chmod 600` 私钥文件）
- [ ] 定期轮换证书（建议 90 天）

#### ✅ 推荐项
- [ ] 启用审计日志持久化
- [ ] 配置 SIEM 告警规则
- [ ] 实施网络隔离（DMZ）
- [ ] 定期安全扫描（`cargo audit`）
- [ ] 启用 OpenTelemetry 追踪

#### ✅ 高级项
- [ ] 集成 HSM（硬件安全模块）存储私钥
- [ ] 实施 WAF（Web 应用防火墙）
- [ ] 配置 DDoS 防护
- [ ] 实施零信任网络架构
- [ ] 定期渗透测试

---

## 🚨 常见安全问题

### Q: 如何防止中间人攻击？
**A**: 启用 mTLS，客户端和服务端互相验证证书。

### Q: JWT 令牌泄露怎么办？
**A**: 
1. 设置短过期时间（建议 1 小时）
2. 实施令牌黑名单机制
3. 使用 HTTPS 传输

### Q: 如何防止重放攻击？
**A**:
1. JWT 中包含 `jti`（唯一 ID）和 `exp`（过期时间）
2. 服务端缓存已使用的 `jti`
3. 使用 TLS 防止窃听

### Q: 如何防止暴力破解？
**A**:
1. 启用请求限流
2. 实施账户锁定策略
3. 记录失败尝试并告警

---

## 📊 性能影响

| 安全特性 | 延迟影响 | 吞吐量影响 | 建议 |
|----------|----------|------------|------|
| TLS | +1-2ms | -5% | 生产必须 |
| mTLS | +2-3ms | -10% | 高安全场景 |
| JWT 验证 | +0.1ms | -1% | 推荐启用 |
| RBAC 检查 | +0.05ms | ~0% | 推荐启用 |
| 请求限流 | +0.01ms | ~0% | 推荐启用 |
| 熔断器 | +0.01ms | ~0% | 推荐启用 |

**总体影响**：在启用全部安全特性的情况下，预期延迟增加 3-5ms，吞吐量下降 10-15%。

---

## 🔧 运维命令

### 编译
```bash
cd rust_core
cargo build --release
```

### 运行
```bash
export SIDECAR_CONFIG=./sidecar.toml
./target/release/sidecar
```

### 测试
```bash
cargo test
```

### 安全审计
```bash
cargo audit
cargo deny check
```

### 性能测试
```bash
cargo bench
```

---

## 📞 安全报告

发现安全漏洞请发送邮件至：
**security@cy-llm.example.com**

我们承诺在 24 小时内响应。
