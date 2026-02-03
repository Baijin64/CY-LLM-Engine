/// 指标收集模块 - 数据面代理
use prometheus::{Counter, Histogram, HistogramOpts, IntCounter, IntGauge, Opts, Registry};
use std::sync::Arc;

/// 指标收集器
#[derive(Clone)]
pub struct MetricsCollector {
    registry: Arc<Registry>,

    // 请求计数
    pub requests_total: IntCounter,
    pub requests_success: IntCounter,
    pub requests_failed: IntCounter,

    // Token 计费
    pub tokens_processed_total: Counter,

    // Worker 状态
    pub worker_connection_status: IntGauge,
    pub worker_connection_errors: IntCounter,

    // 代理性能
    pub gateway_send_errors: IntCounter,
    pub request_duration: Histogram,

    // 活跃连接数
    pub active_connections: IntGauge,
}

impl MetricsCollector {
    /// 创建新的指标收集器
    pub fn new() -> anyhow::Result<Self> {
        let registry = Arc::new(Registry::new());

        let requests_total = IntCounter::with_opts(Opts::new(
            "sidecar_requests_total",
            "Total number of requests",
        ))?;

        let requests_success = IntCounter::with_opts(Opts::new(
            "sidecar_requests_success_total",
            "Total number of successful requests",
        ))?;

        let requests_failed = IntCounter::with_opts(Opts::new(
            "sidecar_requests_failed_total",
            "Total number of failed requests",
        ))?;

        let tokens_processed_total = Counter::with_opts(Opts::new(
            "sidecar_tokens_processed_total",
            "Total number of tokens processed",
        ))?;

        let worker_connection_status = IntGauge::with_opts(Opts::new(
            "sidecar_worker_connection_status",
            "Worker connection status (1=connected, 0=disconnected, 2=reloading)",
        ))?;

        let worker_connection_errors = IntCounter::with_opts(Opts::new(
            "sidecar_worker_connection_errors_total",
            "Total number of worker connection errors",
        ))?;

        let gateway_send_errors = IntCounter::with_opts(Opts::new(
            "sidecar_gateway_send_errors_total",
            "Total number of gateway send errors",
        ))?;

        let request_duration = Histogram::with_opts(HistogramOpts::new(
            "sidecar_request_duration_seconds",
            "Request duration in seconds",
        ))?;

        let active_connections = IntGauge::with_opts(Opts::new(
            "sidecar_active_connections",
            "Number of active connections",
        ))?;

        // 注册指标
        registry.register(Box::new(requests_total.clone()))?;
        registry.register(Box::new(requests_success.clone()))?;
        registry.register(Box::new(requests_failed.clone()))?;
        registry.register(Box::new(tokens_processed_total.clone()))?;
        registry.register(Box::new(worker_connection_status.clone()))?;
        registry.register(Box::new(worker_connection_errors.clone()))?;
        registry.register(Box::new(gateway_send_errors.clone()))?;
        registry.register(Box::new(request_duration.clone()))?;
        registry.register(Box::new(active_connections.clone()))?;

        Ok(Self {
            registry,
            requests_total,
            requests_success,
            requests_failed,
            tokens_processed_total,
            worker_connection_status,
            worker_connection_errors,
            gateway_send_errors,
            request_duration,
            active_connections,
        })
    }

    /// 获取 Prometheus Registry
    pub fn registry(&self) -> Arc<Registry> {
        Arc::clone(&self.registry)
    }

    /// 记录请求
    pub fn record_request(&self, success: bool, duration_secs: f64) {
        self.requests_total.inc();
        if success {
            self.requests_success.inc();
        } else {
            self.requests_failed.inc();
        }
        self.request_duration.observe(duration_secs);
    }

    /// 记录 Token 处理
    pub fn record_tokens(&self, count: f64) {
        self.tokens_processed_total.inc_by(count);
    }

    /// 更新 Worker 连接状态
    pub fn set_worker_status(&self, status: WorkerConnectionStatus) {
        let value = match status {
            WorkerConnectionStatus::Connected => 1,
            WorkerConnectionStatus::Disconnected => 0,
            WorkerConnectionStatus::Reloading => 2,
        };
        self.worker_connection_status.set(value);
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new().expect("Failed to create metrics collector")
    }
}

/// Worker 连接状态
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerConnectionStatus {
    Connected,
    Disconnected,
    Reloading,
}
