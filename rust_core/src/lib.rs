/// Rust Sidecar - Worker 数据面代理
/// 
/// 核心功能：
/// - 协议卸载（gRPC 双向流处理）
/// - Token 精确计费
/// - Worker 健康检查与熔断
/// - 无感热更新支持

pub mod config;
pub mod errors;
pub mod metering;
pub mod metrics;
pub mod proxy;
pub mod health;
pub mod grpc_service;

// Re-exports
pub use config::SidecarConfig;
pub use errors::{SidecarError, Result};
pub use metering::TokenCounter;
pub use metrics::MetricsCollector;

/// Sidecar 版本信息
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
