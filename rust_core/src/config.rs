/// 配置管理模块 - 简化版
use crate::errors::{Result, SidecarError};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Sidecar 主配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SidecarConfig {
    pub server: ServerConfig,
    pub metering: MeteringConfig,
    pub health: HealthConfig,
    pub observability: ObservabilityConfig,
}

/// 服务器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// 外部 gRPC 监听地址（来自 Gateway）
    #[serde(default = "default_bind_addr")]
    pub bind_addr: String,

    /// 本地 Worker UDS 路径
    #[serde(default = "default_worker_uds")]
    pub worker_uds: String,
}

/// Token 计费配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeteringConfig {
    /// 计费上报地址（可选）
    pub billing_endpoint: Option<String>,

    /// 批量上报间隔（秒）
    #[serde(default = "default_batch_interval")]
    pub batch_interval_secs: u64,

    /// 离线队列大小
    #[serde(default = "default_offline_queue_size")]
    pub offline_queue_size: usize,
}

/// 健康检查配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthConfig {
    /// Worker 健康检查间隔（秒）
    #[serde(default = "default_check_interval")]
    pub check_interval_secs: u64,

    /// 重连等待时间（秒）
    #[serde(default = "default_reconnect_delay")]
    pub reconnect_delay_secs: u64,

    /// 最大重连次数
    #[serde(default = "default_max_reconnect_attempts")]
    pub max_reconnect_attempts: u32,
}

/// 可观测性配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservabilityConfig {
    /// Prometheus 指标端口
    #[serde(default = "default_metrics_port")]
    pub metrics_port: u16,

    /// 日志级别
    #[serde(default = "default_log_level")]
    pub log_level: String,
}

impl SidecarConfig {
    /// 从 TOML 文件加载配置
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| SidecarError::ConfigError(format!("Failed to read config file: {}", e)))?;

        toml::from_str(&content)
            .map_err(|e| SidecarError::ConfigError(format!("Failed to parse config: {}", e)))
    }

    /// 从环境变量加载配置（覆盖文件配置）
    pub fn merge_env(&mut self) {
        if let Ok(bind_addr) = std::env::var("SIDECAR_BIND_ADDR") {
            self.server.bind_addr = bind_addr;
        }

        if let Ok(worker_uds) = std::env::var("SIDECAR_WORKER_UDS") {
            self.server.worker_uds = worker_uds;
        }

        if let Ok(billing_endpoint) = std::env::var("SIDECAR_BILLING_ENDPOINT") {
            self.metering.billing_endpoint = Some(billing_endpoint);
        }

        if let Ok(log_level) = std::env::var("SIDECAR_LOG_LEVEL") {
            self.observability.log_level = log_level;
        }
    }

    /// 验证配置有效性
    pub fn validate(&self) -> Result<()> {
        // 检查 Worker UDS 路径
        if !self.server.worker_uds.starts_with("unix://") {
            return Err(SidecarError::ConfigError(
                "worker_uds must start with 'unix://'".to_string(),
            ));
        }

        // 检查 bind_addr 格式
        if !self.server.bind_addr.contains(':') {
            return Err(SidecarError::ConfigError(
                "bind_addr must be in format 'host:port'".to_string(),
            ));
        }

        Ok(())
    }
}

impl Default for SidecarConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig {
                bind_addr: default_bind_addr(),
                worker_uds: default_worker_uds(),
            },
            metering: MeteringConfig {
                billing_endpoint: None,
                batch_interval_secs: default_batch_interval(),
                offline_queue_size: default_offline_queue_size(),
            },
            health: HealthConfig {
                check_interval_secs: default_check_interval(),
                reconnect_delay_secs: default_reconnect_delay(),
                max_reconnect_attempts: default_max_reconnect_attempts(),
            },
            observability: ObservabilityConfig {
                metrics_port: default_metrics_port(),
                log_level: default_log_level(),
            },
        }
    }
}

// Default value functions
fn default_bind_addr() -> String {
    "0.0.0.0:50051".to_string()
}

fn default_worker_uds() -> String {
    "unix:///tmp/cy_worker.sock".to_string()
}

fn default_batch_interval() -> u64 {
    10
}

fn default_offline_queue_size() -> usize {
    10000
}

fn default_check_interval() -> u64 {
    5
}

fn default_reconnect_delay() -> u64 {
    2
}

fn default_max_reconnect_attempts() -> u32 {
    5
}

fn default_metrics_port() -> u16 {
    9090
}

fn default_log_level() -> String {
    "info".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SidecarConfig::default();
        assert_eq!(config.server.bind_addr, "0.0.0.0:50051");
        assert_eq!(config.server.worker_uds, "unix:///tmp/cy_worker.sock");
        assert_eq!(config.metering.batch_interval_secs, 10);
        assert_eq!(config.health.check_interval_secs, 5);
        assert_eq!(config.observability.metrics_port, 9090);
    }

    #[test]
    fn test_config_validation() {
        let mut config = SidecarConfig::default();
        assert!(config.validate().is_ok());

        config.server.worker_uds = "/tmp/invalid.sock".to_string();
        assert!(config.validate().is_err());

        config.server.worker_uds = "unix:///tmp/valid.sock".to_string();
        config.server.bind_addr = "invalid_addr".to_string();
        assert!(config.validate().is_err());
    }
}
