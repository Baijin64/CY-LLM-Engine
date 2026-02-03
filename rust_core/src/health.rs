/// Worker 健康检查模块
use crate::errors::{Result, SidecarError};
use std::time::Duration;
use std::sync::Arc;
use tokio::time::sleep;
use tracing::{debug, error, info, warn};

/// Worker 连接状态
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerStatus {
    /// 已连接
    Connected,
    /// 断开连接
    Disconnected,
    /// 正在重载（热更新）
    Reloading,
}

/// 健康检查器
pub struct HealthChecker {
    /// UDS 路径
    worker_uds: String,
    /// 健康检查间隔
    check_interval: Duration,
    /// 重连延迟
    reconnect_delay: Duration,
    /// 最大重连次数
    max_reconnect_attempts: u32,
}

impl HealthChecker {
    /// 创建新的健康检查器
    pub fn new(
        worker_uds: String,
        check_interval_secs: u64,
        reconnect_delay_secs: u64,
        max_reconnect_attempts: u32,
    ) -> Self {
        Self {
            worker_uds,
            check_interval: Duration::from_secs(check_interval_secs),
            reconnect_delay: Duration::from_secs(reconnect_delay_secs),
            max_reconnect_attempts,
        }
    }

    /// 检查 Worker 是否可用（通过 UDS socket 是否存在）
    pub async fn check_worker_health(&self) -> WorkerStatus {
        // 提取 UDS 路径（去掉 "unix://" 前缀）
        let socket_path = self.worker_uds.strip_prefix("unix://").unwrap_or(&self.worker_uds);

        // 检查 socket 文件是否存在
        if tokio::fs::metadata(socket_path).await.is_ok() {
            debug!("Worker socket exists at {}", socket_path);
            WorkerStatus::Connected
        } else {
            warn!("Worker socket not found at {}", socket_path);
            WorkerStatus::Disconnected
        }
    }

    /// 等待 Worker 重新连接（带重试逻辑）
    pub async fn wait_for_reconnect(&self) -> Result<()> {
        info!("Waiting for Worker to reconnect...");

        for attempt in 1..=self.max_reconnect_attempts {
            sleep(self.reconnect_delay).await;

            let status = self.check_worker_health().await;
            if status == WorkerStatus::Connected {
                info!(
                    "Worker reconnected successfully after {} attempt(s)",
                    attempt
                );
                return Ok(());
            }

            warn!(
                "Worker still disconnected (attempt {}/{})",
                attempt, self.max_reconnect_attempts
            );
        }

        error!(
            "Worker failed to reconnect after {} attempts",
            self.max_reconnect_attempts
        );
        Err(SidecarError::WorkerDisconnected(
            format!(
                "Failed to reconnect after {} attempts",
                self.max_reconnect_attempts
            ),
        ))
    }

    /// 启动后台健康检查任务
    pub fn start_background_check(
        self: Arc<Self>,
        status_callback: impl Fn(WorkerStatus) + Send + 'static,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            info!("Starting background health check loop");

            let mut last_status = WorkerStatus::Disconnected;

            loop {
                let current_status = self.check_worker_health().await;

                // 只有状态变化时才调用回调
                if current_status != last_status {
                    info!("Worker status changed: {:?} -> {:?}", last_status, current_status);
                    status_callback(current_status);
                    last_status = current_status;
                }

                sleep(self.check_interval).await;
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::net::UnixListener;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_worker_health_check() {
        let temp_dir = TempDir::new().unwrap();
        let socket_path = temp_dir.path().join("test_worker.sock");
        let uds_uri = format!("unix://{}", socket_path.display());

        let checker = HealthChecker::new(uds_uri.clone(), 1, 1, 3);

        // Worker 未启动
        assert_eq!(checker.check_worker_health().await, WorkerStatus::Disconnected);

        // 创建 socket
        let _listener = UnixListener::bind(&socket_path).unwrap();

        // Worker 已启动
        assert_eq!(checker.check_worker_health().await, WorkerStatus::Connected);
    }

    #[tokio::test]
    async fn test_wait_for_reconnect() {
        let temp_dir = TempDir::new().unwrap();
        let socket_path = temp_dir.path().join("test_worker.sock");
        let uds_uri = format!("unix://{}", socket_path.display());

        let checker = HealthChecker::new(uds_uri.clone(), 1, 1, 2);

        // 在后台延迟创建 socket
        let socket_path_clone = socket_path.clone();
        tokio::spawn(async move {
            sleep(Duration::from_millis(500)).await;
            let _listener = UnixListener::bind(&socket_path_clone).unwrap();
            // 保持 listener 存活
            sleep(Duration::from_secs(5)).await;
        });

        // 等待重连（应该成功）
        assert!(checker.wait_for_reconnect().await.is_ok());
    }
}
