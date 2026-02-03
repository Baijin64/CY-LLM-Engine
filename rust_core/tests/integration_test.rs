/// 集成测试 - Rust Sidecar 与 Python Worker UDS 通信
use rust_core::config::SidecarConfig;
use rust_core::health::{HealthChecker, WorkerStatus};
use std::os::unix::net::UnixListener;
use tempfile::TempDir;
use tokio::time::{sleep, Duration};

#[tokio::test]
async fn test_health_check_with_mock_worker() {
    // 创建临时目录用于 UDS socket
    let temp_dir = TempDir::new().unwrap();
    let socket_path = temp_dir.path().join("test_worker.sock");
    let uds_uri = format!("unix://{}", socket_path.display());

    let health_checker = HealthChecker::new(uds_uri.clone(), 1, 1, 3);

    // 初始状态：Worker 未启动
    assert_eq!(
        health_checker.check_worker_health().await,
        WorkerStatus::Disconnected
    );

    // 模拟 Worker 启动（创建 socket）
    let _listener = UnixListener::bind(&socket_path).unwrap();

    // Worker 已启动
    assert_eq!(
        health_checker.check_worker_health().await,
        WorkerStatus::Connected
    );
}

#[tokio::test]
async fn test_config_loading() {
    // 测试默认配置
    let config = SidecarConfig::default();
    assert_eq!(config.server.bind_addr, "0.0.0.0:50051");
    assert_eq!(config.server.worker_uds, "unix:///tmp/cy_worker.sock");

    // 验证配置
    assert!(config.validate().is_ok());
}

#[tokio::test]
async fn test_health_checker_reconnect() {
    let temp_dir = TempDir::new().unwrap();
    let socket_path = temp_dir.path().join("test_worker.sock");
    let uds_uri = format!("unix://{}", socket_path.display());

    let health_checker = HealthChecker::new(uds_uri.clone(), 1, 1, 2);

    // 在后台延迟创建 socket（模拟 Worker 重启）
    let socket_path_clone = socket_path.clone();
    tokio::spawn(async move {
        sleep(Duration::from_millis(500)).await;
        let _listener = UnixListener::bind(&socket_path_clone).unwrap();
        sleep(Duration::from_secs(5)).await; // 保持 listener 存活
    });

    // 等待重连（应该成功）
    let result = health_checker.wait_for_reconnect().await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_token_counter() {
    use rust_core::metering::TokenCounter;

    let counter = TokenCounter::new();

    let session_id = counter.start_session("qwen-7b".to_string(), Some("user123".to_string()));

    counter.add_tokens(&session_id, 10);
    counter.add_tokens(&session_id, 20);
    counter.add_tokens(&session_id, 30);

    let stats = counter.get_session_stats(&session_id).unwrap();
    assert_eq!(stats.0, "qwen-7b");
    assert_eq!(stats.1, 60);

    let usage = counter.end_session(&session_id).unwrap();
    assert_eq!(usage.tokens_generated, 60);
    assert_eq!(usage.model_id, "qwen-7b");
}

#[test]
fn test_metrics_collector() {
    use rust_core::metrics::{MetricsCollector, WorkerConnectionStatus};

    let metrics = MetricsCollector::new().unwrap();

    // 记录请求
    metrics.record_request(true, 0.5);
    metrics.record_request(false, 1.0);

    // 记录 Token
    metrics.record_tokens(100.0);

    // 更新 Worker 状态
    metrics.set_worker_status(WorkerConnectionStatus::Connected);

    // 验证指标
    assert_eq!(metrics.requests_total.get(), 2);
    assert_eq!(metrics.requests_success.get(), 1);
    assert_eq!(metrics.requests_failed.get(), 1);
    assert_eq!(metrics.worker_connection_status.get(), 1);
}
