/// Rust Sidecar 主程序
/// Worker 数据面代理 - 启动入口
use rust_core::{SidecarConfig, Result};
use std::net::SocketAddr;
use std::path::Path;
use tracing::{info, Level};
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // 初始化日志
    init_logging();

    info!("CY-LLM Rust Sidecar v{}", rust_core::VERSION);
    info!("Loading configuration...");

    // 加载配置
    let mut config = load_config()?;
    config.merge_env();
    config.validate()?;
    
    info!("Configuration loaded successfully");
    info!("- gRPC Server: {}", config.server.bind_addr);
    info!("- Worker UDS: {}", config.server.worker_uds);
    info!("- Metrics Port: {}", config.observability.metrics_port);

    // 解析监听地址
    let addr: SocketAddr = config.server.bind_addr
        .parse()
        .map_err(|e| rust_core::SidecarError::ConfigError(format!("Invalid bind address: {}", e)))?;

    info!("Initializing proxy service...");
    
    // 创建代理实例
    let proxy = rust_core::proxy::WorkerProxy::new(config.clone())?;
    
    // 启动健康检查
    info!("Starting background health checker...");
    proxy.start_health_check();
    
    // 启动 Metrics 服务器
    let metrics_addr = format!("0.0.0.0:{}", config.observability.metrics_port);
    let metrics = proxy.metrics();
    tokio::spawn(async move {
        info!("Starting Metrics server on {}...", metrics_addr);
        match start_metrics_server(&metrics_addr, metrics).await {
            Ok(_) => info!("Metrics server stopped"),
            Err(e) => tracing::error!("Metrics server error: {}", e),
        }
    });
    
    // 启动 gRPC 服务器
    info!("Starting gRPC server on {}...", addr);
    let grpc_server = start_grpc_server(addr, proxy);
    
    info!("Sidecar is ready to accept connections");
    info!("- gRPC: {}", addr);
    info!("- Metrics: http://0.0.0.0:{}/metrics", config.observability.metrics_port);

    // 等待服务器或 Ctrl+C
    tokio::select! {
        result = grpc_server => {
            if let Err(e) = result {
                tracing::error!("gRPC server error: {}", e);
            }
        }
        _ = tokio::signal::ctrl_c() => {
            info!("Received shutdown signal");
        }
    }

    info!("Shutting down gracefully...");
    Ok(())
}

/// 启动 gRPC 服务器
async fn start_grpc_server(
    addr: SocketAddr,
    proxy: rust_core::proxy::WorkerProxy,
) -> Result<()> {
    use tonic::transport::Server;
    use rust_core::grpc_service::AiInferenceService;
    use rust_core::proxy::proto::ai_inference_server::AiInferenceServer;
    
    info!("gRPC server binding to {}", addr);
    
    let service = AiInferenceService::new(proxy);
    
    Server::builder()
        .add_service(AiInferenceServer::new(service))
        .serve(addr)
        .await
        .map_err(|e| rust_core::SidecarError::InternalError(format!("gRPC server error: {}", e)))?;
    
    info!("gRPC server stopped");
    Ok(())
}

/// 启动 Metrics HTTP 服务器
async fn start_metrics_server(
    addr: &str,
    metrics: std::sync::Arc<rust_core::metrics::MetricsCollector>,
) -> Result<()> {
    use warp::Filter;
    use prometheus::Encoder;
    
    let addr: SocketAddr = addr
        .parse()
        .map_err(|e| rust_core::SidecarError::ConfigError(format!("Invalid metrics address: {}", e)))?;
    
    let metrics_route = warp::path!("metrics")
        .map(move || {
            let encoder = prometheus::TextEncoder::new();
            let metric_families = metrics.registry().gather();
            let mut buffer = Vec::new();
            if let Err(e) = encoder.encode(&metric_families, &mut buffer) {
                tracing::error!("Failed to encode metrics: {}", e);
                return warp::reply::with_header(
                    Vec::new(),
                    "Content-Type",
                    "text/plain; version=0.0.4",
                );
            }
            warp::reply::with_header(
                buffer,
                "Content-Type",
                "text/plain; version=0.0.4",
            )
        });
    
    info!("Metrics server listening on http://{}/metrics", addr);
    warp::serve(metrics_route).run(addr).await;
    
    Ok(())
}

/// 加载配置文件
fn load_config() -> Result<SidecarConfig> {
    // 从环境变量或默认路径加载配置
    let config_path = std::env::var("SIDECAR_CONFIG")
        .unwrap_or_else(|_| "sidecar.toml".to_string());
    
    let path = Path::new(&config_path);
    
    if path.exists() {
        SidecarConfig::from_file(path)
    } else {
        // 使用默认配置
        Ok(SidecarConfig::default())
    }
}

/// 初始化日志系统
fn init_logging() {
    let log_level = std::env::var("SIDECAR_LOG_LEVEL")
        .unwrap_or_else(|_| "info".to_string());

    let level = match log_level.to_lowercase().as_str() {
        "trace" => Level::TRACE,
        "debug" => Level::DEBUG,
        "info" => Level::INFO,
        "warn" => Level::WARN,
        "error" => Level::ERROR,
        _ => Level::INFO,
    };

    tracing_subscriber::fmt()
        .with_max_level(level)
        .with_target(false)
        .with_thread_ids(true)
        .with_line_number(true)
        .init();
}
