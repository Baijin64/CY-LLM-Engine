/// 错误类型定义 - 数据面代理
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SidecarError {
    /// Worker 连接错误
    #[error("Worker connection failed: {0}")]
    WorkerConnectionError(String),

    /// Worker 断开连接
    #[error("Worker disconnected: {0}")]
    WorkerDisconnected(String),

    /// Worker 正在重载
    #[error("Worker is reloading, please retry in {0}s")]
    WorkerReloading(u64),

    /// 配置错误
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Token 计数错误
    #[error("Token metering error: {0}")]
    MeteringError(String),

    /// 代理错误
    #[error("Proxy error: {0}")]
    ProxyError(String),

    /// 内部错误
    #[error("Internal error: {0}")]
    InternalError(String),

    /// IO 错误
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// gRPC 传输错误
    #[error("gRPC transport error: {0}")]
    TransportError(#[from] tonic::transport::Error),

    /// gRPC 状态错误
    #[error("gRPC status error: {0}")]
    GrpcStatus(#[from] tonic::Status),
}

pub type Result<T> = std::result::Result<T, SidecarError>;
