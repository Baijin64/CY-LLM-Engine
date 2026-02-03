/// gRPC 代理核心逻辑 - Worker 数据面代理
use crate::config::SidecarConfig;
use crate::errors::{Result, SidecarError};
use crate::health::{HealthChecker, WorkerStatus};
use crate::metering::TokenCounter;
use crate::metrics::{MetricsCollector, WorkerConnectionStatus};
use std::sync::Arc;
use std::pin::Pin;
use futures_core::Stream;
use tokio::sync::RwLock;
use tonic::transport::{Channel, Uri};
use tracing::{debug, error, info, warn};

// Generated protobuf code will be included here
pub mod proto {
    tonic::include_proto!("cy.llm");
}

use proto::ai_inference_client::AiInferenceClient;
use proto::{StreamPredictRequest, StreamPredictResponse};

/// Worker 代理服务
pub struct WorkerProxy {
    /// 配置
    config: Arc<SidecarConfig>,

    /// Worker gRPC 客户端
    worker_client: Arc<RwLock<Option<AiInferenceClient<Channel>>>>,

    /// Token 计数器
    token_counter: Arc<TokenCounter>,

    /// 指标收集器
    metrics: Arc<MetricsCollector>,

    /// 健康检查器
    health_checker: Arc<HealthChecker>,

    /// 当前 Worker 状态 (使用 std 锁以支持同步回调且避免 runtime 阻塞)
    worker_status: Arc<std::sync::RwLock<WorkerStatus>>,
}

impl WorkerProxy {
    /// 创建新的代理实例
    pub fn new(config: SidecarConfig) -> Result<Self> {
        let config = Arc::new(config);

        let health_checker = Arc::new(HealthChecker::new(
            config.server.worker_uds.clone(),
            config.health.check_interval_secs,
            config.health.reconnect_delay_secs,
            config.health.max_reconnect_attempts,
        ));

        let token_counter = Arc::new(TokenCounter::new());
        let metrics = Arc::new(MetricsCollector::new().map_err(|e| {
            SidecarError::InternalError(format!("Failed to create metrics collector: {}", e))
        })?);

        Ok(Self {
            config,
            worker_client: Arc::new(RwLock::new(None)),
            token_counter,
            metrics,
            health_checker,
            worker_status: Arc::new(std::sync::RwLock::new(WorkerStatus::Disconnected)),
        })
    }

    /// 连接到 Worker
    pub async fn connect_to_worker(&self) -> Result<()> {
        info!("Connecting to Worker at {}", self.config.server.worker_uds);

        // 检查 Worker 健康状态
        let status = self.health_checker.check_worker_health().await;
        if status != WorkerStatus::Connected {
            return Err(SidecarError::WorkerDisconnected(
                "Worker is not available".to_string(),
            ));
        }

        // 连接到 Worker UDS
        let uri = self.config.server.worker_uds.parse::<Uri>().map_err(|e| {
            SidecarError::ConfigError(format!("Invalid worker_uds URI: {}", e))
        })?;

        let channel = Channel::builder(uri)
            .connect()
            .await
            .map_err(|e| SidecarError::WorkerConnectionError(format!("Failed to connect: {}", e)))?;

        let client = AiInferenceClient::new(channel);

        // 更新客户端
        let mut worker_client = self.worker_client.write().await;
        *worker_client = Some(client);

        // 更新状态
        {
            let mut worker_status = self.worker_status.write().unwrap();
            *worker_status = WorkerStatus::Connected;
        }
        self.metrics.set_worker_status(WorkerConnectionStatus::Connected);

        info!("Successfully connected to Worker");
        Ok(())
    }

    /// 转发流式推理请求到 Worker
    pub async fn forward_stream_request(
        &self,
        request: tonic::Request<tonic::Streaming<StreamPredictRequest>>,
    ) -> Result<tonic::Response<Pin<Box<dyn Stream<Item = std::result::Result<StreamPredictResponse, tonic::Status>> + Send + 'static>>>> {
        let start_time = std::time::Instant::now();

        // 检查 Worker 连接状态
        let status = *self.worker_status.read().unwrap();
        match status {
            WorkerStatus::Disconnected => {
                self.metrics.worker_connection_errors.inc();
                return Err(SidecarError::WorkerDisconnected(
                    "Worker is not connected".to_string(),
                ));
            }
            WorkerStatus::Reloading => {
                return Err(SidecarError::WorkerReloading(30));
            }
            WorkerStatus::Connected => {}
        }

        // 获取 Worker 客户端
        let worker_client = self.worker_client.read().await;
        let mut client = match worker_client.as_ref() {
            Some(client) => client.clone(),
            None => {
                self.metrics.worker_connection_errors.inc();
                return Err(SidecarError::WorkerDisconnected(
                    "Worker client not initialized".to_string(),
                ));
            }
        };
        drop(worker_client);

        // 转发请求到 Worker
        debug!("Forwarding stream request to Worker");
        self.metrics.active_connections.inc();

        // 将入站流的 `Result<Message, Status>` 转为仅包含 `Message` 的 stream，转发给 Worker
        let mut client_inbound = request.into_inner();
        let (in_tx, in_rx) = tokio::sync::mpsc::channel::<StreamPredictRequest>(128);

        // 将客户端消息转发到 worker 请求流
        tokio::spawn(async move {
            while let Ok(opt) = client_inbound.message().await {
                match opt {
                    Some(msg) => {
                        if in_tx.send(msg).await.is_err() {
                            warn!("Worker input channel closed");
                            break;
                        }
                    }
                    None => break,
                }
            }
        });

        // 使用 ReceiverStream 作为要发送到 Worker 的流
        let worker_request_stream = tokio_stream::wrappers::ReceiverStream::new(in_rx);

        let response = client
            .stream_predict(tonic::Request::new(worker_request_stream))
            .await
            .map_err(|e| {
                self.metrics.requests_failed.inc();
                self.metrics.active_connections.dec();
                error!("Failed to forward request to Worker: {}", e);
                SidecarError::ProxyError(format!("Worker request failed: {}", e))
            })?;

        // 包装响应流以进行 Token 计数
        let mut worker_inbound = response.into_inner();
        let token_counter = Arc::clone(&self.token_counter);
        let metrics = Arc::clone(&self.metrics);
        let session_id = uuid::Uuid::new_v4().to_string();

        // 启动会话
        let model_id = "unknown".to_string(); // 从请求中提取
        token_counter.start_session(model_id, None);

        let (tx, rx) = tokio::sync::mpsc::channel::<std::result::Result<StreamPredictResponse, tonic::Status>>(128);

        // 在后台任务中处理 worker 返回流
        tokio::spawn(async move {
            let mut total_tokens = 0u64;

            while let Ok(Some(chunk)) = worker_inbound.message().await {
                // 计数 Token（假设每个 chunk 包含 1 个 token）
                // TODO: 更准确的 token 计数逻辑
                let token_count = chunk.chunk.len() as u64;
                total_tokens += token_count;
                token_counter.add_tokens(&session_id, token_count);

                // 发送到客户端
                if tx.send(Ok(chunk)).await.is_err() {
                    warn!("Client disconnected");
                    break;
                }
            }

            // 记录总 Token 数
            metrics.record_tokens(total_tokens as f64);
            metrics.active_connections.dec();

            info!("Stream completed, total tokens: {}", total_tokens);
        });

        let outbound_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        let boxed_stream: Pin<Box<dyn Stream<Item = std::result::Result<StreamPredictResponse, tonic::Status>> + Send + 'static>> = Box::pin(outbound_stream);

        // 记录请求成功
        let duration = start_time.elapsed();
        self.metrics.record_request(true, duration.as_secs_f64());

        Ok(tonic::Response::new(boxed_stream))
    }

    /// 启动后台健康检查
    pub fn start_health_check(&self) {
        let metrics = Arc::clone(&self.metrics);
        let worker_status = Arc::clone(&self.worker_status);

        self.health_checker.clone().start_background_check(move |status| {
            {
                let mut worker_status = worker_status.write().unwrap();
                *worker_status = status;
            }

            let metrics_status = match status {
                WorkerStatus::Connected => WorkerConnectionStatus::Connected,
                WorkerStatus::Disconnected => WorkerConnectionStatus::Disconnected,
                WorkerStatus::Reloading => WorkerConnectionStatus::Reloading,
            };

            metrics.set_worker_status(metrics_status);
        });
    }

    /// 获取指标收集器
    pub fn metrics(&self) -> Arc<MetricsCollector> {
        Arc::clone(&self.metrics)
    }

    /// 获取 Token 计数器
    pub fn token_counter(&self) -> Arc<TokenCounter> {
        Arc::clone(&self.token_counter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_proxy_creation() {
        let config = SidecarConfig::default();
        let proxy = WorkerProxy::new(config);
        assert!(proxy.is_ok());
    }
}
