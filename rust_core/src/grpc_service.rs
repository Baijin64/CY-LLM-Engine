/// gRPC 服务实现 - 将代理逻辑暴露为 gRPC 服务
use crate::proxy::{proto, WorkerProxy};
use std::sync::Arc;
use tonic::{Request, Response, Status, Streaming};
use tracing::{debug, error, info};

/// AiInference gRPC 服务实现
pub struct AiInferenceService {
    proxy: Arc<WorkerProxy>,
}

impl AiInferenceService {
    pub fn new(proxy: WorkerProxy) -> Self {
        Self {
            proxy: Arc::new(proxy),
        }
    }
}

#[tonic::async_trait]
impl proto::ai_inference_server::AiInference for AiInferenceService {
    type StreamPredictStream = std::pin::Pin<
        Box<
            dyn futures_core::Stream<Item = Result<proto::StreamPredictResponse, Status>>
                + Send
                + 'static,
        >,
    >;

    async fn stream_predict(
        &self,
        request: Request<Streaming<proto::StreamPredictRequest>>,
    ) -> Result<Response<Self::StreamPredictStream>, Status> {
        let trace_id = request
            .metadata()
            .get("trace-id")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("unknown")
            .to_string();

        info!("Received StreamPredict request, trace_id: {}", trace_id);
        debug!("Request metadata: {:?}", request.metadata());

        // 转发请求到 WorkerProxy
        match self.proxy.forward_stream_request(request).await {
            Ok(response) => {
                info!("Successfully forwarded request, trace_id: {}", trace_id);
                Ok(response)
            }
            Err(e) => {
                error!("Failed to forward request: {}, trace_id: {}", e, trace_id);
                
                // 根据错误类型返回合适的 gRPC Status
                let status = match e {
                    crate::errors::SidecarError::WorkerDisconnected(msg) => {
                        Status::unavailable(format!("Worker unavailable: {}", msg))
                    }
                    crate::errors::SidecarError::WorkerReloading(secs) => {
                        Status::unavailable(format!("Worker is reloading, retry in {}s", secs))
                    }
                    crate::errors::SidecarError::WorkerConnectionError(msg) => {
                        Status::internal(format!("Worker connection error: {}", msg))
                    }
                    _ => Status::internal(format!("Internal error: {}", e)),
                };
                
                Err(status)
            }
        }
    }

    async fn health(
        &self,
        request: Request<proto::WorkerHealthRequest>,
    ) -> Result<Response<proto::WorkerHealthResponse>, Status> {
        let trace_id = &request.get_ref().trace_id;
        debug!("Health check request, trace_id: {}", trace_id);

        // 简单返回健康状态
        // TODO: 从 WorkerProxy 获取实际的健康状态和指标
        let response = proto::WorkerHealthResponse {
            healthy: true,
            metrics: std::collections::HashMap::new(),
        };

        Ok(Response::new(response))
    }

    async fn control(
        &self,
        request: Request<proto::ControlMessage>,
    ) -> Result<Response<proto::ControlMessage>, Status> {
        let msg = request.get_ref();
        info!("Control message received: command={}, trace_id={}", 
              msg.command, msg.trace_id);

        // TODO: 实现控制命令处理逻辑
        // 目前只返回确认消息
        let response = proto::ControlMessage {
            trace_id: msg.trace_id.clone(),
            command: format!("ACK: {}", msg.command),
            payload: std::collections::HashMap::new(),
        };

        Ok(Response::new(response))
    }
}
