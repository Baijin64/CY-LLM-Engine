# Design Document: CY-LLM Engine Optimization

## Overview

本设计文档详细描述 CY-LLM Engine 的优化方案，包括 OpenAI 兼容 API、训练模块优化、推理性能优化、运行时体验改进、监控可观测性以及企业级安全功能的技术实现。

设计遵循现有的三层架构（Gateway → Coordinator → Worker），在保持向后兼容的同时引入新功能。

## Architecture

### 整体架构增强

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Gateway (Kotlin)                                 │
│  Spring WebFlux + gRPC Client                                           │
│  ├─ OpenAiController        # NEW: OpenAI 兼容 API 端点                  │
│  ├─ InferenceController     # 现有推理接口                               │
│  ├─ TrainingController      # 现有训练接口                               │
│  ├─ ApiKeyFilter            # 增强: 多租户 + 速率限制                    │
│  ├─ RequestTracer           # NEW: 分布式追踪                            │
│  └─ AuditLogger             # 增强: 结构化审计日志                       │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │ gRPC (:50050)
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Coordinator (Kotlin)                                │
│  Spring Boot 3 + Redis + gRPC + Coroutines                              │
│  ├─ TaskQueueService        # 增强: 优先级队列 + 租户隔离                │
│  ├─ PromptCacheService      # 增强: 智能缓存驱逐                         │
│  ├─ WorkerPoolManager       # 增强: 健康检查 + 负载均衡                  │
│  ├─ TelemetryAggregator     # NEW: 指标聚合 + OpenTelemetry              │
│  └─ ConfigHotReloader       # NEW: 配置热重载                            │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │ gRPC (:50051)
         ┌──────────────────┴──────────────────┐
         ▼                                     ▼
┌─────────────────────────────┐     ┌─────────────────────────────┐
│      Worker (Python)        │     │      Worker (Python)        │
│  ├─ InferenceEngine         │     │  ├─ InferenceEngine         │
│  │   ├─ ContinuousBatcher   │     │  │   ├─ ContinuousBatcher   │
│  │   ├─ KVCacheManager      │     │  │   ├─ KVCacheManager      │
│  │   └─ LoRAHotSwapper      │     │  │   └─ LoRAHotSwapper      │
│  ├─ TrainingEngine          │     │  ├─ TrainingEngine          │
│  │   ├─ DatasetValidator    │     │  │   ├─ DatasetValidator    │
│  │   ├─ CheckpointManager   │     │  │   ├─ CheckpointManager   │
│  │   └─ EarlyStopMonitor    │     │  │   └─ EarlyStopMonitor    │
│  └─ TelemetryExporter       │     │  └─ TelemetryExporter       │
└─────────────────────────────┘     └─────────────────────────────┘
```

### OpenAI API 兼容层架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    OpenAI API Layer                              │
├─────────────────────────────────────────────────────────────────┤
│  /v1/chat/completions  ──┐                                      │
│  /v1/completions       ──┼──▶ OpenAiRequestMapper               │
│  /v1/models            ──┘         │                            │
│                                    ▼                            │
│                         ┌─────────────────────┐                 │
│                         │ InternalRequest     │                 │
│                         │ (统一内部格式)       │                 │
│                         └─────────┬───────────┘                 │
│                                   │                             │
│                                   ▼                             │
│                         ┌─────────────────────┐                 │
│                         │ InferenceService    │                 │
│                         └─────────┬───────────┘                 │
│                                   │                             │
│                                   ▼                             │
│                         ┌─────────────────────┐                 │
│                         │ OpenAiResponseMapper│                 │
│                         └─────────────────────┘                 │
└─────────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. OpenAI 兼容 API 组件

#### 1.1 OpenAiController (Kotlin)

```kotlin
// 位置: gateway/src/main/kotlin/com/cy/llm/controller/OpenAiController.kt

@RestController
@RequestMapping("/v1")
class OpenAiController(
    private val inferenceService: InferenceService,
    private val modelRegistry: ModelRegistry,
    private val requestMapper: OpenAiRequestMapper,
    private val responseMapper: OpenAiResponseMapper
) {
    @PostMapping("/chat/completions", produces = [MediaType.APPLICATION_JSON_VALUE])
    suspend fun chatCompletions(
        @Valid @RequestBody request: ChatCompletionRequest
    ): ResponseEntity<Any> {
        return if (request.stream == true) {
            streamChatCompletions(request)
        } else {
            syncChatCompletions(request)
        }
    }
    
    @PostMapping("/completions", produces = [MediaType.APPLICATION_JSON_VALUE])
    suspend fun completions(
        @Valid @RequestBody request: CompletionRequest
    ): ResponseEntity<Any>
    
    @GetMapping("/models")
    suspend fun listModels(): ModelListResponse
}
```

#### 1.2 OpenAI 数据模型

```kotlin
// ChatCompletionRequest
data class ChatCompletionRequest(
    val model: String,
    val messages: List<ChatMessage>,
    val temperature: Double? = 0.7,
    val top_p: Double? = 1.0,
    val n: Int? = 1,
    val stream: Boolean? = false,
    val stop: Any? = null,  // String or List<String>
    val max_tokens: Int? = null,
    val presence_penalty: Double? = 0.0,
    val frequency_penalty: Double? = 0.0,
    val user: String? = null
)

data class ChatMessage(
    val role: String,  // "system", "user", "assistant"
    val content: String,
    val name: String? = null
)

// ChatCompletionResponse
data class ChatCompletionResponse(
    val id: String,
    val `object`: String = "chat.completion",
    val created: Long,
    val model: String,
    val choices: List<ChatChoice>,
    val usage: UsageInfo
)

// Streaming chunk
data class ChatCompletionChunk(
    val id: String,
    val `object`: String = "chat.completion.chunk",
    val created: Long,
    val model: String,
    val choices: List<ChatChoiceDelta>
)
```

### 2. 训练模块优化组件

#### 2.1 DatasetValidator (Python)

```python
# 位置: worker/training/validation/dataset_validator.py

@dataclass
class ValidationResult:
    is_valid: bool
    row_count: int
    columns: List[str]
    errors: List[ValidationError]
    warnings: List[str]
    sample_rows: List[Dict[str, Any]]

class DatasetValidator:
    """数据集验证器"""
    
    SUPPORTED_FORMATS = ["jsonl", "json", "parquet", "csv"]
    REQUIRED_COLUMNS = {
        "instruction": ["instruction", "input", "output"],
        "conversation": ["conversations"],
        "sharegpt": ["conversations"],
    }
    
    def validate(
        self,
        path: str,
        format_type: str = "auto",
        sample_size: int = 5
    ) -> ValidationResult:
        """验证数据集格式和内容"""
        
    def detect_format(self, path: str) -> str:
        """自动检测数据集格式"""
        
    def validate_row(self, row: Dict, schema: str) -> List[ValidationError]:
        """验证单行数据"""
```

#### 2.2 CheckpointManager (Python)

```python
# 位置: worker/training/checkpoint/checkpoint_manager.py

@dataclass
class CheckpointInfo:
    path: str
    step: int
    epoch: float
    loss: float
    timestamp: datetime
    is_valid: bool
    metadata: Dict[str, Any]

class CheckpointManager:
    """检查点管理器"""
    
    def __init__(
        self,
        output_dir: str,
        max_checkpoints: int = 5,
        save_total_limit: int = 3
    ):
        self.output_dir = output_dir
        self.max_checkpoints = max_checkpoints
        
    def save_checkpoint(
        self,
        trainer,
        step: int,
        metrics: Dict[str, float]
    ) -> CheckpointInfo:
        """保存检查点"""
        
    def find_latest_checkpoint(self) -> Optional[CheckpointInfo]:
        """查找最新有效检查点"""
        
    def validate_checkpoint(self, path: str) -> bool:
        """验证检查点完整性"""
        
    def cleanup_old_checkpoints(self) -> List[str]:
        """清理旧检查点"""
```

#### 2.3 EarlyStopMonitor (Python)

```python
# 位置: worker/training/callbacks/early_stop.py

@dataclass
class EarlyStopConfig:
    patience: int = 3
    min_delta: float = 0.001
    metric: str = "eval_loss"
    mode: str = "min"  # "min" or "max"

class EarlyStopMonitor:
    """早停监控器"""
    
    def __init__(self, config: EarlyStopConfig):
        self.config = config
        self.best_value: Optional[float] = None
        self.counter: int = 0
        
    def should_stop(self, metrics: Dict[str, float]) -> bool:
        """检查是否应该早停"""
        
    def update(self, metrics: Dict[str, float]) -> EarlyStopStatus:
        """更新监控状态"""
```

### 3. 推理性能优化组件

#### 3.1 KVCacheManager (Python)

```python
# 位置: worker/engines/cache/kv_cache_manager.py

@dataclass
class CacheEntry:
    prefix_hash: str
    kv_cache: Any  # 实际的 KV cache tensor
    token_count: int
    last_access: float
    hit_count: int

class KVCacheManager:
    """KV Cache 管理器，支持前缀缓存"""
    
    def __init__(
        self,
        max_cache_size_gb: float = 2.0,
        eviction_policy: str = "lru"  # "lru", "lfu", "adaptive"
    ):
        self._cache: Dict[str, CacheEntry] = {}
        self._max_size = max_cache_size_gb * 1024 * 1024 * 1024
        
    def get_prefix_cache(self, prompt: str) -> Optional[Any]:
        """获取前缀缓存"""
        
    def store_prefix_cache(
        self,
        prompt: str,
        kv_cache: Any,
        token_count: int
    ) -> None:
        """存储前缀缓存"""
        
    def evict_if_needed(self) -> int:
        """按策略驱逐缓存"""
```

#### 3.2 LoRAHotSwapper (Python)

```python
# 位置: worker/engines/lora/hot_swapper.py

@dataclass
class LoRAAdapter:
    name: str
    path: str
    rank: int
    alpha: int
    loaded: bool = False
    last_used: float = 0.0

class LoRAHotSwapper:
    """LoRA 适配器热切换管理器"""
    
    def __init__(
        self,
        base_model: Any,
        max_loaded_adapters: int = 4
    ):
        self._base_model = base_model
        self._adapters: Dict[str, LoRAAdapter] = {}
        self._active_adapter: Optional[str] = None
        
    def register_adapter(
        self,
        name: str,
        path: str,
        preload: bool = False
    ) -> None:
        """注册 LoRA 适配器"""
        
    def switch_adapter(self, name: str) -> float:
        """切换到指定适配器，返回切换耗时"""
        
    def get_active_adapter(self) -> Optional[str]:
        """获取当前激活的适配器"""
```

### 4. 监控与可观测性组件

#### 4.1 TelemetryExporter (Python)

```python
# 位置: worker/core/telemetry_exporter.py

@dataclass
class RequestMetrics:
    trace_id: str
    model_id: str
    prompt_tokens: int
    completion_tokens: int
    queue_time_ms: float
    ttft_ms: float  # Time to First Token
    total_time_ms: float
    tokens_per_second: float
    cache_hit: bool
    status: str  # "success", "error", "timeout"

class TelemetryExporter:
    """遥测数据导出器"""
    
    def __init__(
        self,
        prometheus_port: int = 9090,
        otlp_endpoint: Optional[str] = None
    ):
        self._metrics = PrometheusMetrics()
        self._tracer = OTLPTracer(otlp_endpoint) if otlp_endpoint else None
        
    def record_request(self, metrics: RequestMetrics) -> None:
        """记录请求指标"""
        
    def record_training_step(
        self,
        job_id: str,
        step: int,
        loss: float,
        lr: float,
        throughput: float
    ) -> None:
        """记录训练步骤指标"""
        
    def export_prometheus(self) -> str:
        """导出 Prometheus 格式指标"""
```

#### 4.2 RequestTracer (Kotlin)

```kotlin
// 位置: gateway/src/main/kotlin/com/cy/llm/tracing/RequestTracer.kt

data class TraceContext(
    val traceId: String,
    val spanId: String,
    val parentSpanId: String?,
    val startTime: Instant,
    val attributes: MutableMap<String, String>
)

class RequestTracer(
    private val otlpExporter: OtlpExporter?
) {
    fun startTrace(request: ServerHttpRequest): TraceContext
    
    fun addSpan(
        context: TraceContext,
        name: String,
        attributes: Map<String, String> = emptyMap()
    ): String
    
    fun endSpan(context: TraceContext, spanId: String)
    
    fun endTrace(context: TraceContext, status: String)
}
```

### 5. 安全与多租户组件

#### 5.1 RateLimiter (Kotlin)

```kotlin
// 位置: gateway/src/main/kotlin/com/cy/llm/security/RateLimiter.kt

data class RateLimitConfig(
    val requestsPerMinute: Int = 60,
    val requestsPerHour: Int = 1000,
    val tokensPerMinute: Int = 100000,
    val burstSize: Int = 10
)

class RateLimiter(
    private val redisTemplate: ReactiveRedisTemplate<String, String>
) {
    suspend fun checkLimit(
        apiKey: String,
        tenant: String?,
        estimatedTokens: Int
    ): RateLimitResult
    
    suspend fun recordUsage(
        apiKey: String,
        tenant: String?,
        actualTokens: Int
    )
}

data class RateLimitResult(
    val allowed: Boolean,
    val remaining: Int,
    val resetAt: Instant,
    val retryAfter: Duration?
)
```

## Data Models

### OpenAI API 数据模型

```kotlin
// 完整的 OpenAI 兼容数据模型
// 位置: gateway/src/main/kotlin/com/cy/llm/model/OpenAiModels.kt

// === Chat Completions ===
data class ChatCompletionRequest(
    @field:NotBlank val model: String,
    @field:NotEmpty val messages: List<ChatMessage>,
    val temperature: Double? = 0.7,
    val top_p: Double? = 1.0,
    val n: Int? = 1,
    val stream: Boolean? = false,
    val stop: Any? = null,
    val max_tokens: Int? = null,
    val presence_penalty: Double? = 0.0,
    val frequency_penalty: Double? = 0.0,
    val logit_bias: Map<String, Double>? = null,
    val user: String? = null
)

data class ChatCompletionResponse(
    val id: String,
    val `object`: String = "chat.completion",
    val created: Long,
    val model: String,
    val choices: List<ChatChoice>,
    val usage: UsageInfo,
    val system_fingerprint: String? = null
)

data class ChatChoice(
    val index: Int,
    val message: ChatMessage,
    val finish_reason: String?  // "stop", "length", "content_filter"
)

data class UsageInfo(
    val prompt_tokens: Int,
    val completion_tokens: Int,
    val total_tokens: Int
)

// === Streaming ===
data class ChatCompletionChunk(
    val id: String,
    val `object`: String = "chat.completion.chunk",
    val created: Long,
    val model: String,
    val choices: List<ChatChoiceDelta>
)

data class ChatChoiceDelta(
    val index: Int,
    val delta: DeltaContent,
    val finish_reason: String?
)

data class DeltaContent(
    val role: String? = null,
    val content: String? = null
)

// === Models ===
data class ModelListResponse(
    val `object`: String = "list",
    val data: List<ModelInfo>
)

data class ModelInfo(
    val id: String,
    val `object`: String = "model",
    val created: Long,
    val owned_by: String
)

// === Errors ===
data class OpenAiError(
    val error: ErrorDetail
)

data class ErrorDetail(
    val message: String,
    val type: String,
    val param: String?,
    val code: String?
)
```

### 训练数据模型

```python
# 位置: worker/training/models.py

@dataclass
class TrainingConfig:
    """训练配置"""
    base_model: str
    output_dir: str
    dataset_path: str
    
    # LoRA 配置
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None
    
    # 训练超参数
    num_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    max_seq_length: int = 2048
    
    # 检查点配置
    save_steps: int = 100
    save_total_limit: int = 3
    resume_from_checkpoint: Optional[str] = None
    
    # 早停配置
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # 混合精度
    fp16: bool = False
    bf16: bool = True
    
    # 梯度检查点
    gradient_checkpointing: bool = True

@dataclass
class TrainingMetrics:
    """训练指标"""
    job_id: str
    step: int
    epoch: float
    loss: float
    learning_rate: float
    grad_norm: float
    gpu_memory_gb: float
    gpu_utilization: float
    samples_per_second: float
    tokens_per_second: float
    elapsed_seconds: float
    eta_seconds: float
```

## Error Handling

### 错误码体系

```kotlin
// 位置: gateway/src/main/kotlin/com/cy/llm/error/ErrorCodes.kt

enum class ErrorCode(
    val code: String,
    val httpStatus: Int,
    val message: String
) {
    // 认证错误 (401)
    INVALID_API_KEY("invalid_api_key", 401, "Invalid API key provided"),
    EXPIRED_API_KEY("expired_api_key", 401, "API key has expired"),
    
    // 授权错误 (403)
    RATE_LIMIT_EXCEEDED("rate_limit_exceeded", 429, "Rate limit exceeded"),
    QUOTA_EXCEEDED("quota_exceeded", 403, "Usage quota exceeded"),
    
    // 请求错误 (400)
    INVALID_REQUEST("invalid_request_error", 400, "Invalid request"),
    INVALID_MODEL("model_not_found", 400, "Model not found"),
    CONTEXT_LENGTH_EXCEEDED("context_length_exceeded", 400, "Context length exceeded"),
    
    // 服务器错误 (500)
    INTERNAL_ERROR("internal_error", 500, "Internal server error"),
    MODEL_LOAD_ERROR("model_load_error", 500, "Failed to load model"),
    GPU_OOM("gpu_out_of_memory", 500, "GPU out of memory"),
    
    // 服务不可用 (503)
    SERVICE_OVERLOADED("service_overloaded", 503, "Service is overloaded"),
    WORKER_UNAVAILABLE("worker_unavailable", 503, "No worker available")
}
```

### 结构化错误响应

```kotlin
// OpenAI 兼容错误格式
data class ApiError(
    val error: ApiErrorDetail
)

data class ApiErrorDetail(
    val message: String,
    val type: String,
    val param: String? = null,
    val code: String,
    val suggestion: String? = null,  // 扩展: 修复建议
    val trace_id: String? = null     // 扩展: 追踪 ID
)
```

## Testing Strategy

### 测试方法

本项目采用双重测试策略：
1. **单元测试**: 验证具体示例和边界情况
2. **属性测试**: 验证跨所有输入的通用属性

属性测试使用 Property-Based Testing (PBT) 框架：
- Kotlin: Kotest Property Testing
- Python: Hypothesis



## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: OpenAI API Response Schema Conformance

*For any* valid ChatCompletionRequest or CompletionRequest, the Gateway response SHALL conform to the OpenAI API response schema, including all required fields (`id`, `object`, `created`, `model`, `choices`, `usage`).

**Validates: Requirements 1.1, 1.2, 1.4**

### Property 2: Streaming Response Format Conformance

*For any* streaming request with `stream: true`, all response chunks SHALL be prefixed with `data: ` and contain valid JSON matching the ChatCompletionChunk schema, ending with `data: [DONE]`.

**Validates: Requirements 1.3**

### Property 3: Parameter Passthrough Integrity

*For any* request with OpenAI-compatible parameters (`temperature`, `top_p`, `max_tokens`, `stop`, `presence_penalty`, `frequency_penalty`), the parameters SHALL be correctly passed to the inference engine without modification.

**Validates: Requirements 1.6**

### Property 4: Error Response Format Conformance

*For any* invalid request, the Gateway SHALL return an error response matching OpenAI error schema with appropriate HTTP status code, error type, and error code.

**Validates: Requirements 1.7**

### Property 5: Message Role Parsing Correctness

*For any* messages array containing `system`, `user`, and `assistant` roles, the Gateway SHALL correctly parse and preserve the role and content of each message in the order provided.

**Validates: Requirements 1.9**

### Property 6: Dataset Validation Completeness

*For any* dataset file, the DatasetValidator SHALL detect format errors, missing required columns, and invalid row data, returning a ValidationResult with all errors enumerated.

**Validates: Requirements 2.1**

### Property 7: Checkpoint Detection and Loading

*For any* output directory containing valid checkpoints, the CheckpointManager SHALL correctly identify and return the latest checkpoint by step number, and validate its integrity before loading.

**Validates: Requirements 2.4**

### Property 8: Training Queue Priority Ordering

*For any* set of queued training jobs with different priorities, jobs with higher priority SHALL be dequeued and executed before jobs with lower priority.

**Validates: Requirements 2.7**

### Property 9: Early Stopping Trigger Correctness

*For any* sequence of validation losses, the EarlyStopMonitor SHALL trigger early stopping if and only if the loss has not improved by `min_delta` for `patience` consecutive evaluations.

**Validates: Requirements 2.10**

### Property 10: Prefix Cache Hit Consistency

*For any* prompt that has been previously processed with caching enabled, subsequent requests with the same prompt prefix SHALL result in a cache hit and return consistent results.

**Validates: Requirements 3.2**

### Property 11: Latency Metrics Completeness

*For any* completed inference request, the Telemetry_Service SHALL record and report queue_time_ms, ttft_ms (time to first token), and total_time_ms, all with non-negative values.

**Validates: Requirements 3.5**

### Property 12: LoRA Adapter Switching Isolation

*For any* LoRA adapter switch operation, the switch SHALL complete without reloading the base model, and subsequent inference SHALL use only the newly activated adapter's weights.

**Validates: Requirements 3.6**

### Property 13: Request Priority Processing Order

*For any* set of concurrent requests with different priorities, requests with higher priority SHALL have lower average queue time than requests with lower priority.

**Validates: Requirements 3.7**

### Property 14: GPU OOM Error Suggestions

*For any* GPU out-of-memory error, the error response SHALL include at least one actionable suggestion (e.g., reduce batch size, enable quantization, reduce max_model_len).

**Validates: Requirements 3.8**

### Property 15: Percentile Latency Calculation Correctness

*For any* set of recorded latencies, the calculated P50, P95, and P99 values SHALL satisfy: P50 ≤ P95 ≤ P99, and each percentile SHALL be within the range of recorded values.

**Validates: Requirements 3.9**

### Property 16: Cache Eviction Policy Correctness

*For any* cache at capacity, the eviction policy SHALL remove entries based on the configured strategy (LRU: least recently accessed, LFU: least frequently accessed).

**Validates: Requirements 3.10**

### Property 17: Structured Error Message Format

*For any* error condition, the error response SHALL include: error_code (string), message (string), and optionally suggestion (string) and trace_id (string).

**Validates: Requirements 4.2**

### Property 18: Configuration Validation Completeness

*For any* invalid configuration, the System SHALL report all validation errors with specific field names and expected formats before startup fails.

**Validates: Requirements 4.4**

### Property 19: Timeout Error Context

*For any* request timeout, the error response SHALL indicate which stage timed out (queue, inference, streaming) and the configured timeout value.

**Validates: Requirements 4.6**

### Property 20: Trace ID Propagation

*For any* request with a trace_id, the same trace_id SHALL appear in logs and metrics across Gateway, Coordinator, and Worker components.

**Validates: Requirements 4.7**

### Property 21: Prometheus Metrics Format Conformance

*For any* metrics export, the output SHALL conform to Prometheus text format specification with valid metric names, labels, and values.

**Validates: Requirements 5.1**

### Property 22: Request Metrics Recording Completeness

*For any* completed request, the Telemetry_Service SHALL record: model_id, prompt_tokens, completion_tokens, duration_ms, and status.

**Validates: Requirements 5.2, 5.8**

### Property 23: Alert Threshold Triggering

*For any* metric exceeding its configured threshold (GPU utilization, queue depth), the Telemetry_Service SHALL emit exactly one warning alert until the metric returns below threshold.

**Validates: Requirements 5.4, 5.6**

### Property 24: Cache Hit Rate Calculation

*For any* sequence of cache operations, the reported cache_hit_rate SHALL equal (cache_hits / total_requests) * 100, rounded to two decimal places.

**Validates: Requirements 5.5**

### Property 25: Audit Log Completeness

*For any* API request, an audit log entry SHALL be created containing: timestamp, trace_id, api_key_hash, endpoint, method, status_code, and duration_ms.

**Validates: Requirements 5.10, 6.10**

### Property 26: API Key Authentication Correctness

*For any* request, authentication SHALL succeed if and only if the provided API key exists in the registry and has not expired.

**Validates: Requirements 1.8, 6.1**

### Property 27: Tenant Metrics Isolation

*For any* request with a tenant identifier, metrics SHALL be recorded with the tenant label, and aggregated metrics SHALL correctly separate values by tenant.

**Validates: Requirements 6.2**

### Property 28: Rate Limit Enforcement

*For any* API key or tenant, requests exceeding the configured rate limit SHALL be rejected with HTTP 429 and a Retry-After header indicating when to retry.

**Validates: Requirements 6.3**

### Property 29: Input Sanitization Effectiveness

*For any* input containing potential injection patterns (SQL, script tags, control characters), the Gateway SHALL sanitize or reject the input before processing.

**Validates: Requirements 6.6**

### Property 30: Sensitive Data Redaction

*For any* log entry containing API keys, passwords, or tokens, the sensitive values SHALL be redacted or masked (e.g., `sk-...xxxx`).

**Validates: Requirements 6.7**

### Property 31: Request Size Limit Enforcement

*For any* request exceeding the configured size limit, the Gateway SHALL reject the request with HTTP 413 before processing.

**Validates: Requirements 6.8**

### Property 32: Multi-Level Timeout Enforcement

*For any* request, timeouts SHALL be enforced at Gateway level (total), Coordinator level (queue + routing), and Worker level (inference), with the most specific timeout taking precedence.

**Validates: Requirements 6.9**

