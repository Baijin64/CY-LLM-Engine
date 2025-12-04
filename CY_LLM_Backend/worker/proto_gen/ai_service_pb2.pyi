from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TrainingStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TRAINING_STATUS_UNKNOWN: _ClassVar[TrainingStatus]
    TRAINING_STATUS_QUEUED: _ClassVar[TrainingStatus]
    TRAINING_STATUS_RUNNING: _ClassVar[TrainingStatus]
    TRAINING_STATUS_COMPLETED: _ClassVar[TrainingStatus]
    TRAINING_STATUS_FAILED: _ClassVar[TrainingStatus]
    TRAINING_STATUS_CANCELLED: _ClassVar[TrainingStatus]
TRAINING_STATUS_UNKNOWN: TrainingStatus
TRAINING_STATUS_QUEUED: TrainingStatus
TRAINING_STATUS_RUNNING: TrainingStatus
TRAINING_STATUS_COMPLETED: TrainingStatus
TRAINING_STATUS_FAILED: TrainingStatus
TRAINING_STATUS_CANCELLED: TrainingStatus

class StreamMetadata(_message.Message):
    __slots__ = ("trace_id", "tenant", "player_id", "locale", "extra")
    class ExtraEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    TENANT_FIELD_NUMBER: _ClassVar[int]
    PLAYER_ID_FIELD_NUMBER: _ClassVar[int]
    LOCALE_FIELD_NUMBER: _ClassVar[int]
    EXTRA_FIELD_NUMBER: _ClassVar[int]
    trace_id: str
    tenant: str
    player_id: str
    locale: str
    extra: _containers.ScalarMap[str, str]
    def __init__(self, trace_id: _Optional[str] = ..., tenant: _Optional[str] = ..., player_id: _Optional[str] = ..., locale: _Optional[str] = ..., extra: _Optional[_Mapping[str, str]] = ...) -> None: ...

class GenerationParameters(_message.Message):
    __slots__ = ("max_new_tokens", "temperature", "top_p", "repetition_penalty")
    MAX_NEW_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    TOP_P_FIELD_NUMBER: _ClassVar[int]
    REPETITION_PENALTY_FIELD_NUMBER: _ClassVar[int]
    max_new_tokens: int
    temperature: float
    top_p: float
    repetition_penalty: float
    def __init__(self, max_new_tokens: _Optional[int] = ..., temperature: _Optional[float] = ..., top_p: _Optional[float] = ..., repetition_penalty: _Optional[float] = ...) -> None: ...

class StreamPredictRequest(_message.Message):
    __slots__ = ("model_id", "prompt", "adapter", "priority", "generation", "metadata", "worker_hint")
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    ADAPTER_FIELD_NUMBER: _ClassVar[int]
    PRIORITY_FIELD_NUMBER: _ClassVar[int]
    GENERATION_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    WORKER_HINT_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    prompt: str
    adapter: str
    priority: int
    generation: GenerationParameters
    metadata: StreamMetadata
    worker_hint: str
    def __init__(self, model_id: _Optional[str] = ..., prompt: _Optional[str] = ..., adapter: _Optional[str] = ..., priority: _Optional[int] = ..., generation: _Optional[_Union[GenerationParameters, _Mapping]] = ..., metadata: _Optional[_Union[StreamMetadata, _Mapping]] = ..., worker_hint: _Optional[str] = ...) -> None: ...

class StreamPredictResponse(_message.Message):
    __slots__ = ("trace_id", "chunk", "end_of_stream", "index")
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    CHUNK_FIELD_NUMBER: _ClassVar[int]
    END_OF_STREAM_FIELD_NUMBER: _ClassVar[int]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    trace_id: str
    chunk: str
    end_of_stream: bool
    index: int
    def __init__(self, trace_id: _Optional[str] = ..., chunk: _Optional[str] = ..., end_of_stream: bool = ..., index: _Optional[int] = ...) -> None: ...

class ControlMessage(_message.Message):
    __slots__ = ("trace_id", "command", "payload")
    class PayloadEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    COMMAND_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    trace_id: str
    command: str
    payload: _containers.ScalarMap[str, str]
    def __init__(self, trace_id: _Optional[str] = ..., command: _Optional[str] = ..., payload: _Optional[_Mapping[str, str]] = ...) -> None: ...

class WorkerHealthRequest(_message.Message):
    __slots__ = ("trace_id",)
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    trace_id: str
    def __init__(self, trace_id: _Optional[str] = ...) -> None: ...

class WorkerHealthResponse(_message.Message):
    __slots__ = ("healthy", "metrics")
    class MetricsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    HEALTHY_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    healthy: bool
    metrics: _containers.ScalarMap[str, str]
    def __init__(self, healthy: bool = ..., metrics: _Optional[_Mapping[str, str]] = ...) -> None: ...

class LoraConfig(_message.Message):
    __slots__ = ("r", "lora_alpha", "lora_dropout", "target_modules")
    R_FIELD_NUMBER: _ClassVar[int]
    LORA_ALPHA_FIELD_NUMBER: _ClassVar[int]
    LORA_DROPOUT_FIELD_NUMBER: _ClassVar[int]
    TARGET_MODULES_FIELD_NUMBER: _ClassVar[int]
    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, r: _Optional[int] = ..., lora_alpha: _Optional[int] = ..., lora_dropout: _Optional[float] = ..., target_modules: _Optional[_Iterable[str]] = ...) -> None: ...

class QuantizationConfig(_message.Message):
    __slots__ = ("use_4bit", "bnb_4bit_compute_dtype", "bnb_4bit_quant_type")
    USE_4BIT_FIELD_NUMBER: _ClassVar[int]
    BNB_4BIT_COMPUTE_DTYPE_FIELD_NUMBER: _ClassVar[int]
    BNB_4BIT_QUANT_TYPE_FIELD_NUMBER: _ClassVar[int]
    use_4bit: bool
    bnb_4bit_compute_dtype: str
    bnb_4bit_quant_type: str
    def __init__(self, use_4bit: bool = ..., bnb_4bit_compute_dtype: _Optional[str] = ..., bnb_4bit_quant_type: _Optional[str] = ...) -> None: ...

class TrainingHyperparams(_message.Message):
    __slots__ = ("num_train_epochs", "per_device_batch_size", "gradient_accumulation_steps", "learning_rate", "warmup_ratio", "max_seq_length", "save_steps", "logging_steps")
    NUM_TRAIN_EPOCHS_FIELD_NUMBER: _ClassVar[int]
    PER_DEVICE_BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    GRADIENT_ACCUMULATION_STEPS_FIELD_NUMBER: _ClassVar[int]
    LEARNING_RATE_FIELD_NUMBER: _ClassVar[int]
    WARMUP_RATIO_FIELD_NUMBER: _ClassVar[int]
    MAX_SEQ_LENGTH_FIELD_NUMBER: _ClassVar[int]
    SAVE_STEPS_FIELD_NUMBER: _ClassVar[int]
    LOGGING_STEPS_FIELD_NUMBER: _ClassVar[int]
    num_train_epochs: int
    per_device_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_ratio: float
    max_seq_length: int
    save_steps: int
    logging_steps: int
    def __init__(self, num_train_epochs: _Optional[int] = ..., per_device_batch_size: _Optional[int] = ..., gradient_accumulation_steps: _Optional[int] = ..., learning_rate: _Optional[float] = ..., warmup_ratio: _Optional[float] = ..., max_seq_length: _Optional[int] = ..., save_steps: _Optional[int] = ..., logging_steps: _Optional[int] = ...) -> None: ...

class DatasetConfig(_message.Message):
    __slots__ = ("path", "format", "setting_oversample")
    PATH_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    SETTING_OVERSAMPLE_FIELD_NUMBER: _ClassVar[int]
    path: str
    format: str
    setting_oversample: float
    def __init__(self, path: _Optional[str] = ..., format: _Optional[str] = ..., setting_oversample: _Optional[float] = ...) -> None: ...

class TrainingRequest(_message.Message):
    __slots__ = ("job_id", "base_model", "output_dir", "character_name", "dataset", "lora", "quantization", "hyperparams", "metadata")
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    BASE_MODEL_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_DIR_FIELD_NUMBER: _ClassVar[int]
    CHARACTER_NAME_FIELD_NUMBER: _ClassVar[int]
    DATASET_FIELD_NUMBER: _ClassVar[int]
    LORA_FIELD_NUMBER: _ClassVar[int]
    QUANTIZATION_FIELD_NUMBER: _ClassVar[int]
    HYPERPARAMS_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    base_model: str
    output_dir: str
    character_name: str
    dataset: DatasetConfig
    lora: LoraConfig
    quantization: QuantizationConfig
    hyperparams: TrainingHyperparams
    metadata: StreamMetadata
    def __init__(self, job_id: _Optional[str] = ..., base_model: _Optional[str] = ..., output_dir: _Optional[str] = ..., character_name: _Optional[str] = ..., dataset: _Optional[_Union[DatasetConfig, _Mapping]] = ..., lora: _Optional[_Union[LoraConfig, _Mapping]] = ..., quantization: _Optional[_Union[QuantizationConfig, _Mapping]] = ..., hyperparams: _Optional[_Union[TrainingHyperparams, _Mapping]] = ..., metadata: _Optional[_Union[StreamMetadata, _Mapping]] = ...) -> None: ...

class TrainingProgress(_message.Message):
    __slots__ = ("job_id", "status", "current_epoch", "total_epochs", "current_step", "total_steps", "progress_percent", "loss", "learning_rate", "gpu_memory_used", "gpu_utilization", "elapsed_seconds", "eta_seconds", "message", "error")
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    CURRENT_EPOCH_FIELD_NUMBER: _ClassVar[int]
    TOTAL_EPOCHS_FIELD_NUMBER: _ClassVar[int]
    CURRENT_STEP_FIELD_NUMBER: _ClassVar[int]
    TOTAL_STEPS_FIELD_NUMBER: _ClassVar[int]
    PROGRESS_PERCENT_FIELD_NUMBER: _ClassVar[int]
    LOSS_FIELD_NUMBER: _ClassVar[int]
    LEARNING_RATE_FIELD_NUMBER: _ClassVar[int]
    GPU_MEMORY_USED_FIELD_NUMBER: _ClassVar[int]
    GPU_UTILIZATION_FIELD_NUMBER: _ClassVar[int]
    ELAPSED_SECONDS_FIELD_NUMBER: _ClassVar[int]
    ETA_SECONDS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    status: TrainingStatus
    current_epoch: int
    total_epochs: int
    current_step: int
    total_steps: int
    progress_percent: float
    loss: float
    learning_rate: float
    gpu_memory_used: float
    gpu_utilization: float
    elapsed_seconds: float
    eta_seconds: float
    message: str
    error: str
    def __init__(self, job_id: _Optional[str] = ..., status: _Optional[_Union[TrainingStatus, str]] = ..., current_epoch: _Optional[int] = ..., total_epochs: _Optional[int] = ..., current_step: _Optional[int] = ..., total_steps: _Optional[int] = ..., progress_percent: _Optional[float] = ..., loss: _Optional[float] = ..., learning_rate: _Optional[float] = ..., gpu_memory_used: _Optional[float] = ..., gpu_utilization: _Optional[float] = ..., elapsed_seconds: _Optional[float] = ..., eta_seconds: _Optional[float] = ..., message: _Optional[str] = ..., error: _Optional[str] = ...) -> None: ...

class TrainingStatusRequest(_message.Message):
    __slots__ = ("job_id",)
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    def __init__(self, job_id: _Optional[str] = ...) -> None: ...

class TrainingStatusResponse(_message.Message):
    __slots__ = ("job_id", "status", "latest_progress", "output_path")
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    LATEST_PROGRESS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_PATH_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    status: TrainingStatus
    latest_progress: TrainingProgress
    output_path: str
    def __init__(self, job_id: _Optional[str] = ..., status: _Optional[_Union[TrainingStatus, str]] = ..., latest_progress: _Optional[_Union[TrainingProgress, _Mapping]] = ..., output_path: _Optional[str] = ...) -> None: ...

class CancelTrainingRequest(_message.Message):
    __slots__ = ("job_id",)
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    def __init__(self, job_id: _Optional[str] = ...) -> None: ...

class CancelTrainingResponse(_message.Message):
    __slots__ = ("job_id", "success", "message")
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    success: bool
    message: str
    def __init__(self, job_id: _Optional[str] = ..., success: bool = ..., message: _Optional[str] = ...) -> None: ...

class ListTrainingJobsRequest(_message.Message):
    __slots__ = ("status_filter", "limit")
    STATUS_FILTER_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    status_filter: _containers.RepeatedScalarFieldContainer[TrainingStatus]
    limit: int
    def __init__(self, status_filter: _Optional[_Iterable[_Union[TrainingStatus, str]]] = ..., limit: _Optional[int] = ...) -> None: ...

class TrainingJobSummary(_message.Message):
    __slots__ = ("job_id", "status", "base_model", "character_name", "progress_percent", "created_at", "updated_at")
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    BASE_MODEL_FIELD_NUMBER: _ClassVar[int]
    CHARACTER_NAME_FIELD_NUMBER: _ClassVar[int]
    PROGRESS_PERCENT_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    UPDATED_AT_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    status: TrainingStatus
    base_model: str
    character_name: str
    progress_percent: float
    created_at: str
    updated_at: str
    def __init__(self, job_id: _Optional[str] = ..., status: _Optional[_Union[TrainingStatus, str]] = ..., base_model: _Optional[str] = ..., character_name: _Optional[str] = ..., progress_percent: _Optional[float] = ..., created_at: _Optional[str] = ..., updated_at: _Optional[str] = ...) -> None: ...

class ListTrainingJobsResponse(_message.Message):
    __slots__ = ("jobs",)
    JOBS_FIELD_NUMBER: _ClassVar[int]
    jobs: _containers.RepeatedCompositeFieldContainer[TrainingJobSummary]
    def __init__(self, jobs: _Optional[_Iterable[_Union[TrainingJobSummary, _Mapping]]] = ...) -> None: ...

class CustomScriptExecutionRequest(_message.Message):
    __slots__ = ("job_id", "script_path", "args", "env", "working_dir", "timeout_seconds", "metadata")
    class EnvEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    SCRIPT_PATH_FIELD_NUMBER: _ClassVar[int]
    ARGS_FIELD_NUMBER: _ClassVar[int]
    ENV_FIELD_NUMBER: _ClassVar[int]
    WORKING_DIR_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_SECONDS_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    script_path: str
    args: _containers.RepeatedScalarFieldContainer[str]
    env: _containers.ScalarMap[str, str]
    working_dir: str
    timeout_seconds: int
    metadata: StreamMetadata
    def __init__(self, job_id: _Optional[str] = ..., script_path: _Optional[str] = ..., args: _Optional[_Iterable[str]] = ..., env: _Optional[_Mapping[str, str]] = ..., working_dir: _Optional[str] = ..., timeout_seconds: _Optional[int] = ..., metadata: _Optional[_Union[StreamMetadata, _Mapping]] = ...) -> None: ...

class CustomScriptExecutionProgress(_message.Message):
    __slots__ = ("job_id", "status", "stdout_lines", "stderr_lines", "return_code", "elapsed_seconds", "metrics", "error")
    class MetricsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    STDOUT_LINES_FIELD_NUMBER: _ClassVar[int]
    STDERR_LINES_FIELD_NUMBER: _ClassVar[int]
    RETURN_CODE_FIELD_NUMBER: _ClassVar[int]
    ELAPSED_SECONDS_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    status: str
    stdout_lines: _containers.RepeatedScalarFieldContainer[str]
    stderr_lines: _containers.RepeatedScalarFieldContainer[str]
    return_code: int
    elapsed_seconds: float
    metrics: _containers.ScalarMap[str, str]
    error: str
    def __init__(self, job_id: _Optional[str] = ..., status: _Optional[str] = ..., stdout_lines: _Optional[_Iterable[str]] = ..., stderr_lines: _Optional[_Iterable[str]] = ..., return_code: _Optional[int] = ..., elapsed_seconds: _Optional[float] = ..., metrics: _Optional[_Mapping[str, str]] = ..., error: _Optional[str] = ...) -> None: ...

class CancelCustomScriptRequest(_message.Message):
    __slots__ = ("job_id",)
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    def __init__(self, job_id: _Optional[str] = ...) -> None: ...

class CancelCustomScriptResponse(_message.Message):
    __slots__ = ("job_id", "success", "message")
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    success: bool
    message: str
    def __init__(self, job_id: _Optional[str] = ..., success: bool = ..., message: _Optional[str] = ...) -> None: ...
