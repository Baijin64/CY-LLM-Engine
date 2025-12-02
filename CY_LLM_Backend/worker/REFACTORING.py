# Worker 精简说明
# ==================
# 
# 随着 Coordinator (Kotlin) 层的引入，以下功能已迁移：
#
# ┌─────────────────────────────────────────────────────────────────┐
# │  已迁移到 Coordinator (Kotlin)                                   │
# ├─────────────────────────────────────────────────────────────────┤
# │  ❌ cache/prompt_cache.py   → coordinator/cache/               │
# │  ❌ core/task_scheduler.py  → coordinator/queue/               │  
# │  ❌ core/telemetry.py       → coordinator/telemetry/           │
# └─────────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────┐
# │  Worker 保留的核心功能 (Python - GPU 计算)                        │
# ├─────────────────────────────────────────────────────────────────┤
# │  ✅ engines/                 # 推理引擎（vLLM/TRT/MindIE）       │
# │  ✅ training/                # 训练引擎（LoRA/PEFT）            │
# │  ✅ core/memory_manager.py   # GPU 内存管理                     │
# │  ✅ core/server.py           # gRPC Server（精简版）            │
# │  ✅ grpc_servicer.py         # gRPC 服务实现                    │
# │  ✅ config/                  # 配置加载                         │
# └─────────────────────────────────────────────────────────────────┘
#
# 架构演进：
#
#   Gateway (Kotlin)
#        │
#        ▼ gRPC
#   Coordinator (Kotlin)
#   ├─ TaskQueue (Redis)
#   ├─ PromptCache (Redis)
#   ├─ WorkerPool
#   └─ Telemetry
#        │
#        ▼ gRPC (精简协议)
#   Worker (Python)
#   ├─ InferenceEngine (GPU)
#   ├─ TrainingEngine (GPU)
#   └─ GPUMemoryManager
#
# 迁移后的优势：
# 1. Worker 专注于 GPU 计算，更轻量
# 2. Coordinator 用 Kotlin 处理 IO 密集型任务，并发性能更好
# 3. 缓存和队列使用 Redis，支持分布式部署
# 4. 遥测集中聚合，减少 Worker 负担

# 标记已废弃的模块
DEPRECATED_MODULES = [
    "cache.prompt_cache",      # → coordinator/cache/PromptCacheService.kt
    "core.task_scheduler",     # → coordinator/queue/TaskQueueService.kt
    "core.telemetry",          # → coordinator/telemetry/TelemetryService.kt
]

# Worker 核心模块
CORE_MODULES = [
    "engines",                 # 推理引擎
    "training",                # 训练引擎
    "core.memory_manager",     # GPU 内存管理
    "core.server",             # gRPC Server
    "config",                  # 配置
]
