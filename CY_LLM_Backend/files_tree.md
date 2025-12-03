# CY_LLM_Backend 文件结构（自动生成占位说明）

以下为当前仓库 `CY_LLM_Backend/` 的文件树概览，文件均为占位注释或配置示例，便于快速浏览项目架构。

说明：旧的顶层脚本 `ai_backend.py` 已移除，替换为更稳健的 `scripts/diagnose_env.py`（位于项目根的 `scripts/` 目录中），并在 `CY_LLM_Backend` 内保留占位资源与部署文件。

CY_LLM_Backend/
├── proto/
│   └── ai_service.proto                 # [Core] gRPC 定义：StreamPredict、ControlMessage、Backpressure（占位）
├── gateway/ (Kotlin Spring Boot)
│   ├── build.gradle.kts                 # [构建] Gradle Kotlin DSL 占位
│   ├── Dockerfile.gateway               # [部署] Gateway 占位 Dockerfile（Java 21 LTS 基础镜像）
│   └── src/main/kotlin/com/cy/llm/
│       ├── Application.kt               # [启动] Spring Boot 启动类占位
│       ├── config/
│       │   ├── GrpcConfig.kt            # [gRPC 客户端] 占位
│       │   ├── ResilienceConfig.kt      # [Resilience] Resilience4j 占位
│       │   └── CircuitBreakerConfig.kt  # [Circuit Breaker] 断路器规则占位
│       ├── controller/
│       │   └── InferenceController.kt   # [边界] REST/WebSocket 接口占位
│       ├── service/
│       │   ├── ModelRegistry.kt         # [路由] 逻辑模型映射占位
│       │   └── InferenceService.kt      # [编排] 推理转发占位
│       └── model/
│           └── DomainModels.kt          # [DTO] 域模型占位
├── tests/
│   └── test_integration.py              # 交互式测试入口：目前包含引擎冒烟、内存管理与端到端集成三种模式
├── worker/ (Python PyTorch)
│   ├── main.py                          # [入口] Worker 启动脚本占位
│   ├── requirements.txt                 # [依赖] 通用占位
│   ├── requirements_ascend.txt          # [依赖] Ascend 专用占位
│   ├── Dockerfile.worker.nvidia         # [部署] Nvidia Worker 占位（CUDA 11.8 / Ubuntu 24.04 基础镜像注释）
│   ├── Dockerfile.worker.ascend         # [部署] Ascend Worker 占位（Ubuntu 24.04 基础镜像注释）
│   ├── config/
│   │   └── config_loader.py             # [配置] 硬件/配置检测占位
│   ├── core/
│   │   ├── memory_manager.py            # GPU LRU + 引用计数管理实现
│   │   ├── server.py                    # gRPC 双向流服务骨架
│   │   ├── task_scheduler.py            # 推理任务调度占位
│   │   └── telemetry.py                 # 指标与日志采集占位
│   ├── engines/
│   │   ├── abstract_engine.py           # 推理引擎抽象基类
│   │   ├── nvidia_engine.py             # CUDA 引擎实现（可选 4bit 量化 + LoRA 支持）
│   │   ├── ascend_engine.py             # 昇腾引擎占位
│   │   ├── hybrid_engine.py             # 多后端调度占位
│   │   └── engine_factory.py            # 引擎工厂与注册表占位
│   └── utils/
│       └── stream_buffer.py             # [流控] token/帧流缓冲占位
└── deploy/
    ├── config.json                      # [部署配置] 逻辑->物理映射示例（占位）
    ├── Dockerfile.gateway               # [部署] Gateway 占位（deploy 参考）
    ├── Dockerfile.worker.nvidia         # [部署] Nvidia Worker 占位（deploy 参考）
    ├── Dockerfile.worker.ascend         # [部署] Ascend Worker 占位（deploy 参考）
    └── docker-compose.yml               # [本地开发] Compose 占位

scripts/
└── diagnose_env.py                      # [工具] 新增的硬件诊断脚本：支持 CUDA 与 Ascend 检测与推理自检（位于仓库根）

说明：当前所有文件均为占位注释或配置样例；如需我继续生成关键文件的骨架（例如 `memory_manager.py` 方法签名、`ModelRegistry.kt` 接口或 `ai_service.proto` 的双向流定义），请告诉我优先级。