# 架构重构与开源版迁移待办列表 (TODO List)

本文档基于新的架构规划，旨在实现“核心+插件化”的模块化架构，并完成开源版（Community Edition）的基础组件开发。
*直接沿用现有的 `ai_service.proto` 作为组件间通信标准。*

## 1. 代码库重组 (Refactoring & Cleanup)
- [x] **剥离 Kotlin 组件 (Enterprise Move)**
    - [x] 将 `CY_LLM_Backend/gateway` (Kotlin) 移动到 `legacy_enterprise/gateway` 或归档文件夹。
    - [x] 将 `CY_LLM_Backend/coordinator` (Kotlin) 移动到 `legacy_enterprise/coordinator` 或归档文件夹。
    - [x] 更新根目录 `settings.gradle.kts` 移除上述模块引用。
- [x] **建立开源版目录结构**
    - [x] 创建 `CY_LLM_Backend/gateway_lite/` (Python/FastAPI)。
    - [x] 创建 `CY_LLM_Backend/coordinator_lite/` (Python)。

## 2. 开源版网关开发 (Gateway Lite)
*目标：轻量级 Python 网关，替换 Kotlin 网关，提供基础 API 转发。*
- [ ] **基础框架搭建**
    - [ ] 初始化 Python 项目，引入 FastAPI, uvicorn, grpcio-tools。
    - [ ] 编写 `Dockerfile.gateway.lite`。
    - [ ] **生成 gRPC 代码**：使用现有的 `CY_LLM_Backend/proto/ai_service.proto` 生成 Python 客户端代码。
- [ ] **接口实现**
    - [ ] 实现 `/v1/chat/completions` (OpenAI 兼容接口)。
    - [ ] 实现 `/health` 检查接口。
    - [ ] 实现 gRPC 客户端，将请求**直接透传**给 Coordinator。
- [ ] **鉴权与配置**
    - [ ] 实现基于静态 Token 的简单鉴权（配置文件）。
    - [ ] 支持通过环境变量配置 Coordinator 地址。

## 3. 开源版调度器开发 (Coordinator Lite)
*目标：直通/原型级 Python 调度器，不做复杂调度，仅做请求透传。*
- [ ] **基础框架搭建**
    - [ ] 初始化 Python 项目，引入 grpcio, grpcio-tools。
    - [ ] 编写 `Dockerfile.coordinator.lite`。
- [ ] **核心逻辑实现**
    - [ ] 实现 gRPC Server (`StreamPredict` 等接口)，复用 `ai_service.proto`。
    - [ ] 实现简单的 Worker 注册与发现机制 (或读取静态配置 `models.json`)。
    - [ ] 实现 Round-Robin 或 Random 简单的负载均衡策略 (或仅支持单 Worker 直连)。
    - [ ] 建立与 Worker 的 gRPC 连接通道。

## 4. 推理层插件化适配 (Worker Plugin Adaptation)
*目标：确保现有 Worker 能以标准插件形式工作。*
- [ ] **引擎加载机制检查**
    - [ ] 验证 `CY_LLM_Backend/worker/core/engine_factory.py` 是否能动态加载不同后端。
    - [ ] 确保 vLLM 和 TRT-LLM 依赖解耦（按需安装）。
- [ ] **配置标准化**
    - [ ] 统一通过环境变量或 ConfigMap 注入模型路径和引擎类型。

## 5. 集成与测试 (Integration & Testing)
- [ ] **容器编排更新**
    - [ ] 创建新的 `docker-compose.community.yml`。
    - [ ] 编排 `gateway_lite`, `coordinator_lite`, `worker` (vLLM/Dummy)。
- [ ] **端到端验证**
    - [ ] 使用 `curl` 或 Postman 测试 `gateway_lite` API。
    - [ ] 验证请求完整链路：Gateway -> Coordinator -> Worker -> Return。

## 6. 文档 (Documentation)
- [ ] 更新 `README.md` 说明开源版架构。
- [ ] 编写 `DEVELOPMENT_LITE.md` 指导如何运行轻量版。
