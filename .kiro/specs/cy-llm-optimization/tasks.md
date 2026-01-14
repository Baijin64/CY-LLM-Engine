# Implementation Plan: CY-LLM Engine Optimization

## Overview

本实现计划将 CY-LLM Engine 优化分为六个主要阶段，按优先级和依赖关系排序。每个阶段包含具体的编码任务和测试任务。

## Tasks

- [ ] 1. OpenAI 兼容 API 实现
  - 实现企业级 API 服务的核心功能
  - _Requirements: 1.1-1.10_

  - [ ] 1.1 创建 OpenAI 数据模型
    - 在 `gateway/src/main/kotlin/com/cy/llm/model/` 创建 `OpenAiModels.kt`
    - 实现 ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChunk 等数据类
    - 实现 CompletionRequest, CompletionResponse 数据类
    - 实现 ModelListResponse, ModelInfo 数据类
    - 实现 OpenAiError, ErrorDetail 数据类
    - _Requirements: 1.1, 1.2, 1.4, 1.7_

  - [ ] 1.2 实现请求/响应映射器
    - 创建 `gateway/src/main/kotlin/com/cy/llm/mapper/OpenAiMapper.kt`
    - 实现 ChatCompletionRequest → InferenceRequest 转换
    - 实现 InferenceChunk → ChatCompletionChunk 转换
    - 实现 messages 数组解析（system/user/assistant 角色）
    - 实现参数映射（temperature, top_p, max_tokens 等）
    - _Requirements: 1.5, 1.6, 1.9_

  - [ ] 1.3 编写映射器属性测试
    - **Property 3: Parameter Passthrough Integrity**
    - **Property 5: Message Role Parsing Correctness**
    - **Validates: Requirements 1.6, 1.9**

  - [ ] 1.4 实现 OpenAiController
    - 创建 `gateway/src/main/kotlin/com/cy/llm/controller/OpenAiController.kt`
    - 实现 POST `/v1/chat/completions` 端点（流式和非流式）
    - 实现 POST `/v1/completions` 端点
    - 实现 GET `/v1/models` 端点
    - 实现 SSE 流式响应格式（`data: {...}` + `data: [DONE]`）
    - _Requirements: 1.1, 1.2, 1.3, 1.10_

  - [ ] 1.5 编写 API 端点属性测试
    - **Property 1: OpenAI API Response Schema Conformance**
    - **Property 2: Streaming Response Format Conformance**
    - **Property 4: Error Response Format Conformance**
    - **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.7**

  - [ ] 1.6 实现错误处理和格式化
    - 创建 `gateway/src/main/kotlin/com/cy/llm/error/OpenAiErrorHandler.kt`
    - 实现 OpenAI 兼容错误响应格式
    - 实现错误码映射（内部错误码 → OpenAI 错误码）
    - 添加 GlobalExceptionHandler 支持
    - _Requirements: 1.7_

- [ ] 2. Checkpoint - 确保 OpenAI API 测试通过
  - 运行所有 OpenAI API 相关测试
  - 如有问题请询问用户

- [ ] 3. 训练模块优化
  - 增强训练引擎的可靠性和监控能力
  - _Requirements: 2.1-2.10_

  - [ ] 3.1 实现数据集验证器
    - 创建 `worker/training/validation/dataset_validator.py`
    - 实现格式自动检测（jsonl, json, parquet, csv）
    - 实现必需列验证（instruction/conversation/sharegpt 格式）
    - 实现行级数据验证
    - 返回详细的 ValidationResult
    - _Requirements: 2.1_

  - [ ] 3.2 编写数据集验证器属性测试
    - **Property 6: Dataset Validation Completeness**
    - **Validates: Requirements 2.1**

  - [ ] 3.3 实现检查点管理器
    - 创建 `worker/training/checkpoint/checkpoint_manager.py`
    - 实现检查点保存（包含 step, epoch, loss, metadata）
    - 实现最新检查点自动检测
    - 实现检查点完整性验证
    - 实现旧检查点清理
    - _Requirements: 2.3, 2.4_

  - [ ] 3.4 编写检查点管理器属性测试
    - **Property 7: Checkpoint Detection and Loading**
    - **Validates: Requirements 2.4**

  - [ ] 3.5 实现早停监控器
    - 创建 `worker/training/callbacks/early_stop.py`
    - 实现 EarlyStopConfig 配置类
    - 实现 patience 和 min_delta 逻辑
    - 实现 should_stop() 判断方法
    - 集成到 TrainingEngine
    - _Requirements: 2.10_

  - [ ] 3.6 编写早停监控器属性测试
    - **Property 9: Early Stopping Trigger Correctness**
    - **Validates: Requirements 2.10**

  - [ ] 3.7 增强训练队列优先级支持
    - 修改 `worker/training/engine.py` 中的任务队列
    - 实现优先级排序逻辑
    - 添加优先级字段到 TrainingJob
    - _Requirements: 2.7_

  - [ ] 3.8 编写训练队列属性测试
    - **Property 8: Training Queue Priority Ordering**
    - **Validates: Requirements 2.7**

  - [ ] 3.9 增强训练进度报告
    - 修改 ProgressCallback 添加更多指标
    - 添加 GPU 利用率、ETA、samples/second 指标
    - 实现可配置的报告间隔
    - _Requirements: 2.2, 2.8_

- [ ] 4. Checkpoint - 确保训练模块测试通过
  - 运行所有训练模块相关测试
  - 如有问题请询问用户

- [ ] 5. 推理性能优化
  - 提升推理吞吐量和延迟
  - _Requirements: 3.1-3.10_

  - [ ] 5.1 实现 KV Cache 管理器
    - 创建 `worker/engines/cache/kv_cache_manager.py`
    - 实现前缀哈希计算
    - 实现 LRU/LFU 缓存驱逐策略
    - 实现缓存大小限制
    - 集成到 VllmCudaEngine
    - _Requirements: 3.2, 3.10_

  - [ ] 5.2 编写 KV Cache 属性测试
    - **Property 10: Prefix Cache Hit Consistency**
    - **Property 16: Cache Eviction Policy Correctness**
    - **Validates: Requirements 3.2, 3.10**

  - [ ] 5.3 实现 LoRA 热切换管理器
    - 创建 `worker/engines/lora/hot_swapper.py`
    - 实现适配器注册和预加载
    - 实现无需重载基座模型的切换
    - 实现最大加载适配器数量限制
    - _Requirements: 3.6_

  - [ ] 5.4 编写 LoRA 热切换属性测试
    - **Property 12: LoRA Adapter Switching Isolation**
    - **Validates: Requirements 3.6**

  - [ ] 5.5 增强延迟指标收集
    - 修改 `worker/core/telemetry.py`
    - 添加 queue_time_ms, ttft_ms, total_time_ms 指标
    - 添加 tokens_per_second 计算
    - 实现 P50/P95/P99 百分位计算
    - _Requirements: 3.5, 3.9_

  - [ ] 5.6 编写延迟指标属性测试
    - **Property 11: Latency Metrics Completeness**
    - **Property 15: Percentile Latency Calculation Correctness**
    - **Validates: Requirements 3.5, 3.9**

  - [ ] 5.7 实现请求优先级处理
    - 修改 `worker/core/task_scheduler.py`
    - 实现优先级队列
    - 添加客户端层级支持
    - _Requirements: 3.7_

  - [ ] 5.8 编写请求优先级属性测试
    - **Property 13: Request Priority Processing Order**
    - **Validates: Requirements 3.7**

  - [ ] 5.9 增强 GPU OOM 错误处理
    - 修改 `worker/exceptions.py` 中的 GPUMemoryError
    - 添加 suggestions 字段
    - 实现常见优化建议生成
    - _Requirements: 3.8_

  - [ ] 5.10 编写 OOM 错误属性测试
    - **Property 14: GPU OOM Error Suggestions**
    - **Validates: Requirements 3.8**

- [ ] 6. Checkpoint - 确保推理优化测试通过
  - 运行所有推理优化相关测试
  - 如有问题请询问用户

- [ ] 7. 运行时体验优化
  - 改善开发和运维体验
  - _Requirements: 4.1-4.10_

  - [ ] 7.1 实现结构化错误系统
    - 创建 `gateway/src/main/kotlin/com/cy/llm/error/ErrorCodes.kt`
    - 定义错误码枚举
    - 实现错误响应格式化
    - 添加修复建议生成
    - _Requirements: 4.2_

  - [ ] 7.2 编写错误系统属性测试
    - **Property 17: Structured Error Message Format**
    - **Validates: Requirements 4.2**

  - [ ] 7.3 实现配置验证器
    - 创建 `worker/config/validator.py` 增强
    - 实现启动时配置验证
    - 实现详细错误报告
    - _Requirements: 4.4_

  - [ ] 7.4 编写配置验证属性测试
    - **Property 18: Configuration Validation Completeness**
    - **Validates: Requirements 4.4**

  - [ ] 7.5 实现请求追踪器
    - 创建 `gateway/src/main/kotlin/com/cy/llm/tracing/RequestTracer.kt`
    - 实现 trace_id 生成和传播
    - 添加 span 支持
    - 集成到 Gateway/Coordinator/Worker
    - _Requirements: 4.7_

  - [ ] 7.6 编写请求追踪属性测试
    - **Property 20: Trace ID Propagation**
    - **Validates: Requirements 4.7**

  - [ ] 7.7 增强超时错误上下文
    - 修改超时错误处理
    - 添加超时阶段信息（queue/inference/streaming）
    - 添加配置的超时值
    - _Requirements: 4.6_

  - [ ] 7.8 编写超时错误属性测试
    - **Property 19: Timeout Error Context**
    - **Validates: Requirements 4.6**

  - [ ] 7.9 实现结构化日志
    - 配置 JSON 格式日志输出
    - 添加可配置日志级别
    - 实现敏感数据脱敏
    - _Requirements: 4.10, 6.7_

  - [ ] 7.10 编写日志脱敏属性测试
    - **Property 30: Sensitive Data Redaction**
    - **Validates: Requirements 6.7**

- [ ] 8. Checkpoint - 确保运行时体验测试通过
  - 运行所有运行时体验相关测试
  - 如有问题请询问用户

- [ ] 9. 监控与可观测性
  - 实现全面的监控能力
  - _Requirements: 5.1-5.10_

  - [ ] 9.1 增强 Prometheus 指标导出
    - 修改 `worker/core/telemetry.py`
    - 添加更多指标（model_id 标签、租户标签）
    - 确保 Prometheus 格式合规
    - _Requirements: 5.1, 5.8_

  - [ ] 9.2 编写 Prometheus 指标属性测试
    - **Property 21: Prometheus Metrics Format Conformance**
    - **Property 22: Request Metrics Recording Completeness**
    - **Validates: Requirements 5.1, 5.2, 5.8**

  - [ ] 9.3 实现告警阈值触发
    - 添加 GPU 利用率阈值告警
    - 添加队列深度阈值告警
    - 实现告警去重（阈值恢复前只告警一次）
    - _Requirements: 5.4, 5.6_

  - [ ] 9.4 编写告警触发属性测试
    - **Property 23: Alert Threshold Triggering**
    - **Validates: Requirements 5.4, 5.6**

  - [ ] 9.5 实现缓存命中率统计
    - 修改 Prompt Cache 统计逻辑
    - 实现命中率计算
    - 添加到 Prometheus 指标
    - _Requirements: 5.5_

  - [ ] 9.6 编写缓存命中率属性测试
    - **Property 24: Cache Hit Rate Calculation**
    - **Validates: Requirements 5.5**

  - [ ] 9.7 实现审计日志
    - 创建 `gateway/src/main/kotlin/com/cy/llm/audit/AuditLogger.kt` 增强
    - 记录所有 API 请求
    - 包含 timestamp, trace_id, api_key_hash, endpoint, method, status_code, duration_ms
    - _Requirements: 5.10, 6.10_

  - [ ] 9.8 编写审计日志属性测试
    - **Property 25: Audit Log Completeness**
    - **Validates: Requirements 5.10, 6.10**

- [ ] 10. Checkpoint - 确保监控测试通过
  - 运行所有监控相关测试
  - 如有问题请询问用户

- [ ] 11. 安全与多租户
  - 实现企业级安全功能
  - _Requirements: 6.1-6.10_

  - [ ] 11.1 增强 API Key 认证
    - 修改 `gateway/src/main/kotlin/com/cy/llm/filter/ApiKeyFilter.kt`
    - 添加密钥过期检查
    - 添加密钥轮换支持
    - _Requirements: 6.1_

  - [ ] 11.2 编写认证属性测试
    - **Property 26: API Key Authentication Correctness**
    - **Validates: Requirements 1.8, 6.1**

  - [ ] 11.3 实现租户隔离
    - 添加租户标识解析
    - 实现租户级别指标隔离
    - 添加租户标签到所有指标
    - _Requirements: 6.2_

  - [ ] 11.4 编写租户隔离属性测试
    - **Property 27: Tenant Metrics Isolation**
    - **Validates: Requirements 6.2**

  - [ ] 11.5 实现速率限制器
    - 创建 `gateway/src/main/kotlin/com/cy/llm/security/RateLimiter.kt`
    - 实现基于 Redis 的滑动窗口限流
    - 支持 API Key 和租户级别限流
    - 返回 Retry-After 头
    - _Requirements: 6.3_

  - [ ] 11.6 编写速率限制属性测试
    - **Property 28: Rate Limit Enforcement**
    - **Validates: Requirements 6.3**

  - [ ] 11.7 实现输入验证和清理
    - 添加输入参数验证
    - 实现注入攻击防护
    - 添加控制字符过滤
    - _Requirements: 6.6_

  - [ ] 11.8 编写输入验证属性测试
    - **Property 29: Input Sanitization Effectiveness**
    - **Validates: Requirements 6.6**

  - [ ] 11.9 实现请求大小限制
    - 配置请求体大小限制
    - 实现超限拒绝（HTTP 413）
    - _Requirements: 6.8_

  - [ ] 11.10 编写请求大小限制属性测试
    - **Property 31: Request Size Limit Enforcement**
    - **Validates: Requirements 6.8**

  - [ ] 11.11 实现多级超时
    - 配置 Gateway 级别超时
    - 配置 Coordinator 级别超时
    - 配置 Worker 级别超时
    - 实现超时优先级逻辑
    - _Requirements: 6.9_

  - [ ] 11.12 编写多级超时属性测试
    - **Property 32: Multi-Level Timeout Enforcement**
    - **Validates: Requirements 6.9**

- [ ] 12. Final Checkpoint - 确保所有测试通过
  - 运行完整测试套件
  - 验证所有功能正常工作
  - 如有问题请询问用户

## Notes

- 所有任务都是必须完成的，包括属性测试
- 每个任务引用具体的需求以确保可追溯性
- 检查点任务确保增量验证
- 属性测试验证通用正确性属性
- 单元测试验证具体示例和边界情况
- Kotlin 使用 Kotest Property Testing 框架
- Python 使用 Hypothesis 框架

