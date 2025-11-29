// DomainModels.kt
// [DTO] 网关内部使用的数据模型与 Protobuf <-> Domain 转换器占位文件
// 说明：定义请求/响应的内部表示、错误模型与追踪上下文。

package com.genshin.ai.model

import jakarta.validation.constraints.DecimalMax
import jakarta.validation.constraints.DecimalMin
import jakarta.validation.constraints.Max
import jakarta.validation.constraints.Min
import jakarta.validation.constraints.NotBlank
import jakarta.validation.constraints.Size
import java.time.OffsetDateTime
import java.util.UUID

/**
 * 网关内部使用的域模型 (DTO)
 * - 这些类型用于在 Controller / Service / WorkerClient 之间传递信息
 */
data class GenerationParameters(
	@field:Max(2048)
	@field:Min(1)
	val maxNewTokens: Int = 512,
	@field:DecimalMin("0.0")
	val temperature: Double = 0.7,
	@field:DecimalMin("0.0")
	@field:DecimalMax("1.0")
	val topP: Double = 0.9,
	@field:DecimalMin("0.5")
	val repetitionPenalty: Double = 1.05,
)

/** 请求追踪与多租户/玩家元信息 */
data class RequestMetadata(
	val traceId: String = UUID.randomUUID().toString(),
	val tenant: String? = null,
	val playerId: String? = null,
	val locale: String? = null,
	val extra: Map<String, String> = emptyMap(),
)

/** 外部请求入参 DTO */
data class InferenceRequest(
	@field:NotBlank
	val modelId: String,
	@field:NotBlank
	@field:Size(max = 8000)
	val prompt: String,
	val adapter: String? = null,
	val priority: Int = 0,
	val generation: GenerationParameters = GenerationParameters(),
	val metadata: RequestMetadata = RequestMetadata(),
)

/** 单个流式输出片段 */
data class InferenceChunk(
	val traceId: String,
	val index: Int,
	val content: String,
	val done: Boolean = false,
	val emittedAt: OffsetDateTime = OffsetDateTime.now(),
)

/** 聚合完成后的简要汇总响应 */
data class CompletionSummary(
	val traceId: String,
	val modelId: String,
	val content: String,
	val latencyMillis: Long,
)

/** 后端目标信息：host/port/provider，用于路由与创建 gRPC channel */
data class BackendTarget(
	val id: String,
	val host: String,
	val port: Int,
	val provider: String,
	val secure: Boolean = false,
	val metadata: Map<String, String> = emptyMap(),
) {
	val authority: String = "$host:$port"
}

/** 路由计划：主后端 + 可选回退列表 */
data class RoutePlan(
	val traceId: String,
	val primary: BackendTarget,
	val fallbacks: List<BackendTarget> = emptyList(),
) {
	val candidates: List<BackendTarget> = listOf(primary) + fallbacks
}

class ModelNotFoundException(modelId: String) : IllegalArgumentException("Unknown model id: $modelId")

class BackendSelectionException(message: String, cause: Throwable? = null) : RuntimeException(message, cause)
