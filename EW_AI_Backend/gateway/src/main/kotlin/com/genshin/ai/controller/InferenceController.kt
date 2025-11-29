// InferenceController.kt
// [边界] 对外 REST/WebSocket/HTTP 接口：接收游戏客户端的推理请求并转发到 InferenceService
// 说明：处理认证、速率限制、请求校验与会话管理。

package com.genshin.ai.controller

import com.genshin.ai.model.CompletionSummary
import com.genshin.ai.model.InferenceChunk
import com.genshin.ai.model.InferenceRequest
import com.genshin.ai.service.InferenceService
import jakarta.validation.Valid
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.collect
import org.springframework.http.MediaType
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestBody
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RestController

/**
 * 对外 HTTP 接口：
 * - `/api/v1/inference/stream` 返回 SSE（文本事件流）用于实时消费 token
 * - `/api/v1/inference` 同步汇总后返回完整结果（会阻塞直到流结束）
 */
@RestController
@RequestMapping("/api/v1/inference")
class InferenceController(
	private val inferenceService: InferenceService,
) {

	/**
	 * 流式接口（SSE）：逐片段返回推理输出，适合前端直接订阅显示增量结果
	 */
	@PostMapping("/stream", produces = [MediaType.TEXT_EVENT_STREAM_VALUE])
	fun stream(@Valid @RequestBody request: InferenceRequest): Flow<InferenceChunk> =
		inferenceService.streamInference(request)

	/**
	 * 同步接口：收集流直到结束并返回聚合文本与延迟信息，适合简单客户端或测试使用
	 */
	@PostMapping
	suspend fun infer(@Valid @RequestBody request: InferenceRequest): CompletionSummary {
		val start = System.nanoTime()
		val builder = StringBuilder()
		var trace = request.metadata.traceId

		inferenceService.streamInference(request).collect { chunk ->
			if (chunk.content.isNotEmpty()) {
				builder.append(chunk.content)
			}
			trace = chunk.traceId
		}

		val latencyMs = (System.nanoTime() - start) / 1_000_000
		return CompletionSummary(
			traceId = trace,
			modelId = request.modelId,
			content = builder.toString(),
			latencyMillis = latencyMs,
		)
	}
}
