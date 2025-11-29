// InferenceService.kt
// [编排] 根据 ModelRegistry 与 CircuitBreaker 调度并管理到 Worker 的 gRPC 流式连接
// 说明：负责流会话的创建、转发、重试策略和失败回退。

package com.genshin.ai.service

import com.genshin.ai.model.BackendSelectionException
import com.genshin.ai.model.InferenceChunk
import com.genshin.ai.model.InferenceRequest
import com.genshin.ai.config.BackendCircuitBreakers
import io.github.resilience4j.bulkhead.BulkheadRegistry
import io.github.resilience4j.circuitbreaker.CircuitBreaker
import io.github.resilience4j.retry.Retry
import io.github.resilience4j.retry.RetryRegistry
import io.github.resilience4j.timelimiter.TimeLimiterRegistry
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.channelFlow
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.withTimeout
import org.slf4j.LoggerFactory
import org.springframework.stereotype.Service
import java.time.Duration
import java.util.function.Supplier

/**
 * InferenceService
 * - 调度 ModelRegistry 中选择的后端
 * - 对每个后端应用 Bulkhead/Retry/CircuitBreaker/TimeLimiter 策略
 * - 将后端的 protobuf 流转换为内部的 `InferenceChunk` Flow
 */
@Service
class InferenceService(
	private val registry: ModelRegistry,
	private val workerClient: WorkerStreamClient,
	private val circuitBreakers: BackendCircuitBreakers,
	private val retryRegistry: RetryRegistry,
	private val bulkheadRegistry: BulkheadRegistry,
	private val timeLimiterRegistry: TimeLimiterRegistry,
) {

	private val logger = LoggerFactory.getLogger(InferenceService::class.java)

	/**
	 * 主入口：根据 `InferenceRequest` 返回流式的 `InferenceChunk`。
	 * - 会尝试 plan.candidates 中的后端，若一个后端成功则停止并返回其流
	 * - 全部失败时抛出 BackendSelectionException
	 */
	fun streamInference(request: InferenceRequest): Flow<InferenceChunk> = channelFlow {
		val plan = registry.routeFor(request.modelId, request.metadata.traceId)
		var lastError: Throwable? = null

		for (target in plan.candidates) {
			val breaker = circuitBreakers.forTarget(target)
			val retry = retryRegistry.retry(target.provider)
			val bulkhead = bulkheadRegistry.bulkhead(target.provider)
			val timeoutDuration = timeLimiterRegistry.timeLimiter(target.provider).timeLimiterConfig.timeoutDuration

			// 快速检查 bulkhead（并发限制）是否允许执行
			if (!bulkhead.tryAcquirePermission()) {
				logger.warn("Bulkhead saturated for provider {}. Skipping {}", target.provider, target.authority)
				continue
			}

			try {
				val supplier = Supplier {
					workerClient.stream(target, request, plan.traceId)
				}

				// 将 supplier 包装为 retry + circuit breaker 的防护调用
				val guardedSupplier = Retry.decorateSupplier(retry, CircuitBreaker.decorateSupplier(breaker, supplier))
				val stream = guardedSupplier.get()

				val collectBlock: suspend () -> Unit = {
					stream.collect { response ->
						send(response.toChunk(plan.traceId))
					}
				}

				// 在 time limiter 的超时时间内收集流
				collectWithTimeout(timeoutDuration, collectBlock)
				lastError = null
				return@channelFlow
			} catch (ex: Exception) {
				lastError = ex
				logger.warn(
					"Backend {} failed for trace {}: {}",
					target.authority,
					plan.traceId,
					ex.message,
				)
			} finally {
				bulkhead.releasePermission()
			}
		}

		if (lastError != null) {
			throw BackendSelectionException("All backends failed for model ${request.modelId}", lastError)
		}
	}

	/**
	 * 基于给定 timeout 调用收集逻辑；若 timeout 为 0 或负数则不限制
	 */
	private suspend fun collectWithTimeout(timeout: Duration, block: suspend () -> Unit) {
		if (timeout.isZero || timeout.isNegative) {
			block()
			return
		}

		withTimeout(timeout.toMillis()) {
			block()
		}
	}
}

/** 将 protobuf 响应映射为内部的 InferenceChunk */
private fun com.genshin.ai.proto.StreamPredictResponse.toChunk(defaultTraceId: String): InferenceChunk =
	InferenceChunk(
		traceId = if (traceId.isBlank()) defaultTraceId else traceId,
		index = index.toInt(),
		content = chunk,
		done = endOfStream,
	)
