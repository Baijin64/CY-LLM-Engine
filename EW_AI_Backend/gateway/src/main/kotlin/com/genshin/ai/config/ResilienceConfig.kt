// ResilienceConfig.kt
// [Resilience] Resilience4j 全局配置：time limiter、retry、bulkhead 等
// 说明：定义给不同云/硬件提供者的容错策略与默认策略。

package com.genshin.ai.config

import io.github.resilience4j.bulkhead.BulkheadConfig
import io.github.resilience4j.bulkhead.BulkheadRegistry
import io.github.resilience4j.circuitbreaker.CircuitBreakerConfig
import io.github.resilience4j.circuitbreaker.CircuitBreakerRegistry
import io.github.resilience4j.retry.RetryConfig
import io.github.resilience4j.retry.RetryRegistry
import io.github.resilience4j.timelimiter.TimeLimiterConfig
import io.github.resilience4j.timelimiter.TimeLimiterRegistry
import io.grpc.StatusRuntimeException
import org.springframework.boot.context.properties.ConfigurationProperties
import org.springframework.boot.context.properties.EnableConfigurationProperties
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import java.io.IOException
import java.time.Duration

/**
 * Resilience4j 全局配置封装。
 * - 提供 TimeLimiter/Retry/Bulkhead/CircuitBreaker 的默认 Registry Bean
 * - 可通过 `gateway.resilience.*` 在配置文件中覆盖默认值
 */
@ConfigurationProperties(prefix = "gateway.resilience")
data class ResilienceProperties(
	val timeLimiter: TimeLimiterProperties = TimeLimiterProperties(),
	val retry: RetryProperties = RetryProperties(),
	val bulkhead: BulkheadProperties = BulkheadProperties(),
	val circuitBreaker: CircuitBreakerDefaults = CircuitBreakerDefaults(),
)

/**
 * 超时控制参数
 */
data class TimeLimiterProperties(val timeout: Duration = Duration.ofSeconds(45))

/**
 * 重试策略参数
 */
data class RetryProperties(
	val maxAttempts: Int = 3,
	val waitDuration: Duration = Duration.ofMillis(250),
)

/**
 * 并发隔离参数（信号量模式）
 */
data class BulkheadProperties(
	val maxConcurrent: Int = 8,
	val maxWait: Duration = Duration.ofSeconds(2),
)

/**
 * Circuit Breaker 默认阈值与窗口配置
 */
data class CircuitBreakerDefaults(
	val failureRateThreshold: Float = 50f,
	val slowCallRateThreshold: Float = 50f,
	val slowCallDuration: Duration = Duration.ofSeconds(15),
	val slidingWindowSize: Int = 20,
	val minimumNumberOfCalls: Int = 10,
	val waitDurationInOpen: Duration = Duration.ofSeconds(30),
	val permittedCallsInHalfOpenState: Int = 3,
)

@Configuration
@EnableConfigurationProperties(ResilienceProperties::class)
class ResilienceConfig(private val properties: ResilienceProperties) {

	/** CircuitBreakerRegistry：用于创建/查找 provider 级别的断路器 */
	@Bean
	fun circuitBreakerRegistry(): CircuitBreakerRegistry =
		CircuitBreakerRegistry.of(properties.circuitBreaker.toConfig())

	/** RetryRegistry：用于按 provider 创建重试策略 */
	@Bean
	fun retryRegistry(): RetryRegistry =
		RetryRegistry.of(properties.retry.toConfig())

	/** TimeLimiterRegistry：用于按 provider 创建超时策略 */
	@Bean
	fun timeLimiterRegistry(): TimeLimiterRegistry =
		TimeLimiterRegistry.of(properties.timeLimiter.toConfig())

	/** BulkheadRegistry：信号量/隔离策略 */
	@Bean
	fun bulkheadRegistry(): BulkheadRegistry =
		BulkheadRegistry.of(properties.bulkhead.toConfig())
}

private fun CircuitBreakerDefaults.toConfig(): CircuitBreakerConfig =
	CircuitBreakerConfig.custom()
		.failureRateThreshold(failureRateThreshold)
		.slowCallRateThreshold(slowCallRateThreshold)
		.slowCallDurationThreshold(slowCallDuration)
		.slidingWindowSize(slidingWindowSize)
		.minimumNumberOfCalls(minimumNumberOfCalls)
		.permittedNumberOfCallsInHalfOpenState(permittedCallsInHalfOpenState)
		.waitDurationInOpenState(waitDurationInOpen)
		.build()

private fun RetryProperties.toConfig(): RetryConfig =
	RetryConfig.custom<Any>()
		.maxAttempts(maxAttempts)
		.waitDuration(waitDuration)
		.retryExceptions(IOException::class.java, StatusRuntimeException::class.java)
		.build()

private fun TimeLimiterProperties.toConfig(): TimeLimiterConfig =
	TimeLimiterConfig.custom()
		.timeoutDuration(timeout)
		.build()

private fun BulkheadProperties.toConfig(): BulkheadConfig =
	BulkheadConfig.custom()
		.maxConcurrentCalls(maxConcurrent)
		.maxWaitDuration(maxWait)
		.build()
