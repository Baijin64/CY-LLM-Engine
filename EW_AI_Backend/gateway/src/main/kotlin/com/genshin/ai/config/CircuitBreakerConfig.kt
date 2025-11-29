// CircuitBreakerConfig.kt
// [Circuit Breaker] 定义针对不同云厂商（Azure/Huawei/Nvidia）的断路器规则和回退策略
// 说明：使用 Resilience4j 来实现基于后端健康/错误率的断路器与降级逻辑。

package com.genshin.ai.config

import com.genshin.ai.model.BackendTarget
import io.github.resilience4j.circuitbreaker.CircuitBreaker
import io.github.resilience4j.circuitbreaker.CircuitBreakerConfig
import io.github.resilience4j.circuitbreaker.CircuitBreakerRegistry
import org.springframework.boot.context.properties.ConfigurationProperties
import org.springframework.boot.context.properties.EnableConfigurationProperties
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import java.time.Duration

/**
 * 为不同后端 Provider 提供可配置的 Circuit Breaker 映射：
 * - 支持在配置中按 provider 名称覆盖断路器阈值
 * - 返回一个 `BackendCircuitBreakers` 用于运行时按 target 快速查找断路器
 */
@ConfigurationProperties(prefix = "gateway.backends")
data class BackendCircuitBreakerProperties(
	val circuitBreakers: Map<String, CircuitBreakerOverride> = emptyMap(),
)

/** 可选覆盖项，优先使用此项合并默认配置 */
data class CircuitBreakerOverride(
	val failureRateThreshold: Float? = null,
	val slowCallRateThreshold: Float? = null,
	val slowCallDuration: Duration? = null,
	val slidingWindowSize: Int? = null,
	val minimumCalls: Int? = null,
	val waitDurationInOpen: Duration? = null,
	val permittedCallsInHalfOpen: Int? = null,
)

@Configuration
@EnableConfigurationProperties(BackendCircuitBreakerProperties::class)
class CircuitBreakerConfig(
	private val registry: CircuitBreakerRegistry,
	private val resilienceProperties: ResilienceProperties,
	private val backendProperties: BackendCircuitBreakerProperties,
) {

	/**
	 * 根据 `gateway.backends.circuitBreakers` 创建 provider -> CircuitBreaker 的映射
	 * 并且保留一个名为 `default` 的后备断路器。
	 */
	@Bean
	fun backendCircuitBreakers(): BackendCircuitBreakers {
		val defaults = resilienceProperties.circuitBreaker
		val breakerMap = backendProperties.circuitBreakers.entries.associate { (key, override) ->
			val config = override.merge(defaults)
			key.lowercase() to registry.circuitBreaker(key.lowercase(), config)
		}

		val fallback = registry.circuitBreaker("default")
		return BackendCircuitBreakers(breakerMap, fallback)
	}
}

/** 按 provider 快速返回对应的 CircuitBreaker，若没有配置则返回 fallback */
class BackendCircuitBreakers(
	private val breakers: Map<String, CircuitBreaker>,
	private val fallback: CircuitBreaker,
) {
	fun forTarget(target: BackendTarget): CircuitBreaker = forProvider(target.provider)

	fun forProvider(provider: String?): CircuitBreaker {
		val key = provider?.lowercase() ?: return fallback
		return breakers[key] ?: fallback
	}
}

/** 将 override 与默认值合并成最终的 CircuitBreakerConfig */
private fun CircuitBreakerOverride.merge(defaults: CircuitBreakerDefaults): CircuitBreakerConfig {
	val builder = CircuitBreakerConfig.custom()
		.failureRateThreshold(failureRateThreshold ?: defaults.failureRateThreshold)
		.slowCallRateThreshold(slowCallRateThreshold ?: defaults.slowCallRateThreshold)
		.slowCallDurationThreshold(slowCallDuration ?: defaults.slowCallDuration)
		.slidingWindowSize(slidingWindowSize ?: defaults.slidingWindowSize)
		.minimumNumberOfCalls(minimumCalls ?: defaults.minimumNumberOfCalls)
		.permittedNumberOfCallsInHalfOpenState(
			permittedCallsInHalfOpen ?: defaults.permittedCallsInHalfOpenState,
		)
		.waitDurationInOpenState(waitDurationInOpen ?: defaults.waitDurationInOpen)

	return builder.build()
}
