// GrpcConfig.kt
// [gRPC 客户端] 管理 gRPC 渠道池、超时与连接策略
// 说明：封装到 Worker 的 gRPC client 配置（连接池、线程/流管理、背压策略）。

package com.genshin.ai.config

import com.genshin.ai.model.BackendTarget
import io.grpc.ManagedChannel
import io.grpc.ManagedChannelBuilder
import org.springframework.boot.context.properties.ConfigurationProperties
import org.springframework.boot.context.properties.EnableConfigurationProperties
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import org.springframework.util.unit.DataSize
import java.time.Duration
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.TimeUnit

/**
 * gRPC 客户端配置属性。
 * - 可通过 `gateway.grpc.*` 在 `application.yml` 中调整连接行为和消息大小上限。
 */
@ConfigurationProperties(prefix = "gateway.grpc")
data class GrpcClientProperties(
	val plaintext: Boolean = true,
	val keepAliveTime: Duration = Duration.ofSeconds(30),
	val keepAliveTimeout: Duration = Duration.ofSeconds(10),
	val idleTimeout: Duration = Duration.ofMinutes(5),
	val shutdownAwait: Duration = Duration.ofSeconds(5),
	val maxInboundMessageSize: DataSize = DataSize.ofMegabytes(16),
)

/**
 * 提供 `GrpcChannelFactory` Bean 的 Spring 配置。
 */
@Configuration
@EnableConfigurationProperties(GrpcClientProperties::class)
class GrpcConfig {

	@Bean(destroyMethod = "close")
	fun grpcChannelFactory(properties: GrpcClientProperties): GrpcChannelFactory = GrpcChannelFactory(properties)
}

/**
 * 简单的 ManagedChannel 缓存工厂：
 * - 基于 target host:port 缓存 channel
 * - 负责创建与优雅关闭 channel
 */
class GrpcChannelFactory(private val properties: GrpcClientProperties) : AutoCloseable {

	private val channels = ConcurrentHashMap<String, ManagedChannel>()

	/** 获取或创建到指定后端的 ManagedChannel */
	fun get(target: BackendTarget): ManagedChannel {
		val authority = "${target.host}:${target.port}"
		return channels.computeIfAbsent(authority) { buildChannel(target) }
	}

	private fun buildChannel(target: BackendTarget): ManagedChannel {
		val builder = ManagedChannelBuilder.forAddress(target.host, target.port)
			.keepAliveTime(properties.keepAliveTime.toMillis(), TimeUnit.MILLISECONDS)
			.keepAliveTimeout(properties.keepAliveTimeout.toMillis(), TimeUnit.MILLISECONDS)
			.idleTimeout(properties.idleTimeout.toMillis(), TimeUnit.MILLISECONDS)
			.maxInboundMessageSize(properties.maxInboundMessageSize.toBytes().toInt())

		if (properties.plaintext || !target.secure) {
			builder.usePlaintext()
		}

		return builder.build()
	}

	/** 关闭指定 target 的 channel（如存在） */
	fun shutdown(target: BackendTarget) {
		val authority = "${target.host}:${target.port}"
		channels.remove(authority)?.shutdownGracefully()
	}

	private fun ManagedChannel.shutdownGracefully() {
		shutdown()
		awaitTermination(properties.shutdownAwait.toMillis(), TimeUnit.MILLISECONDS)
	}

	/** 关闭并清空所有缓存的 channel（应用停止时调用） */
	override fun close() {
		channels.values.forEach { channel ->
			channel.shutdownGracefully()
		}
		channels.clear()
	}
}
