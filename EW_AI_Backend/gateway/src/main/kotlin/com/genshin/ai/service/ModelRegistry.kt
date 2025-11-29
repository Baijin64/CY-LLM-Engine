// ModelRegistry.kt
// [路由] 逻辑模型 ID -> 物理后端 映射与路由决策
// 说明：负责根据模型 ID、地域、硬件能力、流量策略返回目标后端地址列表或优先级。

package com.genshin.ai.service

import com.fasterxml.jackson.core.json.JsonReadFeature
import com.fasterxml.jackson.core.type.TypeReference
import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.databind.json.JsonMapper
import com.fasterxml.jackson.module.kotlin.KotlinModule
import com.fasterxml.jackson.module.kotlin.readValue
import com.genshin.ai.model.BackendTarget
import com.genshin.ai.model.ModelNotFoundException
import com.genshin.ai.model.RoutePlan
import org.slf4j.LoggerFactory
import org.springframework.boot.context.properties.ConfigurationProperties
import org.springframework.stereotype.Component
import java.nio.file.Files
import java.nio.file.InvalidPathException
import java.nio.file.Path
import java.nio.file.Paths
import java.time.Instant
import java.util.UUID
import java.util.concurrent.atomic.AtomicReference

/**
 * ModelRegistry 负责：
 * - 从 JSON 文件或环境变量加载逻辑模型到物理后端的映射
 * - 提供基于 modelId 的路由计划 `RoutePlan`
 * - 在找不到配置时回退到默认的 localhost:50051
 */
@ConfigurationProperties(prefix = "gateway.registry")
data class RegistryProperties(
	val path: String? = null,
	val defaultModel: String = "default",
	val defaultHost: String = "127.0.0.1",
	val defaultPort: Int = 50051,
	val defaultProvider: String = "nvidia",
)

@Component
class ModelRegistry(private val properties: RegistryProperties) {

	private val logger = LoggerFactory.getLogger(ModelRegistry::class.java)
	private val mapper = JsonMapper.builder()
		.enable(JsonReadFeature.ALLOW_JAVA_COMMENTS.mappedFeature())
		.enable(JsonReadFeature.ALLOW_YAML_COMMENTS.mappedFeature())
		.enable(JsonReadFeature.ALLOW_TRAILING_COMMA.mappedFeature())
		.enable(JsonReadFeature.ALLOW_SINGLE_QUOTES.mappedFeature())
		.build()
		.registerModule(KotlinModule.Builder().build())

	// 内部快照，线程安全的原子引用
	private val registry = AtomicReference(loadSnapshot())

	/** 从文件重新加载并替换当前快照，返回注册模型数量 */
	fun reload(): Int {
		val snapshot = runCatching { loadSnapshot() }
			.onFailure { logger.warn("Failed to reload model registry, keeping previous snapshot: {}", it.message) }
			.getOrElse { registry.get() }

		registry.set(snapshot)
		return snapshot.routes.size
	}

	/** 返回给定逻辑模型的路由计划，找不到时抛出 ModelNotFoundException */
	fun routeFor(modelId: String, traceId: String = UUID.randomUUID().toString()): RoutePlan {
		val entry = registry.get().routes[modelId] ?: throw ModelNotFoundException(modelId)
		return entry.toPlan(traceId)
	}

	fun availableModels(): Set<String> = registry.get().routes.keys

	fun describe(): Map<String, List<BackendTarget>> =
		registry.get().routes.mapValues { (_, entry) -> entry.describe() }

	/** 从 filesystem 或默认值构建快照 */
	private fun loadSnapshot(): RegistrySnapshot {
		val path = resolvePath()
		val routes = if (path != null && Files.exists(path)) {
			logger.info("Loading model registry from {}", path)
			parseRoutes(Files.readString(path), path.toString())
		} else {
			logger.warn("Model registry file not found, using default localhost route")
			mapOf(properties.defaultModel to defaultEntry(properties.defaultModel))
		}

		return RegistrySnapshot(routes = routes, source = path?.toString() ?: "default", loadedAt = Instant.now())
	}

	/** 将 JSON payload 解析为 RouteEntry 列表 */
	private fun parseRoutes(payload: String, source: String): Map<String, RouteEntry> {
		val root = mapper.readTree(payload)
		val modelsNode: JsonNode = root.takeIf { it.isObject && it.has("models") }?.get("models") ?: root
		val definitions: Map<String, ModelDefinition> = mapper.convertValue(modelsNode, object : TypeReference<Map<String, ModelDefinition>>() {})

		if (definitions.isEmpty()) {
			logger.warn("Registry source {} does not define any models, using default route", source)
			return mapOf(properties.defaultModel to defaultEntry(properties.defaultModel))
		}

		val defaultTarget = defaultTarget()
		return definitions.mapValues { (modelId, definition) ->
			definition.toEntry(modelId, defaultTarget)
		}
	}

	private fun defaultEntry(modelId: String): RouteEntry {
		val target = defaultTarget()
		return RouteEntry(modelId, target, emptyList())
	}

	private fun defaultTarget(): BackendTarget {
		val host = System.getenv("EW_WORKER_HOST") ?: properties.defaultHost
		val port = System.getenv("EW_WORKER_PORT")?.toIntOrNull() ?: properties.defaultPort
		return BackendTarget(
			id = "${properties.defaultModel}-primary",
			host = host,
			port = port,
			provider = properties.defaultProvider,
		)
	}

	/** 解析并验证配置文件路径，非法路径会被忽略 */
	private fun resolvePath(): Path? {
		val candidate = properties.path?.takeIf { it.isNotBlank() }
			?: System.getenv("EW_GATEWAY_REGISTRY_PATH")
		if (candidate.isNullOrBlank()) {
			return null
		}
		return try {
			Paths.get(candidate).toAbsolutePath().normalize()
		} catch (ex: InvalidPathException) {
			logger.warn("Invalid registry path '{}': {}", candidate, ex.message)
			null
		}
	}
}

private data class RegistrySnapshot(
	val routes: Map<String, RouteEntry>,
	val source: String,
	val loadedAt: Instant,
)

private data class RouteEntry(
	val modelId: String,
	val primary: BackendTarget,
	val fallbacks: List<BackendTarget>,
) {
	fun toPlan(traceId: String): RoutePlan = RoutePlan(traceId, primary, fallbacks)
	fun describe(): List<BackendTarget> = listOf(primary) + fallbacks
}

private data class ModelDefinition(
	val primary: BackendDefinition? = null,
	val targets: List<BackendDefinition>? = null,
	val fallbacks: List<BackendDefinition>? = null,
)

private data class BackendDefinition(
	val id: String? = null,
	val host: String? = null,
	val port: Int? = null,
	val provider: String? = null,
	val secure: Boolean? = null,
	val metadata: Map<String, String>? = null,
)

/** 将 ModelDefinition 展开为 RouteEntry（将 target definition 解析为 BackendTarget） */
private fun ModelDefinition.toEntry(modelId: String, defaultTarget: BackendTarget): RouteEntry {
	val ordered = when {
		primary != null -> listOf(primary) + (fallbacks ?: emptyList())
		!targets.isNullOrEmpty() -> targets
		else -> throw IllegalArgumentException("Model $modelId must define at least one backend target")
	}

	val resolved = ordered.mapIndexed { index, definition ->
		val fallbackId = definition.id ?: "$modelId-$index"
		definition.toTarget(defaultTarget, fallbackId)
	}

	val primaryTarget = resolved.firstOrNull() ?: defaultTarget
	val fallbackTargets = resolved.drop(1)
	return RouteEntry(modelId, primaryTarget, fallbackTargets)
}

private fun BackendDefinition.toTarget(defaultTarget: BackendTarget, fallbackId: String): BackendTarget {
	val resolvedHost = host?.takeIf { it.isNotBlank() } ?: defaultTarget.host
	val resolvedPort = port ?: defaultTarget.port
	val resolvedProvider = provider ?: defaultTarget.provider
	val resolvedSecure = secure ?: defaultTarget.secure
	val resolvedMetadata = metadata ?: emptyMap()

	return BackendTarget(
		id = fallbackId,
		host = resolvedHost,
		port = resolvedPort,
		provider = resolvedProvider,
		secure = resolvedSecure,
		metadata = resolvedMetadata,
	)
}
