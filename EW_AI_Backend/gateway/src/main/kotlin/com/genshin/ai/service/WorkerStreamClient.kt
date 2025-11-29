package com.genshin.ai.service

import com.genshin.ai.config.GrpcChannelFactory
import com.genshin.ai.model.BackendTarget
import com.genshin.ai.model.InferenceRequest
import com.genshin.ai.proto.AiInferenceGrpc
import com.genshin.ai.proto.GenerationParameters
import com.genshin.ai.proto.StreamMetadata
import com.genshin.ai.proto.StreamPredictRequest
import com.genshin.ai.proto.StreamPredictResponse
import io.grpc.Status
import io.grpc.StatusRuntimeException
import io.grpc.stub.StreamObserver
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.channels.awaitClose
import org.slf4j.LoggerFactory
import org.springframework.stereotype.Component

/**
 * 抽象 Worker 客户端接口，封装到具体的 gRPC 实现中。
 */
interface WorkerStreamClient {
    fun stream(target: BackendTarget, request: InferenceRequest, traceId: String): Flow<StreamPredictResponse>
}

/**
 * 基于 gRPC 的 Worker 双向流客户端实现。
 * - 将 `InferenceRequest` 转换为 protobuf 请求并发送
 * - 使用 Kotlin Flow 将后端的流式响应暴露给上层
 */
@Component
class GrpcWorkerStreamClient(
    private val channelFactory: GrpcChannelFactory,
) : WorkerStreamClient {

    private val logger = LoggerFactory.getLogger(GrpcWorkerStreamClient::class.java)

    override fun stream(target: BackendTarget, request: InferenceRequest, traceId: String): Flow<StreamPredictResponse> = callbackFlow {
        val channel = channelFactory.get(target)
        val stub = AiInferenceGrpc.newStub(channel)

        val responseObserver = object : StreamObserver<StreamPredictResponse> {
            override fun onNext(value: StreamPredictResponse) {
                trySend(value).isSuccess
            }

            override fun onError(t: Throwable) {
                close(t)
            }

            override fun onCompleted() {
                close()
            }
        }

        val requestObserver = try {
            stub.streamPredict(responseObserver)
        } catch (ex: StatusRuntimeException) {
            close(ex)
            return@callbackFlow
        }

        try {
            // 发送单个预测请求，然后完成（gateway 当前为 request->stream 模式）
            requestObserver.onNext(request.toProto(traceId, target))
            requestObserver.onCompleted()
        } catch (ex: Exception) {
            logger.warn("Failed to send request to {}: {}", target.authority, ex.message)
            requestObserver.onError(ex)
            close(ex)
        }

        // 当上层取消 Flow 时向服务器发送取消信号
        awaitClose {
            try {
                requestObserver.onError(Status.CANCELLED.withDescription("client cancelled").asRuntimeException())
            } catch (_: Exception) {
                // ignore cancellation errors
            }
        }
    }

    /** 将内部 DTO 转为 protobuf 请求 */
    private fun InferenceRequest.toProto(traceId: String, target: BackendTarget): StreamPredictRequest {
        val builder = StreamPredictRequest.newBuilder()
            .setModelId(modelId)
            .setPrompt(prompt)
            .setPriority(priority)
            .setGeneration(generation.toProto())
            .setMetadata(metadata.toProto(traceId))
            .setWorkerHint(target.id)

        adapter?.takeIf { it.isNotBlank() }?.let { builder.adapter = it }
        return builder.build()
    }

    private fun com.genshin.ai.model.GenerationParameters.toProto(): GenerationParameters =
        GenerationParameters.newBuilder()
            .setMaxNewTokens(maxNewTokens)
            .setTemperature(temperature)
            .setTopP(topP)
            .setRepetitionPenalty(repetitionPenalty)
            .build()

    private fun com.genshin.ai.model.RequestMetadata.toProto(fallbackTraceId: String): StreamMetadata {
        val builder = StreamMetadata.newBuilder()
            .setTraceId(traceId.ifBlank { fallbackTraceId })

        tenant?.let { builder.tenant = it }
        playerId?.let { builder.playerId = it }
        locale?.let { builder.locale = it }
        extra.forEach { (key, value) -> builder.putExtra(key, value) }

        return builder.build()
    }
}
