package com.genshin.ai

import com.genshin.ai.model.BackendTarget
import com.genshin.ai.model.InferenceRequest
import com.genshin.ai.proto.StreamPredictResponse
import com.genshin.ai.service.WorkerStreamClient
import kotlinx.coroutines.flow.flow
import org.junit.jupiter.api.Test
import org.mockito.kotlin.any
import org.mockito.kotlin.given
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.boot.test.autoconfigure.web.reactive.AutoConfigureWebTestClient
import org.springframework.boot.test.context.SpringBootTest
import org.springframework.boot.test.mock.mockito.MockBean
import org.springframework.http.MediaType
import org.springframework.test.context.ActiveProfiles
import org.springframework.test.web.reactive.server.WebTestClient
import org.springframework.core.ParameterizedTypeReference
import reactor.test.StepVerifier

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@AutoConfigureWebTestClient
@ActiveProfiles("test")
class GatewayIntegrationTest {

    @Autowired
    private lateinit var webClient: WebTestClient

    @MockBean
    private lateinit var workerClient: WorkerStreamClient

    @Test
    fun `should stream inference chunks via SSE`() {
        // Mock worker response
        val mockFlow = flow {
            emit(StreamPredictResponse.newBuilder().setIndex(0).setChunk("Hello").setTraceId("trace-1").build())
            emit(StreamPredictResponse.newBuilder().setIndex(1).setChunk(" World").setTraceId("trace-1").build())
            emit(StreamPredictResponse.newBuilder().setIndex(2).setChunk("!").setTraceId("trace-1").setEndOfStream(true).build())
        }

        given(workerClient.stream(any(), any(), any()))
            .willReturn(mockFlow)

        val request = InferenceRequest(
            modelId = "test-model",
            prompt = "Hi"
        )

        val result = webClient.post().uri("/api/v1/inference/stream")
            .contentType(MediaType.APPLICATION_JSON)
            .bodyValue(request)
            .exchange()
            .expectStatus().isOk
            .expectHeader().contentTypeCompatibleWith(MediaType.TEXT_EVENT_STREAM)
            .returnResult(String::class.java)

        StepVerifier.create(result.responseBody)
            .recordWith { java.util.ArrayList<String>() }
            .expectNextCount(3)
            .consumeRecordedWith { events ->
                val responseBody = events.joinToString("")
                assert(responseBody.contains("Hello"))
                assert(responseBody.contains("World"))
            }
            .verifyComplete()
    }

    @Test
    fun `should return aggregated summary for sync inference`() {
        // Mock worker response
        val mockFlow = flow {
            emit(StreamPredictResponse.newBuilder().setIndex(0).setChunk("Java").setTraceId("trace-2").build())
            emit(StreamPredictResponse.newBuilder().setIndex(1).setChunk("Kotlin").setTraceId("trace-2").setEndOfStream(true).build())
        }

        given(workerClient.stream(any(), any(), any()))
            .willReturn(mockFlow)

        val request = InferenceRequest(
            modelId = "test-model",
            prompt = "Tell me languages"
        )

        webClient.post().uri("/api/v1/inference")
            .contentType(MediaType.APPLICATION_JSON)
            .bodyValue(request)
            .exchange()
            .expectStatus().isOk
            .expectBody()
            .jsonPath("$.content").isEqualTo("JavaKotlin")
            .jsonPath("$.modelId").isEqualTo("test-model")
    }
}
