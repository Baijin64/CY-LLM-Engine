import os
import asyncio

import grpc

from worker.proto_gen import ai_service_pb2, ai_service_pb2_grpc


class AiInferenceProxy(ai_service_pb2_grpc.AiInferenceServicer):
    def __init__(self, worker_addr: str):
        self.worker_addr = worker_addr

    async def StreamPredict(self, request_iterator, context):
        if not self.worker_addr:
            trace_id = ""
            async for req in request_iterator:
                if req.metadata and req.metadata.trace_id:
                    trace_id = req.metadata.trace_id
                break
            yield ai_service_pb2.StreamPredictResponse(
                trace_id=trace_id,
                chunk="worker_not_configured",
                end_of_stream=True,
                index=0,
            )
            return

        async with grpc.aio.insecure_channel(self.worker_addr) as channel:
            stub = ai_service_pb2_grpc.AiInferenceStub(channel)

            async def forward_requests():
                async for req in request_iterator:
                    yield req

            async for resp in stub.StreamPredict(forward_requests()):
                yield resp

    async def Control(self, request, context):
        if not self.worker_addr:
            return ai_service_pb2.ControlMessage(
                trace_id=request.trace_id,
                command="noop",
                payload={"error": "worker_not_configured"},
            )
        async with grpc.aio.insecure_channel(self.worker_addr) as channel:
            stub = ai_service_pb2_grpc.AiInferenceStub(channel)
            return await stub.Control(request)

    async def Health(self, request, context):
        if not self.worker_addr:
            return ai_service_pb2.WorkerHealthResponse(
                healthy=False,
                metrics={"error": "worker_not_configured"},
            )
        async with grpc.aio.insecure_channel(self.worker_addr) as channel:
            stub = ai_service_pb2_grpc.AiInferenceStub(channel)
            return await stub.Health(request)


async def serve():
    bind_addr = os.getenv("COORDINATOR_GRPC_BIND", "0.0.0.0:50051")
    worker_addr = os.getenv("WORKER_GRPC_ADDR", "127.0.0.1:50052")

    server = grpc.aio.server()
    ai_service_pb2_grpc.add_AiInferenceServicer_to_server(AiInferenceProxy(worker_addr), server)
    server.add_insecure_port(bind_addr)

    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
