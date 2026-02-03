import asyncio
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import grpc

from worker.proto_gen import ai_service_pb2, ai_service_pb2_grpc


@dataclass
class CoordinatorConfig:
    workers: List[str] = field(default_factory=list)


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config.json"


def load_config() -> CoordinatorConfig:
    config_path = os.getenv("COORDINATOR_CONFIG", "").strip()
    path = Path(config_path) if config_path else _default_config_path()
    if not path.exists():
        return CoordinatorConfig()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        workers = payload.get("workers") or []
        if not isinstance(workers, list):
            workers = []
        workers = [str(item).strip() for item in workers if str(item).strip()]
        return CoordinatorConfig(workers=workers)
    except Exception as exc:  # noqa: BLE001
        print(f"[CoordinatorLite] Failed to load config from {path}: {exc}")
        return CoordinatorConfig()


def get_worker_addrs() -> List[str]:
    env_multi = os.getenv("WORKER_UDS_PATHS", "").strip()
    if env_multi:
        parts = [item.strip() for item in env_multi.split(",") if item.strip()]
        if parts:
            return [f"unix://{p}" if not p.startswith("unix://") else p for p in parts]
    env_single = os.getenv("WORKER_UDS_PATH", "").strip()
    if env_single:
        addr = f"unix://{env_single}" if not env_single.startswith("unix://") else env_single
        return [addr]
    cfg_workers = load_config().workers
    if cfg_workers:
        return [f"unix://{w}" if not w.startswith("unix://") else w for w in cfg_workers]
    return ["unix:///tmp/cy_worker.sock"]


class RoundRobinSelector:
    def __init__(self, workers: List[str]):
        self._workers = workers
        self._index = 0
        self._lock = threading.Lock()

    def next(self) -> Optional[str]:
        if not self._workers:
            return None
        with self._lock:
            worker = self._workers[self._index % len(self._workers)]
            self._index += 1
            return worker


class AiInferenceProxy(ai_service_pb2_grpc.AiInferenceServicer):
    def __init__(self, worker_addrs: List[str]):
        self.selector = RoundRobinSelector(worker_addrs)

    async def StreamPredict(self, request_iterator, context):
        worker_addr = self.selector.next()
        if not worker_addr:
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

        async with grpc.aio.insecure_channel(worker_addr) as channel:
            stub = ai_service_pb2_grpc.AiInferenceStub(channel)

            async def forward_requests():
                async for req in request_iterator:
                    yield req

            async for resp in stub.StreamPredict(forward_requests()):
                yield resp

    async def Control(self, request, context):
        worker_addr = self.selector.next()
        if not worker_addr:
            return ai_service_pb2.ControlMessage(
                trace_id=request.trace_id,
                command="noop",
                payload={"error": "worker_not_configured"},
            )
        async with grpc.aio.insecure_channel(worker_addr) as channel:
            stub = ai_service_pb2_grpc.AiInferenceStub(channel)
            return await stub.Control(request)

    async def Health(self, request, context):
        worker_addr = self.selector.next()
        if not worker_addr:
            return ai_service_pb2.WorkerHealthResponse(
                healthy=False,
                metrics={"error": "worker_not_configured"},
            )
        async with grpc.aio.insecure_channel(worker_addr) as channel:
            stub = ai_service_pb2_grpc.AiInferenceStub(channel)
            return await stub.Health(request)


async def serve():
    bind_path = os.getenv("COORDINATOR_UDS_PATH", "/tmp/cy_coordinator.sock")
    # 清理旧的 socket 文件
    if os.path.exists(bind_path):
        os.unlink(bind_path)
    
    bind_addr = f"unix://{bind_path}"
    worker_addrs = get_worker_addrs()

    server = grpc.aio.server()
    ai_service_pb2_grpc.add_AiInferenceServicer_to_server(AiInferenceProxy(worker_addrs), server)
    server.add_insecure_port(bind_addr)

    print(f"[CoordinatorLite] Listening on UDS: {bind_addr}")
    print(f"[CoordinatorLite] Workers: {worker_addrs}")
    
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
