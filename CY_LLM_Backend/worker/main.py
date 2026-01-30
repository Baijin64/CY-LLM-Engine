"""Worker entrypoint and compatibility helpers."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import textwrap
import threading
import time
from typing import Iterable, Optional

WORKER_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(WORKER_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 初始化基本日志，确保导入过程中的信息能输出
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger("cy_llm.worker")

from worker.config.config_loader import ModelSpec, WorkerConfig, load_worker_config  # type: ignore
from worker.core.server import InferenceServer  # type: ignore
from worker.core.task_scheduler import TaskScheduler  # type: ignore
from worker.core.telemetry import Telemetry  # type: ignore
from worker.engines.engine_factory import create_engine  # type: ignore
from worker.utils.auth import enforce_internal_token_policy  # type: ignore
from worker.utils.otel import init_tracing  # type: ignore
from worker.health.health_server import start_health_server  # type: ignore


def parse_args() -> argparse.Namespace:
    return _parse_args()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CY-LLM Worker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="default", help="逻辑模型 ID（参考模型注册表）")
    parser.add_argument("--prompt", help="若提供则直接运行一次推理并退出。")
    parser.add_argument("--list-models", action="store_true", help="列出当前可用模型映射。")
    parser.add_argument("--registry", help="指定模型注册表 JSON 路径，默认读取环境变量。")
    parser.add_argument("--config", help="指定配置文件路径（兼容旧 API）。")
    parser.add_argument("--max-workers", type=int, default=2, help="调度线程数量。")
    parser.add_argument("--queue-size", type=int, default=128, help="TaskScheduler 队列容量。")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="一次 CLI 推理生成的最大 token 数。")
    parser.add_argument("--temperature", type=float, default=0.7, help="CLI 推理温度系数。")
    parser.add_argument("--serve", action="store_true", help="阻塞运行，等待上层 gRPC/HTTP 服务接入。")
    parser.add_argument("--port", type=int, default=50051, help="gRPC 服务监听端口。")
    return parser.parse_args()


def _format_registry(config: WorkerConfig) -> str:
    lines = ["可用模型："]
    for name, spec in config.model_registry.items():
        lines.append(
            textwrap.dedent(
                f"  - {name}\n"
                f"    model:   {spec.model_path}\n"
                f"    adapter: {spec.adapter_path or '-'}\n"
                f"    engine:  {spec.engine or config.preferred_backend}\n"
                f"    use_4bit:{spec.use_4bit if spec.use_4bit is not None else 'inherit'}",
            )
        )
    return "\n".join(lines)


def _build_server(args: argparse.Namespace, config: WorkerConfig) -> InferenceServer:
    scheduler = TaskScheduler(max_workers=args.max_workers, queue_size=args.queue_size)
    telemetry = Telemetry()
    return InferenceServer(
        engine_factory=create_engine,
        scheduler=scheduler,
        telemetry=telemetry,
        worker_config=config,
    )


def _resolve_spec(config: WorkerConfig, logical_name: str) -> ModelSpec:
    try:
        return config.model_registry[logical_name]
    except KeyError as exc:
        raise SystemExit(f"找不到逻辑模型 '{logical_name}'，可通过 --list-models 查看。") from exc


def _engine_kwargs(spec: ModelSpec) -> Optional[dict]:
    kwargs: dict = {}
    if spec.use_4bit is not None:
        kwargs["use_4bit"] = spec.use_4bit
    if spec.quantization:
        kwargs["quantization"] = spec.quantization
    if spec.gpu_memory_utilization:
        kwargs["gpu_memory_utilization"] = spec.gpu_memory_utilization
    if spec.max_model_len:
        kwargs["max_model_len"] = spec.max_model_len
    return kwargs or None


def setup_signal_handlers(stop_event: threading.Event) -> None:
    def _handle_signal(signum, _frame):
        LOGGER.info("收到信号 %s，准备关闭。", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)


def graceful_shutdown(server, timeout: int = 5) -> None:
    try:
        server.stop(grace=timeout)
    except Exception:
        return


def _serve_forever(server: InferenceServer, config: WorkerConfig, port: int = 50051) -> None:
    from worker.grpc_servicer import create_grpc_server  # type: ignore

    stop_event = threading.Event()
    setup_signal_handlers(stop_event)

    grpc_server = create_grpc_server(
        inference_server=server,
        config=config,
        port=port,
    )
    grpc_server.start()

    health_server = start_health_server()
    health_thread = threading.Thread(target=health_server.serve_forever, daemon=True)
    health_thread.start()
    LOGGER.info("Worker gRPC 服务已启动，监听端口 %d。Ctrl+C 退出。", port)

    while not stop_event.is_set():
        time.sleep(0.5)

    LOGGER.info("正在关闭 gRPC 服务器...")
    grpc_server.stop(grace=5)
    health_server.shutdown()
    server.shutdown()


def main() -> None:
    init_tracing("cy-llm-worker")
    args = _parse_args()
    enforce_internal_token_policy()
    config = load_worker_config(args.registry)

    if args.list_models:
        print(_format_registry(config))
        if not args.prompt and not args.serve:
            return

    inference_server = _build_server(args, config)

    if args.prompt:
        spec = _resolve_spec(config, args.model)
        for chunk in inference_server.stream_predict(
            model_id=args.model,
            prompt=args.prompt,
            model_path=spec.model_path,
            adapter_path=spec.adapter_path,
            engine_type=spec.engine or config.preferred_backend,
            engine_kwargs=_engine_kwargs(spec),
            generation_kwargs={
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
            },
        ):
            print(chunk, end="", flush=True)
        print()
        if not args.serve:
            inference_server.shutdown()
            return

    if args.serve:
        _serve_forever(inference_server, config, port=args.port)
    else:
        inference_server.shutdown()


def create_server(port: int = 50051, max_workers: int = 10):
    import grpc
    from concurrent import futures
    return grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))


def register_servicers(server) -> None:
    return None


def setup_reflection(server) -> None:
    return None


if __name__ == "__main__":
    main()
