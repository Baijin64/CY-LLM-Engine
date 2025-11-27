"""
main.py
[入口] 启动脚本：解析配置、初始化引擎与内存管理器、启动 gRPC 服务
"""

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

# 允许从任意工作目录运行 main.py。
# 为了让包内的相对导入生效，需要把 worker 的父目录加入 sys.path，
# 这样就可以通过 "import worker.core" 的方式加载模块。
WORKER_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(WORKER_ROOT)
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

from worker.config.config_loader import ModelSpec, WorkerConfig, load_worker_config
from worker.core.server import InferenceServer
from worker.core.task_scheduler import TaskScheduler
from worker.core.telemetry import Telemetry
from worker.engines.engine_factory import create_engine


LOGGER = logging.getLogger("ew.worker")


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Element Warfare AI Worker",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument("--model", default="default", help="逻辑模型 ID（参考模型注册表）")
	parser.add_argument("--prompt", help="若提供则直接运行一次推理并退出。")
	parser.add_argument("--list-models", action="store_true", help="列出当前可用模型映射。")
	parser.add_argument("--preload-all", action="store_true", help="启动时预加载所有模型（可能占用大量显存）。")
	parser.add_argument("--registry", help="指定模型注册表 JSON 路径，默认读取环境变量。")
	parser.add_argument("--max-workers", type=int, default=2, help="调度线程数量。")
	parser.add_argument("--queue-size", type=int, default=128, help="TaskScheduler 队列容量。")
	parser.add_argument("--max-new-tokens", type=int, default=256, help="一次 CLI 推理生成的最大 token 数。")
	parser.add_argument("--temperature", type=float, default=0.7, help="CLI 推理温度系数。")
	parser.add_argument("--serve", action="store_true", help="阻塞运行，等待上层 gRPC/HTTP 服务接入。")
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


def _build_server(args: argparse.Namespace) -> InferenceServer:
	def factory(engine_type: str):
		return create_engine(engine_type)

	scheduler = TaskScheduler(max_workers=args.max_workers, queue_size=args.queue_size)
	telemetry = Telemetry()
	return InferenceServer(engine_factory=factory, scheduler=scheduler, telemetry=telemetry)


def _resolve_spec(config: WorkerConfig, logical_name: str) -> ModelSpec:
	try:
		return config.model_registry[logical_name]
	except KeyError as exc:
		raise SystemExit(f"找不到逻辑模型 '{logical_name}'，可通过 --list-models 查看。") from exc


def _engine_kwargs(spec: ModelSpec) -> Optional[dict]:
	kwargs = {}
	if spec.use_4bit is not None:
		kwargs["use_4bit"] = spec.use_4bit
	return kwargs or None


def _preload_models(server: InferenceServer, config: WorkerConfig, names: Iterable[str]) -> None:
	for name in names:
		spec = _resolve_spec(config, name)
		LOGGER.info("预加载模型 %s -> %s", name, spec.model_path)
		server.ensure_model(
			model_id=name,
			model_path=spec.model_path,
			adapter_path=spec.adapter_path,
			engine_type=spec.engine or config.preferred_backend,
			engine_kwargs=_engine_kwargs(spec),
		)


def _run_prompt_once(
	server: InferenceServer,
	config: WorkerConfig,
	logical_name: str,
	prompt: str,
	*,
	max_new_tokens: int,
	temperature: float,
) -> None:
	spec = _resolve_spec(config, logical_name)
	LOGGER.info("执行一次推理：model=%s prompt_len=%s", logical_name, len(prompt))
	for chunk in server.stream_predict(
		model_id=logical_name,
		prompt=prompt,
		model_path=spec.model_path,
		adapter_path=spec.adapter_path,
		engine_type=spec.engine or config.preferred_backend,
		engine_kwargs=_engine_kwargs(spec),
		generation_kwargs={
			"max_new_tokens": max_new_tokens,
			"temperature": temperature,
		},
	):
		print(chunk, end="", flush=True)
	print()


def _serve_forever(server: InferenceServer) -> None:
	stop_event = threading.Event()

	def _handle_signal(signum, _frame):  # noqa: D401
		LOGGER.info("收到信号 %s，准备关闭。", signum)
		stop_event.set()

	signal.signal(signal.SIGINT, _handle_signal)
	signal.signal(signal.SIGTERM, _handle_signal)
	LOGGER.info("Worker 已启动，等待上层 gRPC 服务接入（占位实现）。Ctrl+C 退出。")
	while not stop_event.is_set():
		time.sleep(0.5)
	server.shutdown()


def main() -> None:
	logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
	args = _parse_args()
	config = load_worker_config(args.registry)
	LOGGER.info(
		"硬件概览：CUDA=%s/%s Ascend=%s/%s 选定后端=%s",
		config.hardware.has_cuda,
		config.hardware.cuda_device_count,
		config.hardware.has_ascend,
		config.hardware.ascend_device_count,
		config.preferred_backend,
	)

	if args.list_models:
		print(_format_registry(config))
		if not args.prompt and not args.preload_all and not args.serve:
			return

	inference_server = _build_server(args)

	if args.preload_all:
		_preload_models(inference_server, config, config.model_registry.keys())

	if args.prompt:
		_run_prompt_once(
			inference_server,
			config,
			args.model,
			args.prompt,
			max_new_tokens=args.max_new_tokens,
			temperature=args.temperature,
		)
		if not args.serve:
			inference_server.shutdown()
			return

	if args.serve:
		_serve_forever(inference_server)
	else:
		LOGGER.info("未指定 --serve，程序执行完毕。")
		inference_server.shutdown()


if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		LOGGER.info("用户中断，退出。")
	except SystemExit as exc:
		raise exc
	except Exception:
		LOGGER.exception("Worker 发生未处理异常：")
		sys.exit(1)
