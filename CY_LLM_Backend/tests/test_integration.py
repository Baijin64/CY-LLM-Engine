"""测试入口脚本：提供多种本地测试选项。"""

import argparse
import os
import sys
import threading
import time
from typing import Callable, Dict, List

import torch

# 确保可以导入 worker 包
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from worker.engines.abstract_engine import BaseEngine
from worker.engines.nvidia_engine import NvidiaEngine
from worker.core.memory_manager import GPUMemoryManager
from worker.core.task_scheduler import SchedulerBusy, TaskScheduler
from worker.core.telemetry import Telemetry
from worker.utils.stream_buffer import BufferClosed, StreamBuffer


def _reset_manager(manager: GPUMemoryManager) -> None:
    """清空单例内的状态，避免不同测试之间互相影响。"""
    manager.loaded_models.clear()
    manager.last_access_time.clear()
    manager.model_locks.clear()


def run_engine_smoke_test() -> None:
    """仅针对 NvidiaEngine 进行快速加载与推理冒烟测试。"""

    print("=== [1] NvidiaEngine 冒烟测试 ===")
    default_model = os.environ.get("BACKEND_TEST_MODEL", "sshleifer/tiny-gpt2")
    model_path = input(f"基础模型 ID [默认 {default_model}]: ").strip() or default_model
    adapter_path = input("LoRA 适配器路径（可选，留空跳过）: ").strip() or None
    prompt = input("测试 Prompt [默认: 你好，你是谁？]: ").strip() or "你好，你是谁？"

    engine = NvidiaEngine()
    try:
        engine.load_model(model_path, adapter_path=adapter_path if adapter_path else None, use_4bit=torch.cuda.is_available())
    except Exception as exc:
        print(f"[SmokeTest] 模型加载失败: {exc}")
        return

    print("Response: ", end="")
    try:
        for chunk in engine.infer(prompt, max_new_tokens=64):
            print(chunk, end="", flush=True)
        print()
    except Exception as exc:
        print(f"\n[SmokeTest] 推理失败: {exc}")
    finally:
        engine.unload_model()
        print("[SmokeTest] 已卸载模型。")


def run_memory_manager_test() -> None:
    """验证 GPUMemoryManager 的 LRU 回收逻辑。"""

    print("=== [2] GPUMemoryManager LRU 测试 ===")
    manager = GPUMemoryManager()
    _reset_manager(manager)

    class DummyEngine(BaseEngine):
        def __init__(self, name: str) -> None:
            self.name = name
            self.loaded = False

        def load_model(self, *args, **kwargs) -> None:
            self.loaded = True

        def infer(self, prompt: str, **kwargs):  # type: ignore[override]
            yield f"{self.name}: {prompt}"

        def unload_model(self) -> None:
            self.loaded = False

        def get_memory_usage(self) -> Dict[str, float]:
            return {"used": 0.0, "total": 0.0}

    # 注册三个模型，模拟访问顺序
    engines = []
    for idx in range(3):
        engine = DummyEngine(f"dummy_{idx}")
        engine.load_model("path")
        engines.append(engine)
        manager.register_model(f"dummy_{idx}", engine)
        time.sleep(0.01)  # 确保时间戳不同

    # 更新访问次序，使 dummy_0 成为最久未使用对象
    manager.access_model("dummy_1")
    manager.access_model("dummy_2")

    print("触发一次 LRU 回收...")
    manager._evict_lru_model()  # type: ignore[attr-defined]

    if "dummy_0" not in manager.loaded_models and not engines[0].loaded:
        print("[MemoryTest] 通过：LRU 正确卸载 dummy_0。")
    else:
        print("[MemoryTest] 未通过：dummy_0 仍然存在。")

    # 清理状态
    _reset_manager(manager)


def run_task_scheduler_test() -> None:
    """验证 TaskScheduler 的优先级与背压行为。"""

    print("=== [3] TaskScheduler 调度测试 ===")
    # 使用 2 个 worker，队列容量为 2。要验证背压，需要同时让 worker 占满并把队列填满。
    scheduler = TaskScheduler(max_workers=2, queue_size=2)
    execution_order: List[str] = []

    def make_task(name: str, sleep_time: float) -> Callable[[], None]:
        def _task() -> None:
            # 模拟长耗时任务，确保在提交 overflow 时队列仍然被占用
            time.sleep(sleep_time)
            execution_order.append(name)

        return _task

    # 提交足够多的任务来占满 worker(2) + 队列(2) = 共 4 个任务
    futures = []
    futures.append(scheduler.submit(make_task("task-1", 0.3), priority=1))
    futures.append(scheduler.submit(make_task("task-2", 0.3), priority=1))
    futures.append(scheduler.submit(make_task("task-3", 0.3), priority=0))
    futures.append(scheduler.submit(make_task("task-4", 0.3), priority=0))

    # 立即尝试提交第 5 个任务，应触发 SchedulerBusy（背压）
    try:
        scheduler.submit(make_task("overflow", 0.01), priority=0)
    except SchedulerBusy:
        print("[SchedulerTest] 背压验证通过：队列已满时抛出 SchedulerBusy。")
    else:
        print("[SchedulerTest] 背压验证失败：应当抛出 SchedulerBusy。")

    # 等待已有任务完成，然后关闭调度器
    for future in futures:
        try:
            future.result(timeout=5)
        except Exception:
            pass

    scheduler.shutdown()
    print(f"执行顺序: {execution_order}")
    print("[SchedulerTest] 完成。")


def run_integration_test() -> None:
    """端到端测试：模型加载、推理以及显存回收流程。"""

    print("=== [3] NvidiaEngine + GPUMemoryManager 集成测试 ===")
    manager = GPUMemoryManager()
    _reset_manager(manager)

    base_model_default = "deepseek-ai/deepseek-llm-7b-chat"
    base_model_path = input(f"基础模型 ID [默认 {base_model_default}]: ").strip() or base_model_default

    adapter_root_default = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../EW_AI_Training/checkpoints/furina_lora_v2")
    )
    adapter_root = input(f"LoRA 根目录 [默认 {adapter_root_default}]: ").strip() or adapter_root_default

    model_id = os.path.basename(adapter_root) or "integration_model"

    adapter_path = adapter_root
    if os.path.exists(adapter_root):
        subdirs = [d for d in os.listdir(adapter_root) if d.startswith("checkpoint-")]
        if subdirs:
            latest_checkpoint = sorted(subdirs, key=lambda x: int(x.split("-")[1]))[-1]
            adapter_path = os.path.join(adapter_root, latest_checkpoint)
            print(f"自动定位到最新 Checkpoint: {adapter_path}")

    engine = NvidiaEngine()
    try:
        if not os.path.exists(adapter_path):
            print(f"警告: 未找到 LoRA 目录 {adapter_path}，仅加载基座模型。")
            engine.load_model(base_model_path)
        else:
            engine.load_model(base_model_path, adapter_path=adapter_path)

        manager.register_model(model_id, engine)
        print("模型加载并注册成功。")
        print(f"加载后显存状态: {manager.get_status_report()}")
    except Exception as exc:
        print(f"[Integration] 模型加载失败: {exc}")
        return

    prompt = input("推理 Prompt [默认: 你好，请介绍一下你自己。]: ").strip() or "你好，请介绍一下你自己。"
    print("Response: ", end="")
    try:
        for chunk in engine.infer(prompt, max_new_tokens=50):
            print(chunk, end="", flush=True)
        print("\n推理完成。")
    except Exception as exc:
        print(f"\n[Integration] 推理失败: {exc}")

    manager.access_model(model_id)
    print("\n模拟显存不足并触发回收...")
    manager.max_gpu_memory_usage = 0.1
    manager.check_memory_and_evict()
    print(f"回收后显存状态: {manager.get_status_report()}")

    if manager.get_loaded_model(model_id) is None and engine.model is None:
        print("[Integration] 通过：模型已被卸载且引擎清理完成。")
    else:
        print("[Integration] 警告：模型仍在内存中，请检查逻辑。")


def run_stream_buffer_test() -> None:
    """验证 StreamBuffer 的背压与关闭语义。"""

    print("=== [4] StreamBuffer 流控测试 ===")
    buffer = StreamBuffer[str](max_size=3, warn_threshold=0.6)
    produced: List[str] = []
    consumed: List[str] = []

    def producer() -> None:
        for idx in range(5):
            item = f"token-{idx}"
            produced.append(item)
            accepted = buffer.push(item, block=False)
            if not accepted:
                print(f"[StreamBufferTest] 背压触发，丢弃: {item}")
                time.sleep(0.05)
                buffer.push(item, block=True)
        buffer.close()

    def consumer() -> None:
        while True:
            try:
                token = buffer.pop(timeout=0.2)
            except BufferClosed:
                break
            consumed.append(token)
            time.sleep(0.03)

    t_prod = threading.Thread(target=producer)
    t_cons = threading.Thread(target=consumer)
    t_prod.start()
    t_cons.start()
    t_prod.join()
    t_cons.join()

    print(f"已生产: {produced}")
    print(f"已消费: {consumed}")
    print(f"统计数据: {buffer.stats()}")
    print("[StreamBufferTest] 完成。")


def run_telemetry_test() -> None:
    """验证 Telemetry 指标快照与导出逻辑。"""

    print("=== [5] Telemetry 指标测试 ===")
    telemetry = Telemetry()

    for latency, success in [(0.12, True), (0.3, False), (0.05, True)]:
        telemetry.track_request_start()
        time.sleep(0.01)
        telemetry.track_request_end(latency, success=success)

    telemetry.record_gpu_usage(used_mb=2048.0, total_mb=8192.0)
    snapshot = telemetry.snapshot()
    print("快照:", snapshot)
    print("Prometheus 导出:\n" + telemetry.export_prometheus())
    print("[TelemetryTest] 完成。")


def main() -> None:
    choices: Dict[str, Callable[[], None]] = {
        "engine": run_engine_smoke_test,
        "memory": run_memory_manager_test,
        "scheduler": run_task_scheduler_test,
        "integration": run_integration_test,
        "stream": run_stream_buffer_test,
        "telemetry": run_telemetry_test,
    }

    parser = argparse.ArgumentParser(description="EW AI Backend 测试入口")
    parser.add_argument("--test", choices=choices.keys(), help="直接运行指定测试，无需交互菜单")
    args = parser.parse_args()

    if args.test:
        choices[args.test]()
        return

    menu_mapping = {
        "1": "engine",
        "2": "memory",
        "3": "scheduler",
        "4": "integration",
        "5": "stream",
        "6": "telemetry",
    }

    while True:
        print("\n请选择要运行的测试：")
        print("  [1] NvidiaEngine 冒烟测试")
        print("  [2] GPUMemoryManager LRU 测试")
        print("  [3] TaskScheduler 调度测试")
        print("  [4] 集成测试 (NvidiaEngine + MemoryManager)")
        print("  [5] StreamBuffer 流控测试")
        print("  [6] Telemetry 指标测试")
        print("  [q] 退出")

        choice = input("请输入选项: ").strip().lower()
        if choice in {"q", "quit", "exit"}:
            print("已退出测试脚本。")
            break

        key = menu_mapping.get(choice, choice)
        test_fn = choices.get(key)
        if test_fn:
            test_fn()
        else:
            print("无效选项，请重新输入。")


if __name__ == "__main__":
    main()
