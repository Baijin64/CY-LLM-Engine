"""scripts/diagnose_env.py
硬件诊断工具（支持 NVIDIA CUDA 与华为 Ascend）。

用途：
- 检查 Python 环境中是否安装并可用 `torch`（PyTorch）与 `torch_npu`（Ascend 支持）。
- 报告库版本、设备名称与（若可用）设备总内存。
- 在检测到硬件时，在对应设备上执行一次简单的矩阵乘法以确保设备能够实际运行张量运算。

退出码（供自动化使用）：
- 0: 自检通过（在加速器或 CPU 上成功执行张量运算）。
- 1: 导入或自检失败（例如无法导入 PyTorch，或在所有设备上运行推理失败）。

示例：
    python3 scripts/diagnose_env.py

注：脚本尽量在不抛出致命错误的情况下执行：对 `torch_npu` 的导入使用安全导入逻辑，
以免在仅有 NVIDIA 机器上运行时报错。
"""

import importlib
import sys
from typing import Any, Dict, Optional


def import_module_safely(module_name: str) -> Optional[Any]:
    """尝试导入模块并在失败时返回 None。

    区别 ModuleNotFoundError（模块不存在）和其他导入时错误，并将错误信息打印到控制台，
    但不抛出异常，保证脚本在缺少可选依赖时仍可继续运行。
    """
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        # 常见情况：机器上没有安装该模块（例如 torch_npu 在非 Ascend 机器上）
        print(f"{module_name} 模块不可用: {exc}")
    except Exception as exc:
        # 其他导入时异常（比如依赖冲突、动态链接错误）记录到 stderr
        print(f"导入 {module_name} 时失败: {exc}", file=sys.stderr)
    return None


def bytes_to_human(size_bytes: Optional[int]) -> str:
    if size_bytes is None:
        return "unknown"
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f}{unit}"
        size /= 1024.0
    return f"{size:.2f}PB"


def gather_cuda_info(torch_module: Any) -> Dict[str, Any]:
    """收集 CUDA（NVIDIA GPU）相关信息。

    返回字典包含：available（bool），version（str，若可得），device_name（str），memory（字节或 None）。
    尝试使用 `torch.cuda.get_device_properties` 获取更详细信息；若失败则使用保守的替代方法。
    """
    info: Dict[str, Any] = {
        "available": False,
        "device_name": None,
        "memory": None,
    }
    try:
        cuda_available = torch_module.cuda.is_available()
    except Exception:
        cuda_available = False
    info["available"] = cuda_available
    if not cuda_available:
        return info

    info["version"] = getattr(torch_module.version, "cuda", "unknown")
    try:
        current_idx = torch_module.cuda.current_device()
    except Exception:
        current_idx = 0
    try:
        props = torch_module.cuda.get_device_properties(current_idx)
    except Exception:
        props = None
    if props is not None:
        info["device_name"] = getattr(props, "name", "Unknown CUDA Device")
        info["memory"] = getattr(props, "total_memory", None)
    else:
        try:
            info["device_name"] = torch_module.cuda.get_device_name(current_idx)
        except Exception:
            info["device_name"] = "Unknown CUDA Device"
    return info


def gather_ascend_info(torch_npu_module: Any) -> Dict[str, Any]:
    """收集华为 Ascend（NPU）相关信息。

    注意：不同版本的 `torch_npu`/厂商 SDK 在接口上可能有差异，函数使用宽松的判断与捕获，
    尽量返回可用的设备名或版本信息，而不会在不存在 NPU 的机器上抛出异常。
    返回字典包含：available（bool），version，device_name，memory（字节或 None）。
    """
    info: Dict[str, Any] = {
        "available": bool(torch_npu_module),
        "device_name": None,
        "memory": None,
        "version": getattr(torch_npu_module, "__version__", "unknown") if torch_npu_module else None,
    }
    if not torch_npu_module:
        return info

    npu_device = getattr(torch_npu_module, "npu", torch_npu_module)
    try:
        device_count = npu_device.device_count()
        if device_count:
            props = npu_device.get_device_properties(0)
            info["device_name"] = getattr(props, "name", "Ascend Device")
            info["memory"] = getattr(props, "total_memory", None)
        else:
            info["device_name"] = "Ascend Device"
    except Exception:
        # 如果 `torch_npu` 不提供与 CUDA 类似的属性接口，则尝试读取常见属性或退化为通用名称
        info["device_name"] = getattr(torch_npu_module, "device", "Ascend Device")
    return info


def run_inference(torch_module: Any, device_str: str) -> bool:
    """在指定设备上运行一次小型矩阵乘法以验证张量运算是否可用。

    device_str 示例："cuda:0", "npu:0", "cpu"。
    返回 True 表示运算成功（即时性验证），False 表示失败。
    """
    print(f"[Inference] 使用设备 {device_str} 执行矩阵乘法测试")
    try:
        device = torch_module.device(device_str)
    except Exception as exc:
        print(f"无法创建设备 {device_str}: {exc}", file=sys.stderr)
        return False
    try:
        # 使用 512x512 的矩阵，计算量适中，能较好地触发设备内核执行和内存分配
        x = torch_module.randn(512, 512, device=device, dtype=torch_module.float32)
        y = torch_module.randn(512, 512, device=device, dtype=torch_module.float32)
        result = torch_module.matmul(x, y)
        # 打印首个元素作为简要结果验证（避免打印整个张量）
        print("[Inference] 矩阵乘法成功，首个元素：", float(result[0, 0]))
        return True
    except Exception as exc:
        print(f"[Inference] 设备 {device_str} 上执行失败: {exc}", file=sys.stderr)
        return False


def main() -> None:
    torch_module = import_module_safely("torch")
    if torch_module is None:
        print("PyTorch 未安装，无法进行硬件诊断。", file=sys.stderr)
        sys.exit(1)

    torch_npu_module = None
    try:
        torch_npu_module = import_module_safely("torch_npu")
    except Exception:  # pragma: no cover
        pass

    print(f"torch 版本: {getattr(torch_module, '__version__', 'unknown')}")
    cuda_info = gather_cuda_info(torch_module)
    ascend_info = gather_ascend_info(torch_npu_module)

    if cuda_info.get("available"):
        print("CUDA 状态: 可用")
        print(f"CUDA 版本: {cuda_info.get('version', 'unknown')}")
        print(f"CUDA 设备: {cuda_info.get('device_name', 'Unknown')}" )
        print(f"显存: {bytes_to_human(cuda_info.get('memory'))}")
    else:
        print("CUDA 状态: 不可用")

    if ascend_info.get("available"):
        print("Ascend 状态: 可用")
        print(f"torch_npu 版本: {ascend_info.get('version', 'unknown')}")
        print(f"Ascend 设备: {ascend_info.get('device_name', 'Ascend Device')}")
        print(f"Ascend 显存: {bytes_to_human(ascend_info.get('memory'))}")
    else:
        print("Ascend 状态: 不可用")

    target_device: str
    if cuda_info.get("available"):
        target_device = "cuda:0"
    elif ascend_info.get("available"):
        target_device = "npu:0"
    else:
        print("警告: 未检测到 CUDA 或 Ascend 加速器，回退到 CPU 进行测试。", file=sys.stderr)
        success = run_inference(torch_module, "cpu")
        if success:
            sys.exit(0)
        sys.exit(1)

    success = run_inference(torch_module, target_device)
    if success:
        sys.exit(0)
    print("硬件推理自检失败，尝试回退到 CPU。")
    if run_inference(torch_module, "cpu"):
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()