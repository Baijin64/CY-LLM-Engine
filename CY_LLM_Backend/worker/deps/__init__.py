"""
CY-LLM Engine Dependency Management System

提供硬件检测、依赖解析和安装推荐功能。
"""

import json
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class HardwareVendor(Enum):
    NVIDIA = "nvidia"
    HUAWEI = "huawei"
    AMD = "amd"
    CPU = "cpu"
    UNKNOWN = "unknown"


@dataclass
class HardwareProfile:
    """硬件配置信息"""

    vendor: HardwareVendor
    device_name: str
    compute_capability: Optional[str] = None
    vram_gb: Optional[float] = None
    driver_version: Optional[str] = None
    cuda_version: Optional[str] = None


@dataclass
class DependencyProfile:
    """依赖配置信息"""

    profile_id: str
    hardware: List[str]
    engine: str
    engine_version: str
    python: str
    dependencies: Dict[str, List[str]]
    env_vars: Dict[str, str]
    warnings: List[str]


class HardwareDetector:
    """硬件检测器 - 自动检测GPU/NPU类型"""

    def __init__(self):
        self.registry_path = Path(__file__).parent.parent / "deploy" / "dependency_registry.json"
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict:
        """加载依赖注册表"""
        if not self.registry_path.exists():
            raise FileNotFoundError(f"Registry not found: {self.registry_path}")
        with open(self.registry_path, "r") as f:
            return json.load(f)

    def detect(self) -> HardwareProfile:
        """检测当前硬件配置"""
        # 尝试检测NVIDIA GPU
        nvidia = self._detect_nvidia()
        if nvidia:
            return nvidia

        # 尝试检测华为Ascend
        ascend = self._detect_ascend()
        if ascend:
            return ascend

        # 回退到CPU
        return HardwareProfile(vendor=HardwareVendor.CPU, device_name="CPU Only", vram_gb=0.0)

    def _detect_nvidia(self) -> Optional[HardwareProfile]:
        """检测NVIDIA GPU"""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,compute_cap,memory.total,driver_version",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return None

            lines = result.stdout.strip().split("\n")
            if not lines:
                return None

            # 解析第一行
            parts = [p.strip() for p in lines[0].split(",")]
            if len(parts) >= 4:
                device_name = parts[0]
                compute_cap = parts[1]
                memory_str = parts[2]
                driver_version = parts[3]

                # 解析显存
                vram_gb = 0.0
                if "MiB" in memory_str:
                    vram_gb = float(memory_str.replace("MiB", "").strip()) / 1024
                elif "GiB" in memory_str:
                    vram_gb = float(memory_str.replace("GiB", "").strip())

                # 获取CUDA版本
                cuda_version = self._get_cuda_version()

                return HardwareProfile(
                    vendor=HardwareVendor.NVIDIA,
                    device_name=device_name,
                    compute_capability=compute_cap,
                    vram_gb=vram_gb,
                    driver_version=driver_version,
                    cuda_version=cuda_version,
                )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return None

    def _detect_ascend(self) -> Optional[HardwareProfile]:
        """检测华为Ascend NPU"""
        try:
            result = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return HardwareProfile(
                    vendor=HardwareVendor.HUAWEI, device_name="Ascend NPU", compute_capability=None
                )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return None

    def _get_cuda_version(self) -> Optional[str]:
        """获取CUDA版本"""
        try:
            result = subprocess.run(
                ["nvcc", "--version"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "release" in line:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == "release":
                                return parts[i + 1].rstrip(",")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return None

    def get_hardware_profile_id(self, hw: HardwareProfile) -> Optional[str]:
        """根据硬件配置获取profile ID"""
        hw_profiles = self.registry.get("hardware_profiles", {})

        if hw.vendor == HardwareVendor.NVIDIA and hw.compute_capability:
            cap = float(hw.compute_capability)
            if cap >= 8.9:
                return "nvidia_ada"
            elif cap >= 8.0:
                return "nvidia_ampere"
            elif cap >= 7.5:
                return "nvidia_turing"
        elif hw.vendor == HardwareVendor.HUAWEI:
            return "ascend_910b"
        elif hw.vendor == HardwareVendor.CPU:
            return "cpu_only"

        return None


class DependencyResolver:
    """依赖解析器 - 根据硬件+引擎推荐依赖配置"""

    def __init__(self):
        self.registry_path = Path(__file__).parent.parent / "deploy" / "dependency_registry.json"
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict:
        """加载依赖注册表"""
        with open(self.registry_path, "r") as f:
            return json.load(f)

    def resolve(
        self, hardware_id: str, engine: str, engine_version: Optional[str] = None
    ) -> Optional[DependencyProfile]:
        """
        解析依赖配置

        Args:
            hardware_id: 硬件profile ID (e.g., "nvidia_ampere")
            engine: 引擎名称 (e.g., "vllm")
            engine_version: 引擎版本 (可选，默认使用注册表中的版本)

        Returns:
            DependencyProfile 或 None
        """
        matrix = self.registry.get("compatibility_matrix", [])

        for entry in matrix:
            hw_match = hardware_id in entry.get("hardware", [])
            engine_match = entry.get("engine") == engine

            if hw_match and engine_match:
                if engine_version is None or entry.get("engine_version") == engine_version:
                    return DependencyProfile(
                        profile_id=entry.get("profile_id", ""),
                        hardware=entry.get("hardware", []),
                        engine=entry.get("engine", ""),
                        engine_version=entry.get("engine_version", ""),
                        python=entry.get("python", ">=3.10"),
                        dependencies=entry.get("dependencies", {}),
                        env_vars=entry.get("env_vars", {}),
                        warnings=entry.get("warnings", []),
                    )

        return None

    def list_available_profiles(self, hardware_id: Optional[str] = None) -> List[Dict]:
        """列出所有可用的依赖配置"""
        matrix = self.registry.get("compatibility_matrix", [])

        if hardware_id:
            return [m for m in matrix if hardware_id in m.get("hardware", [])]
        return matrix

    def generate_requirements(
        self, profile: DependencyProfile, mirror: Optional[str] = None
    ) -> str:
        """
        生成requirements.txt内容

        Args:
            profile: 依赖配置
            mirror: 镜像源名称 (tsinghua/aliyun/douban)

        Returns:
            requirements.txt 内容
        """
        lines = []

        # 添加镜像源
        if mirror:
            mirrors = self.registry.get("mirrors", {})
            if mirror in mirrors:
                m = mirrors[mirror]
                lines.append(f"--index-url {m['url']}")
                lines.append(f"--trusted-host {m['trusted_host']}")
                lines.append("")

        # 添加基础依赖
        base_deps = self.registry.get("base_dependencies", [])
        lines.append("# Base Dependencies")
        for dep in base_deps:
            lines.append(dep)
        lines.append("")

        # 添加引擎特定依赖
        for category, deps in profile.dependencies.items():
            lines.append(f"# {category.title()} Dependencies")
            for dep in deps:
                lines.append(dep)
            lines.append("")

        return "\n".join(lines)

    def check_python_compatibility(self, profile: DependencyProfile) -> Tuple[bool, str]:
        """检查Python版本兼容性"""
        current = platform.python_version()
        requirement = profile.python

        # 简化的版本检查
        if ">=" in requirement and "<" in requirement:
            # 格式: >=3.10,<3.13
            min_ver = requirement.split(">=")[1].split(",")[0]
            max_ver = requirement.split("<")[1]

            current_tuple = tuple(map(int, current.split(".")[:2]))
            min_tuple = tuple(map(int, min_ver.split(".")[:2]))
            max_tuple = tuple(map(int, max_ver.split(".")[:2]))

            if current_tuple < min_tuple:
                return False, f"Python {current} 低于最低要求 {min_ver}"
            if current_tuple >= max_tuple:
                return False, f"Python {current} 高于最高要求 {max_ver}"

            return True, f"Python {current} 符合要求 {requirement}"

        return True, "无法解析版本要求，假设兼容"


def main():
    """CLI入口点"""
    import argparse

    parser = argparse.ArgumentParser(description="CY-LLM Dependency Manager")
    parser.add_argument("command", choices=["detect", "list", "resolve", "generate"])
    parser.add_argument("--hardware", help="Hardware profile ID")
    parser.add_argument("--engine", help="Engine name")
    parser.add_argument("--mirror", help="Mirror name (tsinghua/aliyun/douban)")
    parser.add_argument("--output", help="Output file path")

    args = parser.parse_args()

    if args.command == "detect":
        print("🔍 检测硬件配置...")
        detector = HardwareDetector()
        hw = detector.detect()
        print(f"  厂商: {hw.vendor.value}")
        print(f"  设备: {hw.device_name}")
        if hw.compute_capability:
            print(f"  计算能力: {hw.compute_capability}")
        if hw.vram_gb:
            print(f"  显存: {hw.vram_gb:.1f} GB")
        if hw.driver_version:
            print(f"  驱动: {hw.driver_version}")
        if hw.cuda_version:
            print(f"  CUDA: {hw.cuda_version}")

        profile_id = detector.get_hardware_profile_id(hw)
        if profile_id:
            print(f"\n📋 硬件Profile ID: {profile_id}")

    elif args.command == "list":
        resolver = DependencyResolver()
        profiles = resolver.list_available_profiles(args.hardware)
        print(f"可用配置 ({len(profiles)} 个):")
        for p in profiles:
            print(f"  - {p['profile_id']}: {p['engine']} {p['engine_version']}")

    elif args.command == "resolve":
        if not args.hardware or not args.engine:
            print("错误: --hardware 和 --engine 参数必需")
            return

        resolver = DependencyResolver()
        profile = resolver.resolve(args.hardware, args.engine)

        if profile:
            print(f"✅ 找到配置: {profile.profile_id}")
            print(f"   引擎: {profile.engine} {profile.engine_version}")
            print(f"   Python: {profile.python}")

            ok, msg = resolver.check_python_compatibility(profile)
            print(f"   Python兼容性: {'✅' if ok else '❌'} {msg}")

            if profile.warnings:
                print("\n⚠️  警告:")
                for w in profile.warnings:
                    print(f"   - {w}")
        else:
            print(f"❌ 未找到 {args.hardware} + {args.engine} 的配置")

    elif args.command == "generate":
        if not args.hardware or not args.engine:
            print("错误: --hardware 和 --engine 参数必需")
            return

        resolver = DependencyResolver()
        profile = resolver.resolve(args.hardware, args.engine)

        if profile:
            content = resolver.generate_requirements(profile, args.mirror)
            if args.output:
                with open(args.output, "w") as f:
                    f.write(content)
                print(f"✅ 已生成: {args.output}")
            else:
                print(content)
        else:
            print(f"❌ 未找到 {args.hardware} + {args.engine} 的配置")


if __name__ == "__main__":
    main()
