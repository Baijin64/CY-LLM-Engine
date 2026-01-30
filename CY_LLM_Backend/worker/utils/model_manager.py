"""
model_manager.py
[模型管理器] 统一管理模型的检测、下载、验证、加载全流程

功能：
  - 本地模型检测：优先使用已下载的模型
  - 下载管理：带进度条的模型下载
  - 模型验证：完整性校验（SHA256）
  - 量化选择：交互式量化格式选择界面
  - 分片加载：大模型分片加载支持
  - 智能超时：进度停滞检测，避免误重试
  - 异步加载：可选的后台加载模式
  - 配置保存：Safetensors 格式 + 元数据 YAML

作者：CY-LLM Team
版本：4.4.0.0-Alpha
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import yaml

LOGGER = logging.getLogger("cy_llm.worker.utils.model_manager")


# ============================================================================
# 枚举与数据类
# ============================================================================

class QuantizationType(Enum):
    """量化类型枚举"""
    FP16 = "fp16"
    FP8 = "fp8"
    INT8 = "int8"
    INT4 = "int4"
    AWQ = "awq"
    GPTQ = "gptq"
    FP4 = "fp4"  # TensorRT-LLM 专用（新 NVIDIA 显卡）
    NONE = "none"

    @classmethod
    def get_vllm_supported(cls) -> List["QuantizationType"]:
        """vLLM 支持的量化类型"""
        return [cls.FP16, cls.FP8, cls.INT8, cls.AWQ, cls.GPTQ, cls.NONE]

    @classmethod
    def get_trt_supported(cls) -> List["QuantizationType"]:
        """TensorRT-LLM 支持的量化类型"""
        return [cls.FP16, cls.FP8, cls.INT8, cls.INT4, cls.FP4, cls.NONE]

    def get_vram_multiplier(self) -> float:
        """获取相对于 FP16 的 VRAM 倍数"""
        multipliers = {
            QuantizationType.FP16: 1.0,
            QuantizationType.FP8: 0.5,
            QuantizationType.INT8: 0.5,
            QuantizationType.INT4: 0.25,
            QuantizationType.AWQ: 0.25,
            QuantizationType.GPTQ: 0.25,
            QuantizationType.FP4: 0.25,
            QuantizationType.NONE: 1.0,
        }
        return multipliers.get(self, 1.0)


class LoadMode(Enum):
    """加载模式"""
    CHECK_ONLY = "check_only"  # 仅检查，不加载
    CHECK_AND_LOAD = "check_and_load"  # 检查并加载（默认）
    ASYNC_LOAD = "async_load"  # 异步加载


class ModelStage(Enum):
    """模型加载阶段"""
    DETECTING = "detecting"
    DOWNLOADING = "downloading"
    VALIDATING = "validating"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


@dataclass
class ProgressInfo:
    """进度信息"""
    stage: ModelStage
    progress: float  # 0.0 - 100.0
    message: str
    speed: Optional[str] = None  # 下载速度
    eta: Optional[str] = None  # 预计剩余时间
    last_update: float = field(default_factory=time.time)

    def is_stalled(self, stall_threshold_seconds: float = 180.0) -> bool:
        """检查进度是否停滞（默认 3 分钟）"""
        return time.time() - self.last_update > stall_threshold_seconds


@dataclass
class ModelMetadata:
    """模型元数据"""
    model_id: str
    source: str  # "huggingface", "local", "modelscope"
    download_date: str
    format: str  # "safetensors", "pytorch", "gguf"
    quantization: Optional[str] = None
    original_config: Optional[Dict[str, Any]] = None
    files: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_yaml(self) -> str:
        """导出为 YAML 格式"""
        return yaml.dump(self.__dict__, allow_unicode=True, sort_keys=False)
    
    @classmethod
    def from_yaml(cls, yaml_str: str) -> "ModelMetadata":
        """从 YAML 加载"""
        data = yaml.safe_load(yaml_str)
        return cls(**data)
    
    def save(self, path: Path) -> None:
        """保存元数据到文件"""
        meta_path = path / "model_meta.yaml"
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(self.to_yaml())
        LOGGER.info("模型元数据已保存: %s", meta_path)

    @classmethod
    def load(cls, path: Path) -> Optional["ModelMetadata"]:
        """从文件加载元数据"""
        meta_path = path / "model_meta.yaml"
        if not meta_path.exists():
            return None
        with open(meta_path, "r", encoding="utf-8") as f:
            return cls.from_yaml(f.read())


@dataclass
class QuantizationOption:
    """量化选项（用于选择界面）"""
    quant_type: QuantizationType
    display_name: str
    vram_estimate_gb: float
    description: str
    engine_support: List[str]  # ["vllm", "trt"]


# ============================================================================
# 进度追踪器
# ============================================================================

class ProgressTracker:
    """
    进度追踪器
    
    功能：
      - 追踪各阶段进度
      - 检测进度停滞
      - 支持回调通知
    """

    def __init__(
        self,
        stall_threshold_seconds: float = 180.0,  # 3 分钟
        on_progress: Optional[Callable[[ProgressInfo], None]] = None,
        on_stall: Optional[Callable[[ProgressInfo], None]] = None,
    ):
        self.stall_threshold = stall_threshold_seconds
        self.on_progress = on_progress
        self.on_stall = on_stall
        self._current: Optional[ProgressInfo] = None
        self._lock = threading.Lock()
        self._stall_check_thread: Optional[threading.Thread] = None
        self._running = False

    def start(self, stage: ModelStage, message: str = "") -> None:
        """开始追踪"""
        with self._lock:
            self._current = ProgressInfo(
                stage=stage,
                progress=0.0,
                message=message,
                last_update=time.time()
            )
            self._running = True
        
        # 启动停滞检测线程
        self._stall_check_thread = threading.Thread(
            target=self._stall_check_loop,
            daemon=True
        )
        self._stall_check_thread.start()
        self._notify_progress()

    def update(
        self,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        speed: Optional[str] = None,
        eta: Optional[str] = None,
    ) -> None:
        """更新进度"""
        with self._lock:
            if self._current is None:
                return
            
            if progress is not None:
                self._current.progress = progress
            if message is not None:
                self._current.message = message
            if speed is not None:
                self._current.speed = speed
            if eta is not None:
                self._current.eta = eta
            
            self._current.last_update = time.time()
        
        self._notify_progress()

    def set_stage(self, stage: ModelStage, message: str = "") -> None:
        """切换阶段"""
        with self._lock:
            if self._current is None:
                self._current = ProgressInfo(
                    stage=stage,
                    progress=0.0,
                    message=message,
                    last_update=time.time()
                )
            else:
                self._current.stage = stage
                self._current.progress = 0.0
                self._current.message = message
                self._current.last_update = time.time()
        
        self._notify_progress()

    def finish(self, success: bool = True) -> None:
        """完成追踪"""
        with self._lock:
            self._running = False
            if self._current:
                self._current.stage = ModelStage.READY if success else ModelStage.FAILED
                self._current.progress = 100.0 if success else self._current.progress
        
        self._notify_progress()

    def get_current(self) -> Optional[ProgressInfo]:
        """获取当前进度"""
        with self._lock:
            return self._current

    def _notify_progress(self) -> None:
        """通知进度更新"""
        if self.on_progress and self._current:
            try:
                self.on_progress(self._current)
            except Exception as e:
                LOGGER.warning("进度回调失败: %s", e)

    def _stall_check_loop(self) -> None:
        """停滞检测循环：简单直接的超时检查"""
        while self._running:
            time.sleep(10)  # 每 10 秒检查一次
            with self._lock:
                if not self._current:
                    continue
                
                # 检查是否超时
                stall_duration = time.time() - self._current.last_update
                
                # 特殊处理下载阶段：如果文件正在变化，重置计时器
                if self._current.stage == ModelStage.DOWNLOADING:
                    try:
                        # 识别模型 ID 并探测下载缓存目录
                        id_part = self._current.message.split(': ')[-1].strip()
                        search_paths = [
                            Path.home() / ".cache" / "cy-llm" / "models" / id_part.replace("/", "--"),
                            Path.home() / ".cache" / "huggingface" / "hub" / f"models--{id_part.replace('/', '--')}"
                        ]
                        for path in search_paths:
                            if path.exists():
                                current_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                                if hasattr(self, '_last_size') and current_size > self._last_size:
                                    self._current.last_update = time.time()
                                    stall_duration = 0.0
                                self._last_size = current_size
                                break
                    except Exception:
                        pass

                # 执行强制中止
                if stall_duration > self.stall_threshold:
                    if self.on_stall:
                        try:
                            self.on_stall(self._current)
                        except Exception:
                            pass
                    self._running = False
                    break

    def reset_stall_count(self) -> None:
        """兼容性接口：重置状态快照"""
        with self._lock:
            if hasattr(self, '_last_size'):
                delattr(self, '_last_size')


# ============================================================================
# 模型检测器
# ============================================================================

class ModelDetector:
    """
    模型检测器
    
    功能：
      - 检测本地缓存
      - 搜索多个缓存路径
      - 验证模型完整性
    """

    # 默认缓存路径搜索顺序
    DEFAULT_CACHE_PATHS = [
        Path.home() / ".cache" / "huggingface" / "hub",
        Path.home() / ".cache" / "modelscope" / "hub",
        Path.home() / ".cache" / "cy-llm" / "models",
        Path("/models"),  # Docker 挂载路径
    ]

    def __init__(self, additional_paths: Optional[List[Path]] = None):
        self.cache_paths = list(self.DEFAULT_CACHE_PATHS)
        if additional_paths:
            self.cache_paths = additional_paths + self.cache_paths

    def detect(self, model_id: str) -> Optional[Path]:
        """
        检测模型是否已存在于本地
        
        Args:
            model_id: 模型 ID（如 "deepseek-ai/deepseek-llm-7b-chat"）
            
        Returns:
            模型路径（如果找到），否则 None
        """
        # 1. 如果是绝对路径，直接检查
        if os.path.isabs(model_id):
            path = Path(model_id)
            if self._is_valid_model_dir(path):
                LOGGER.info("✓ 找到本地模型（绝对路径）: %s", path)
                return path
            return None

        # 2. 搜索缓存路径
        # HuggingFace 格式: models--{org}--{name}
        hf_cache_name = f"models--{model_id.replace('/', '--')}"
        
        for cache_path in self.cache_paths:
            if not cache_path.exists():
                continue

            # 直接匹配
            direct_path = cache_path / model_id.replace("/", "--")
            if self._is_valid_model_dir(direct_path):
                LOGGER.info("✓ 找到本地模型: %s", direct_path)
                return direct_path

            # HuggingFace 缓存格式
            hf_path = cache_path / hf_cache_name
            if hf_path.exists():
                # 查找 snapshots 目录中的最新版本
                snapshots_dir = hf_path / "snapshots"
                if snapshots_dir.exists():
                    snapshots = list(snapshots_dir.iterdir())
                    if snapshots:
                        latest = max(snapshots, key=lambda p: p.stat().st_mtime)
                        if self._is_valid_model_dir(latest):
                            LOGGER.info("✓ 找到本地模型（HF 缓存）: %s", latest)
                            return latest

            # 简单名称匹配
            simple_name = model_id.split("/")[-1]
            simple_path = cache_path / simple_name
            if self._is_valid_model_dir(simple_path):
                LOGGER.info("✓ 找到本地模型: %s", simple_path)
                return simple_path

        LOGGER.info("✗ 本地未找到模型: %s", model_id)
        return None

    def _is_valid_model_dir(self, path: Path) -> bool:
        """检查是否是有效的模型目录"""
        if not path.exists() or not path.is_dir():
            return False

        # 检查必要权重文件 (必须包含至少一个权重文件，仅有 config.json 不够)
        required_weights = [
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
            "model.gguf",
        ]
        
        for filename in required_weights:
            if (path / filename).exists():
                return True

        # 检查分片文件
        safetensor_shards = list(path.glob("model-*.safetensors"))
        pytorch_shards = list(path.glob("pytorch_model-*.bin"))
        
        return len(safetensor_shards) > 0 or len(pytorch_shards) > 0

    def get_model_format(self, path: Path) -> str:
        """获取模型格式"""
        if list(path.glob("*.safetensors")):
            return "safetensors"
        elif list(path.glob("*.bin")):
            return "pytorch"
        elif list(path.glob("*.gguf")):
            return "gguf"
        return "unknown"

    def get_model_size(self, path: Path) -> int:
        """获取模型文件总大小（字节）"""
        total = 0
        for pattern in ["*.safetensors", "*.bin", "*.gguf"]:
            for f in path.glob(pattern):
                total += f.stat().st_size
        return total


# ============================================================================
# 模型下载器
# ============================================================================

class ModelDownloader:
    """
    模型下载器
    
    功能：
      - 带进度条的下载
      - 断点续传
      - 多源支持（HuggingFace, ModelScope）
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        progress_tracker: Optional[ProgressTracker] = None,
    ):
        self.cache_dir = cache_dir or (Path.home() / ".cache" / "cy-llm" / "models")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.progress_tracker = progress_tracker

    def download(
        self,
        model_id: str,
        revision: str = "main",
        source: str = "huggingface",
    ) -> Path:
        """
        下载模型
        
        Args:
            model_id: 模型 ID
            revision: 版本/分支
            source: 来源（huggingface, modelscope）
            
        Returns:
            下载后的模型路径
        """
        if self.progress_tracker:
            self.progress_tracker.set_stage(
                ModelStage.DOWNLOADING,
                f"正在下载: {model_id}"
            )

        try:
            if source == "huggingface":
                return self._download_from_hf(model_id, revision)
            elif source == "modelscope":
                return self._download_from_modelscope(model_id, revision)
            else:
                raise ValueError(f"不支持的下载源: {source}")
        except Exception as e:
            LOGGER.error("下载失败: %s", e)
            if self.progress_tracker:
                self.progress_tracker.finish(success=False)
            raise

    def _download_from_hf(self, model_id: str, revision: str) -> Path:
        """从 HuggingFace 下载"""
        try:
            from huggingface_hub import snapshot_download, HfApi
            from huggingface_hub.utils import tqdm as hf_tqdm
        except ImportError:
            raise ImportError("请安装 huggingface-hub: pip install huggingface-hub")

        # 获取模型文件列表和总大小
        api = HfApi()
        try:
            repo_info = api.repo_info(model_id, revision=revision)
            # 某些版本的 hf_hub 返回的 siblings 可能没有 size，或者需要 files 参数
            total_size = 0
            if hasattr(repo_info, "siblings") and repo_info.siblings:
                for f in repo_info.siblings:
                    if self._is_model_file(f.rfilename):
                        # 尝试获取 size，如果不存在则设为 0
                        f_size = getattr(f, "size", 0) or 0
                        total_size += f_size
            
            if total_size > 0:
                LOGGER.info("模型权重总大小: %.2f GB", total_size / (1024**3))
            else:
                LOGGER.info("无法预估模型精细大小，将按文件数追踪进度")
        except Exception as e:
            LOGGER.warning("无法获取模型信息: %s", e)
            total_size = 0

        progress_tracker = self.progress_tracker

        class _TrackerTqdm(hf_tqdm):
            def __init__(self, *args, **kwargs):
                # 禁用默认的 tqdm 文本输出，我们使用自己的 ProgressTracker 渲染
                kwargs["disable"] = True 
                super().__init__(*args, **kwargs)

            def update(self, n=1):  # type: ignore[override]
                out = super().update(n)
                if not progress_tracker:
                    return out
                try:
                    current_n = float(self.n)
                    total_n = float(self.total) if self.total else 0.0
                    
                    if total_n > 0:
                        progress = (current_n / total_n) * 100.0
                    else:
                        progress = 0.0

                    speed = ""
                    eta = ""
                    # 尝试从 tqdm 实例获取速率和剩余时间
                    rate = getattr(self, "avg_rate", None) or getattr(self, "rate", None)
                    if rate:
                        speed = self._format_speed_internal(float(rate))
                    
                    remaining = getattr(self, "remaining", None)
                    if remaining is not None:
                        eta = self._format_eta_internal(float(remaining))

                    msg = f"正在下载: {current_n:.0f}/{total_n:.0f}" if total_n > 0 else "正在下载..."
                    progress_tracker.update(
                        progress=min(99.9, max(0.0, progress)),
                        message=msg,
                        speed=speed,
                        eta=eta,
                    )
                except Exception:
                    pass
                return out

            def _format_speed_internal(self, val):
                if val < 1024: return f"{val:.0f} B/s"
                if val < 1024**2: return f"{val/1024:.1f} KB/s"
                return f"{val/1024**2:.1f} MB/s"

            def _format_eta_internal(self, val):
                if val < 60: return f"{val:.0f}s"
                return f"{val/60:.1f}m"

        # 执行下载
        local_dir = self.cache_dir / model_id.replace("/", "--")
        
        # 调试日志：检查环境变量
        LOGGER.info("正在通过 %s 下载模型...", os.getenv("HF_ENDPOINT", "HuggingFace Official"))

        # 使用 snapshot_download 并配置进度
        result_path = snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_dir=str(local_dir),
            token=os.getenv("HUGGING_FACE_HUB_TOKEN"),
            tqdm_class=_TrackerTqdm,
            max_workers=4,  # 多线程加速
        )

        if self.progress_tracker:
            self.progress_tracker.update(progress=100.0, message="下载完成")

        return Path(result_path)

    def _download_from_modelscope(self, model_id: str, revision: str) -> Path:
        """从 ModelScope 下载"""
        try:
            from modelscope import snapshot_download as ms_download
        except ImportError:
            raise ImportError("请安装 modelscope: pip install modelscope")

        local_dir = self.cache_dir / model_id.replace("/", "--")
        
        result_path = ms_download(
            model_id,
            revision=revision,
            cache_dir=str(self.cache_dir),
        )

        if self.progress_tracker:
            self.progress_tracker.update(progress=100.0, message="下载完成")

        return Path(result_path)

    def _is_model_file(self, filename: str) -> bool:
        """判断是否是模型文件"""
        model_extensions = [".safetensors", ".bin", ".pt", ".gguf"]
        return any(filename.endswith(ext) for ext in model_extensions)

    def _format_speed(self, bytes_per_second: float) -> str:
        """格式化下载速度"""
        if bytes_per_second < 1024:
            return f"{bytes_per_second:.0f} B/s"
        elif bytes_per_second < 1024**2:
            return f"{bytes_per_second/1024:.1f} KB/s"
        elif bytes_per_second < 1024**3:
            return f"{bytes_per_second/1024**2:.1f} MB/s"
        else:
            return f"{bytes_per_second/1024**3:.2f} GB/s"

    def _format_eta(self, seconds: float) -> str:
        """格式化预计剩余时间"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.0f}m"
        else:
            return f"{seconds/3600:.1f}h"


# ============================================================================
# 模型验证器
# ============================================================================

class ModelValidator:
    """
    模型验证器
    
    功能：
      - 文件完整性校验（SHA256）
      - 配置文件验证
      - 分片索引验证
    """

    def __init__(self, progress_tracker: Optional[ProgressTracker] = None):
        self.progress_tracker = progress_tracker

    def validate(self, model_path: Path) -> Tuple[bool, List[str]]:
        """
        验证模型完整性
        
        Args:
            model_path: 模型路径
            
        Returns:
            (是否通过, 错误列表)
        """
        if self.progress_tracker:
            self.progress_tracker.set_stage(
                ModelStage.VALIDATING,
                "正在验证模型完整性..."
            )

        errors = []
        
        # 1. 检查必要文件
        config_path = model_path / "config.json"
        if not config_path.exists():
            errors.append("缺少 config.json")

        # 2. 验证分片索引
        safetensor_index = model_path / "model.safetensors.index.json"
        pytorch_index = model_path / "pytorch_model.bin.index.json"
        
        if safetensor_index.exists():
            index_errors = self._validate_index(model_path, safetensor_index, ".safetensors")
            errors.extend(index_errors)
        elif pytorch_index.exists():
            index_errors = self._validate_index(model_path, pytorch_index, ".bin")
            errors.extend(index_errors)

        # 3. 验证单文件模型
        single_safetensor = model_path / "model.safetensors"
        single_pytorch = model_path / "pytorch_model.bin"
        
        if not safetensor_index.exists() and not pytorch_index.exists():
            if not single_safetensor.exists() and not single_pytorch.exists():
                # 检查分片文件
                shards = list(model_path.glob("model-*.safetensors")) + \
                         list(model_path.glob("pytorch_model-*.bin"))
                if not shards:
                    errors.append("未找到模型权重文件")

        # 更新进度
        if self.progress_tracker:
            if errors:
                self.progress_tracker.update(progress=100.0, message=f"验证失败: {len(errors)} 个错误")
            else:
                self.progress_tracker.update(progress=100.0, message="验证通过")

        return len(errors) == 0, errors

    def _validate_index(
        self,
        model_path: Path,
        index_path: Path,
        extension: str
    ) -> List[str]:
        """验证分片索引文件"""
        errors = []
        
        try:
            with open(index_path, "r") as f:
                index = json.load(f)
        except Exception as e:
            errors.append(f"无法读取索引文件: {e}")
            return errors

        weight_map = index.get("weight_map", {})
        required_files = set(weight_map.values())
        
        total_files = len(required_files)
        checked = 0
        
        for filename in required_files:
            file_path = model_path / filename
            if not file_path.exists():
                errors.append(f"缺少分片文件: {filename}")
            
            checked += 1
            if self.progress_tracker:
                progress = (checked / total_files) * 100
                self.progress_tracker.update(
                    progress=progress,
                    message=f"验证分片: {checked}/{total_files}"
                )

        return errors

    def compute_file_hash(self, file_path: Path, chunk_size: int = 8192) -> str:
        """计算文件 SHA256 哈希"""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                sha256.update(chunk)
        return sha256.hexdigest()


# ============================================================================
# 量化选择器
# ============================================================================

class QuantizationSelector:
    """
    量化格式选择器
    
    功能：
      - 交互式选择界面
      - VRAM 预估
      - 引擎兼容性提示
    """

    def __init__(self, engine_type: str = "vllm"):
        self.engine_type = engine_type

    def get_options(
        self,
        model_params_billion: float,
        available_vram_gb: float,
    ) -> List[QuantizationOption]:
        """
        获取可用的量化选项
        
        Args:
            model_params_billion: 模型参数量（十亿）
            available_vram_gb: 可用显存（GB）
            
        Returns:
            量化选项列表
        """
        # FP16 基准 VRAM 估算: 参数量 * 2 bytes + 开销
        base_vram = model_params_billion * 2 * 1.2  # 1.2 是开销系数

        options = []

        # 根据引擎类型获取支持的量化类型
        if self.engine_type in ["vllm", "cuda-vllm"]:
            supported = QuantizationType.get_vllm_supported()
        elif self.engine_type in ["trt", "cuda-trt"]:
            supported = QuantizationType.get_trt_supported()
        else:
            supported = list(QuantizationType)

        for qtype in supported:
            vram_estimate = base_vram * qtype.get_vram_multiplier()
            
            # 生成描述
            descriptions = {
                QuantizationType.FP16: "原始精度，质量最高",
                QuantizationType.FP8: "半精度压缩，平衡质量与速度",
                QuantizationType.INT8: "8位整数量化，速度快",
                QuantizationType.INT4: "4位整数量化，显存最省",
                QuantizationType.AWQ: "AWQ 4位量化，质量优秀",
                QuantizationType.GPTQ: "GPTQ 4位量化，通用兼容",
                QuantizationType.FP4: "FP4 量化 (TRT专用，需要新显卡)",
                QuantizationType.NONE: "不使用量化",
            }

            # 引擎支持
            engine_support = []
            if qtype in QuantizationType.get_vllm_supported():
                engine_support.append("vllm")
            if qtype in QuantizationType.get_trt_supported():
                engine_support.append("trt")

            # 特殊标记
            display_name = qtype.value.upper()
            if vram_estimate > available_vram_gb:
                display_name += " ⚠️ (显存不足)"
            elif vram_estimate < available_vram_gb * 0.7:
                display_name += " ✓ (推荐)"

            options.append(QuantizationOption(
                quant_type=qtype,
                display_name=display_name,
                vram_estimate_gb=vram_estimate,
                description=descriptions.get(qtype, ""),
                engine_support=engine_support,
            ))

        return options

    def select_interactive(
        self,
        model_params_billion: float,
        available_vram_gb: float,
    ) -> QuantizationType:
        """
        交互式选择量化格式
        
        Args:
            model_params_billion: 模型参数量（十亿）
            available_vram_gb: 可用显存（GB）
            
        Returns:
            选择的量化类型
        """
        options = self.get_options(model_params_billion, available_vram_gb)

        print("\n" + "=" * 60)
        print(f"  选择量化格式 (引擎: {self.engine_type})")
        print("=" * 60)
        print(f"  模型参数量: {model_params_billion:.1f}B")
        print(f"  可用显存: {available_vram_gb:.1f} GB")
        print("-" * 60)

        for i, opt in enumerate(options):
            status = "✓" if opt.vram_estimate_gb <= available_vram_gb else "⚠️"
            print(f"  [{i}] {status} {opt.display_name:<12} - 需要 ~{opt.vram_estimate_gb:.1f}GB")
            print(f"      {opt.description}")
            print(f"      支持引擎: {', '.join(opt.engine_support)}")

        print("-" * 60)
        print("  [A] 自动选择 (根据可用显存)")
        print("=" * 60)

        while True:
            choice = input("\n请选择 [0-{}/A]: ".format(len(options) - 1)).strip().upper()
            
            if choice == "A":
                # 自动选择：在显存允许范围内选择质量最高的
                for opt in options:
                    if opt.vram_estimate_gb <= available_vram_gb * 0.9:
                        print(f"\n✓ 自动选择: {opt.quant_type.value}")
                        return opt.quant_type
                # 如果都不够，选择最省显存的
                return options[-1].quant_type

            try:
                idx = int(choice)
                if 0 <= idx < len(options):
                    selected = options[idx]
                    if selected.vram_estimate_gb > available_vram_gb:
                        confirm = input(f"⚠️ 显存可能不足，是否继续？[y/N]: ").strip().lower()
                        if confirm != "y":
                            continue
                    return selected.quant_type
            except ValueError:
                pass

            print("❌ 无效选择，请重试")

    def select_auto(
        self,
        model_params_billion: float,
        available_vram_gb: float,
    ) -> QuantizationType:
        """
        自动选择最佳量化格式
        
        Args:
            model_params_billion: 模型参数量（十亿）
            available_vram_gb: 可用显存（GB）
            
        Returns:
            推荐的量化类型
        """
        options = self.get_options(model_params_billion, available_vram_gb)
        
        # 按 VRAM 需求从高到低排序，选择能用的最高质量
        for opt in options:
            if opt.vram_estimate_gb <= available_vram_gb * 0.85:  # 留 15% 余量
                LOGGER.info("自动选择量化格式: %s (预计 %.1fGB)", 
                           opt.quant_type.value, opt.vram_estimate_gb)
                return opt.quant_type
        
        # 兜底：选择最省显存的
        return options[-1].quant_type


# ============================================================================
# 分片加载器
# ============================================================================

class ShardedLoader:
    """
    分片加载器
    
    功能：
      - 逐分片加载大模型
      - 内存效率优化
      - 进度追踪
    """

    def __init__(self, progress_tracker: Optional[ProgressTracker] = None):
        self.progress_tracker = progress_tracker

    def load_sharded_weights(
        self,
        model_path: Path,
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """
        分片加载模型权重
        
        Args:
            model_path: 模型路径
            device: 目标设备
            
        Returns:
            权重字典
        """
        # 延迟导入 torch
        try:
            import torch
        except ImportError:
            raise ImportError("请安装 PyTorch: pip install torch")

        if self.progress_tracker:
            self.progress_tracker.set_stage(
                ModelStage.LOADING,
                "正在分片加载模型..."
            )

        # 检查索引文件
        safetensor_index = model_path / "model.safetensors.index.json"
        pytorch_index = model_path / "pytorch_model.bin.index.json"

        if safetensor_index.exists():
            return self._load_safetensor_shards(model_path, safetensor_index, device)
        elif pytorch_index.exists():
            return self._load_pytorch_shards(model_path, pytorch_index, device)
        else:
            # 单文件模型
            return self._load_single_file(model_path, device)

    def _load_safetensor_shards(
        self,
        model_path: Path,
        index_path: Path,
        device: str,
    ) -> Dict[str, Any]:
        """加载 Safetensors 分片"""
        try:
            from safetensors import safe_open
        except ImportError:
            raise ImportError("请安装 safetensors: pip install safetensors")

        with open(index_path, "r") as f:
            index = json.load(f)

        weight_map = index.get("weight_map", {})
        shard_files = list(set(weight_map.values()))
        total_shards = len(shard_files)

        weights = {}
        for i, shard_file in enumerate(shard_files):
            shard_path = model_path / shard_file
            
            if self.progress_tracker:
                progress = ((i + 1) / total_shards) * 100
                self.progress_tracker.update(
                    progress=progress,
                    message=f"加载分片: {i+1}/{total_shards}"
                )

            with safe_open(str(shard_path), framework="pt", device=device) as f:
                for key in f.keys():
                    weights[key] = f.get_tensor(key)

            LOGGER.debug("已加载分片 %d/%d: %s", i + 1, total_shards, shard_file)

        return weights

    def _load_pytorch_shards(
        self,
        model_path: Path,
        index_path: Path,
        device: str,
    ) -> Dict[str, Any]:
        """加载 PyTorch 分片"""
        import torch

        with open(index_path, "r") as f:
            index = json.load(f)

        weight_map = index.get("weight_map", {})
        shard_files = list(set(weight_map.values()))
        total_shards = len(shard_files)

        weights = {}
        for i, shard_file in enumerate(shard_files):
            shard_path = model_path / shard_file
            
            if self.progress_tracker:
                progress = ((i + 1) / total_shards) * 100
                self.progress_tracker.update(
                    progress=progress,
                    message=f"加载分片: {i+1}/{total_shards}"
                )

            shard_weights = torch.load(
                str(shard_path),
                map_location=device,
                weights_only=True
            )
            weights.update(shard_weights)

            LOGGER.debug("已加载分片 %d/%d: %s", i + 1, total_shards, shard_file)

        return weights

    def _load_single_file(self, model_path: Path, device: str) -> Dict[str, Any]:
        """加载单文件模型"""
        safetensor_file = model_path / "model.safetensors"
        pytorch_file = model_path / "pytorch_model.bin"

        if self.progress_tracker:
            self.progress_tracker.update(progress=50.0, message="加载模型文件...")

        if safetensor_file.exists():
            try:
                from safetensors import safe_open
                with safe_open(str(safetensor_file), framework="pt", device=device) as f:
                    weights = {key: f.get_tensor(key) for key in f.keys()}
            except ImportError:
                raise ImportError("请安装 safetensors: pip install safetensors")
        elif pytorch_file.exists():
            import torch
            weights = torch.load(str(pytorch_file), map_location=device, weights_only=True)
        else:
            raise FileNotFoundError("未找到模型权重文件")

        if self.progress_tracker:
            self.progress_tracker.update(progress=100.0, message="模型加载完成")

        return weights


# ============================================================================
# 配置保存器
# ============================================================================

class ConfigPreserver:
    """
    配置保存器
    
    功能：
      - 保存模型配置到 YAML
      - 备份原始 config.json
      - 生成模型元数据
    """

    def save_metadata(
        self,
        model_path: Path,
        model_id: str,
        source: str = "huggingface",
        quantization: Optional[str] = None,
    ) -> ModelMetadata:
        """
        保存模型元数据
        
        Args:
            model_path: 模型路径
            model_id: 模型 ID
            source: 来源
            quantization: 量化方法
            
        Returns:
            ModelMetadata 对象
        """
        # 读取原始配置
        config_path = model_path / "config.json"
        original_config = None
        if config_path.exists():
            with open(config_path, "r") as f:
                original_config = json.load(f)

        # 收集文件信息
        files = []
        for pattern in ["*.safetensors", "*.bin", "*.json", "*.txt"]:
            for f in model_path.glob(pattern):
                files.append({
                    "name": f.name,
                    "size_bytes": f.stat().st_size,
                })

        # 确定格式
        if list(model_path.glob("*.safetensors")):
            model_format = "safetensors"
        elif list(model_path.glob("*.bin")):
            model_format = "pytorch"
        else:
            model_format = "unknown"

        # 创建元数据
        metadata = ModelMetadata(
            model_id=model_id,
            source=source,
            download_date=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            format=model_format,
            quantization=quantization,
            original_config=original_config,
            files=files,
        )

        # 保存
        metadata.save(model_path)

        return metadata

    def convert_to_safetensors(
        self,
        model_path: Path,
        progress_tracker: Optional[ProgressTracker] = None,
    ) -> bool:
        """
        将 PyTorch 格式转换为 Safetensors
        
        Args:
            model_path: 模型路径
            progress_tracker: 进度追踪器
            
        Returns:
            是否转换成功
        """
        pytorch_files = list(model_path.glob("pytorch_model*.bin"))
        if not pytorch_files:
            LOGGER.info("无需转换：未找到 PyTorch 格式文件")
            return True

        try:
            import torch
            from safetensors.torch import save_file
        except ImportError:
            LOGGER.warning("无法转换：缺少 safetensors 库")
            return False

        total_files = len(pytorch_files)
        for i, pt_file in enumerate(pytorch_files):
            if progress_tracker:
                progress = ((i + 1) / total_files) * 100
                progress_tracker.update(
                    progress=progress,
                    message=f"转换格式: {i+1}/{total_files}"
                )

            # 加载 PyTorch 权重
            weights = torch.load(str(pt_file), map_location="cpu", weights_only=True)
            
            # 生成 safetensors 文件名
            st_file = pt_file.with_suffix(".safetensors")
            if "pytorch_model" in pt_file.name:
                st_file = model_path / pt_file.name.replace("pytorch_model", "model").replace(".bin", ".safetensors")
            
            # 保存
            save_file(weights, str(st_file))
            LOGGER.info("已转换: %s -> %s", pt_file.name, st_file.name)

            # 清理内存
            del weights

        # 更新索引文件
        pytorch_index = model_path / "pytorch_model.bin.index.json"
        if pytorch_index.exists():
            with open(pytorch_index, "r") as f:
                index = json.load(f)
            
            # 更新 weight_map
            new_weight_map = {}
            for key, filename in index.get("weight_map", {}).items():
                new_filename = filename.replace("pytorch_model", "model").replace(".bin", ".safetensors")
                new_weight_map[key] = new_filename
            
            index["weight_map"] = new_weight_map
            
            # 保存新索引
            st_index = model_path / "model.safetensors.index.json"
            with open(st_index, "w") as f:
                json.dump(index, f, indent=2)

        return True


# ============================================================================
# 模型管理器（统一入口）
# ============================================================================

class ModelManager:
    """
    模型管理器 - 统一管理模型的检测、下载、验证、加载全流程
    
    使用示例：
        >>> manager = ModelManager(engine_type="vllm")
        >>> 
        >>> # 方式1：检查并加载（默认）
        >>> path = manager.prepare_model("deepseek-ai/deepseek-llm-7b-chat")
        >>> 
        >>> # 方式2：仅检查
        >>> path = manager.prepare_model(
        ...     "deepseek-ai/deepseek-llm-7b-chat",
        ...     mode=LoadMode.CHECK_ONLY
        ... )
        >>> 
        >>> # 方式3：交互式选择量化
        >>> path = manager.prepare_model(
        ...     "deepseek-ai/deepseek-llm-7b-chat",
        ...     interactive=True
        ... )
    """

    def __init__(
        self,
        engine_type: str = "vllm",
        cache_dir: Optional[Path] = None,
        stall_threshold_seconds: float = 180.0,
        timeout_seconds: float = 600.0,  # 10 分钟
        on_progress: Optional[Callable[[ProgressInfo], None]] = None,
    ):
        """
        初始化模型管理器
        
        Args:
            engine_type: 引擎类型 ("vllm", "trt", "cuda-vllm", "cuda-trt")
            cache_dir: 模型缓存目录
            stall_threshold_seconds: 进度停滞阈值（秒）
            timeout_seconds: 总超时时间（秒）
            on_progress: 进度回调函数
        """
        self.engine_type = engine_type
        self.cache_dir = cache_dir or (Path.home() / ".cache" / "cy-llm" / "models")
        self.stall_threshold = stall_threshold_seconds
        self.timeout = timeout_seconds

        # 初始化组件
        self.progress_tracker = ProgressTracker(
            stall_threshold_seconds=stall_threshold_seconds,
            on_progress=on_progress or self._default_progress_callback,
            on_stall=self._on_stall,
        )
        self.detector = ModelDetector()
        self.downloader = ModelDownloader(cache_dir, self.progress_tracker)
        self.validator = ModelValidator(self.progress_tracker)
        self.quant_selector = QuantizationSelector(engine_type)
        self.sharded_loader = ShardedLoader(self.progress_tracker)
        self.config_preserver = ConfigPreserver()

        self._stall_count = 0
        self._max_stall_retries = 3

    def prepare_model(
        self,
        model_id: str,
        mode: LoadMode = LoadMode.CHECK_AND_LOAD,
        quantization: Optional[QuantizationType] = None,
        interactive: bool = False,
        source: str = "huggingface",
        skip_validation: bool = False,
    ) -> Tuple[Path, Optional[QuantizationType]]:
        """
        准备模型（检测 -> 下载 -> 验证 -> 可选加载）
        
        Args:
            model_id: 模型 ID 或路径
            mode: 加载模式
            quantization: 量化类型（None 则自动或交互选择）
            interactive: 是否启用交互式选择
            source: 下载源
            skip_validation: 跳过验证
            
        Returns:
            (模型路径, 选择的量化类型)
        """
        start_time = time.time()
        self._stall_count = 0

        try:
            # === 阶段 1: 检测本地模型 ===
            self.progress_tracker.start(ModelStage.DETECTING, f"检测模型: {model_id}")
            
            model_path = self.detector.detect(model_id)
            
            if model_path:
                LOGGER.info("✓ 使用本地模型: %s", model_path)
                self.progress_tracker.update(progress=100.0, message="找到本地模型")
            else:
                # === 阶段 2: 下载模型 ===
                LOGGER.info("本地未找到，开始下载...")
                model_path = self.downloader.download(model_id, source=source)
                
                # 保存元数据
                self.config_preserver.save_metadata(
                    model_path, model_id, source
                )

            # === 阶段 3: 验证模型 ===
            if not skip_validation:
                is_valid, errors = self.validator.validate(model_path)
                if not is_valid:
                    raise ValueError(f"模型验证失败: {errors}")

            # === 阶段 4: 选择量化格式 ===
            if quantization is None:
                # 获取可用 VRAM
                available_vram = self._get_available_vram()
                model_params = self._estimate_model_params(model_path)

                if interactive:
                    quantization = self.quant_selector.select_interactive(
                        model_params, available_vram
                    )
                else:
                    quantization = self.quant_selector.select_auto(
                        model_params, available_vram
                    )

            # === 阶段 5: 可选的异步/同步加载 ===
            if mode == LoadMode.CHECK_ONLY:
                LOGGER.info("✓ 模型准备完成（仅检查模式）")
            elif mode == LoadMode.ASYNC_LOAD:
                LOGGER.info("✓ 模型准备完成，后台加载中...")
                # 异步加载在后台进行
                threading.Thread(
                    target=self._async_load,
                    args=(model_path,),
                    daemon=True
                ).start()
            else:
                # CHECK_AND_LOAD: 同步加载（由调用方的引擎完成）
                LOGGER.info("✓ 模型准备完成，可以开始加载")

            elapsed = time.time() - start_time
            LOGGER.info("模型准备总耗时: %.1f 秒", elapsed)

            self.progress_tracker.finish(success=True)
            return model_path, quantization

        except Exception as e:
            LOGGER.error("模型准备失败: %s", e)
            self.progress_tracker.finish(success=False)
            raise

    def _get_available_vram(self) -> float:
        """获取可用显存（GB）"""
        try:
            import torch
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                return free / (1024**3)
        except Exception:
            pass
        return 24.0  # 默认假设 24GB

    def _estimate_model_params(self, model_path: Path) -> float:
        """估算模型参数量（十亿）"""
        config_path = model_path / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                
                # 尝试从配置推断参数量
                hidden_size = config.get("hidden_size", 4096)
                num_layers = config.get("num_hidden_layers", 32)
                vocab_size = config.get("vocab_size", 100000)
                
                # 粗略估算: embeddings + attention + ffn
                params = vocab_size * hidden_size  # embeddings
                params += num_layers * (4 * hidden_size * hidden_size)  # attention
                params += num_layers * (8 * hidden_size * hidden_size)  # ffn
                
                return params / 1e9
            except Exception:
                pass

        # 根据文件大小估算（FP16 情况下）
        total_size = self.detector.get_model_size(model_path)
        return total_size / (2 * 1e9)  # 2 bytes per param

    def _default_progress_callback(self, info: ProgressInfo) -> None:
        """默认进度回调（打印进度条）"""
        bar_width = 40
        filled = int(bar_width * info.progress / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        status = f"[{info.stage.value}] [{bar}] {info.progress:.1f}%"
        if info.speed:
            status += f" | {info.speed}"
        if info.eta:
            status += f" | ETA: {info.eta}"
        
        # 使用 \r 覆盖当前行
        print(f"\r{status}", end="", flush=True)
        
        if info.progress >= 100.0 or info.stage in [ModelStage.READY, ModelStage.FAILED]:
            print()  # 换行

    def _on_stall(self, info: ProgressInfo) -> None:
        """停滞回调：抛出超时错误并停止"""
        LOGGER.error(f"❌ 模型操作停滞于 [{info.stage.value}] 阶段，已超过限制时间，强制停止任务。")
        raise TimeoutError(f"模型加载超时 ({self.stall_threshold}s)。")

    def _async_load(self, model_path: Path) -> None:
        """异步加载模型（后台线程）"""
        try:
            self.sharded_loader.load_sharded_weights(model_path)
            LOGGER.info("✓ 异步加载完成")
        except Exception as e:
            LOGGER.error("异步加载失败: %s", e)


# ============================================================================
# 便捷函数
# ============================================================================

def prepare_model(
    model_id: str,
    engine_type: str = "vllm",
    interactive: bool = False,
    **kwargs
) -> Tuple[Path, Optional[QuantizationType]]:
    """
    便捷函数：准备模型
    
    Args:
        model_id: 模型 ID
        engine_type: 引擎类型
        interactive: 是否交互式选择量化
        **kwargs: 传递给 ModelManager.prepare_model
        
    Returns:
        (模型路径, 量化类型)
    """
    manager = ModelManager(engine_type=engine_type)
    return manager.prepare_model(model_id, interactive=interactive, **kwargs)
