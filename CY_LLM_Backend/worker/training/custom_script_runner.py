"""
custom_script_runner.py
[训练] 自定义训练脚本执行器
说明：允许用户运行自己的训练脚本，支持进度回调和资源管理。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

LOGGER = logging.getLogger("ew.worker.training.custom")


class ScriptStatus(Enum):
    """脚本执行状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class ScriptProgress:
    """脚本执行进度"""
    job_id: str
    status: ScriptStatus
    stdout_lines: List[str] = field(default_factory=list)
    stderr_lines: List[str] = field(default_factory=list)
    return_code: Optional[int] = None
    elapsed_seconds: float = 0.0
    # 解析出的训练指标（如果脚本输出 JSON 格式的进度）
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: str = ""


@dataclass
class ScriptJob:
    """脚本执行任务"""
    job_id: str
    script_path: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    working_dir: Optional[str] = None
    timeout_seconds: int = 86400  # 默认 24 小时超时
    status: ScriptStatus = ScriptStatus.PENDING
    process: Optional[subprocess.Popen] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    return_code: Optional[int] = None
    error: Optional[str] = None
    cancel_event: threading.Event = field(default_factory=threading.Event)


class CustomScriptRunner:
    """
    自定义脚本执行器
    
    功能：
    1. 执行任意 Python 脚本
    2. 实时捕获 stdout/stderr
    3. 解析 JSON 格式的进度输出
    4. 支持超时和取消
    5. 资源管理和清理
    
    使用示例：
    ```python
    runner = CustomScriptRunner()
    
    # 创建任务
    job = runner.create_job(
        script_path="/path/to/my_train.py",
        args=["--epochs", "10", "--lr", "1e-5"],
        env={"CUDA_VISIBLE_DEVICES": "0"},
        timeout_seconds=3600,
    )
    
    # 运行并获取进度
    for progress in runner.run_script(job):
        print(f"Status: {progress.status}, Lines: {len(progress.stdout_lines)}")
    ```
    
    脚本进度输出格式（可选）：
    在脚本中输出以下格式的 JSON 行，将被自动解析：
    ```
    {"ew_progress": {"epoch": 1, "step": 100, "loss": 0.5, "lr": 1e-5}}
    ```
    """

    # 允许执行的脚本扩展名
    ALLOWED_EXTENSIONS = {".py", ".sh", ".bash"}
    
    # 危险命令黑名单（安全检查）
    DANGEROUS_PATTERNS = [
        r"rm\s+-rf\s+/",
        r"dd\s+if=",
        r"mkfs\.",
        r":(){ :|:& };:",  # Fork bomb
        r">\s*/dev/sd",
        r"chmod\s+777\s+/",
    ]

    def __init__(
        self,
        max_concurrent: int = 2,
        allowed_paths: Optional[List[str]] = None,
        python_executable: Optional[str] = None,
    ):
        """
        初始化脚本执行器
        
        Args:
            max_concurrent: 最大并发执行数
            allowed_paths: 允许执行脚本的路径列表（安全限制）
            python_executable: Python 解释器路径
        """
        self._jobs: Dict[str, ScriptJob] = {}
        self._lock = threading.Lock()
        self._max_concurrent = max_concurrent
        self._running_count = 0
        self._allowed_paths = allowed_paths or []
        self._python_executable = python_executable or sys.executable

    def create_job(
        self,
        script_path: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        working_dir: Optional[str] = None,
        timeout_seconds: int = 86400,
        job_id: Optional[str] = None,
    ) -> ScriptJob:
        """创建脚本执行任务"""
        job_id = job_id or str(uuid.uuid4())
        
        # 验证脚本路径
        self._validate_script_path(script_path)
        
        job = ScriptJob(
            job_id=job_id,
            script_path=script_path,
            args=args or [],
            env=env or {},
            working_dir=working_dir,
            timeout_seconds=timeout_seconds,
        )
        
        with self._lock:
            self._jobs[job_id] = job
        
        return job

    def get_job(self, job_id: str) -> Optional[ScriptJob]:
        """获取任务信息"""
        return self._jobs.get(job_id)

    def list_jobs(
        self,
        status_filter: Optional[List[ScriptStatus]] = None,
        limit: int = 100,
    ) -> List[ScriptJob]:
        """列出任务"""
        jobs = list(self._jobs.values())
        if status_filter:
            jobs = [j for j in jobs if j.status in status_filter]
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def cancel_job(self, job_id: str) -> bool:
        """取消任务"""
        job = self._jobs.get(job_id)
        if job is None:
            return False
        
        if job.status in (ScriptStatus.COMPLETED, ScriptStatus.FAILED, 
                          ScriptStatus.CANCELLED, ScriptStatus.TIMEOUT):
            return False
        
        job.cancel_event.set()
        
        # 终止进程
        if job.process and job.process.poll() is None:
            try:
                # 先发送 SIGTERM
                job.process.terminate()
                time.sleep(1)
                # 如果还在运行，发送 SIGKILL
                if job.process.poll() is None:
                    job.process.kill()
            except Exception as e:
                LOGGER.warning("Failed to kill process: %s", e)
        
        job.status = ScriptStatus.CANCELLED
        job.finished_at = datetime.now()
        return True

    def run_script(self, job: ScriptJob) -> Generator[ScriptProgress, None, None]:
        """
        运行脚本并生成进度流
        
        Args:
            job: 脚本任务
            
        Yields:
            ScriptProgress: 执行进度
        """
        # 检查并发限制
        with self._lock:
            if self._running_count >= self._max_concurrent:
                job.status = ScriptStatus.FAILED
                job.error = f"Concurrent limit reached ({self._max_concurrent})"
                yield ScriptProgress(
                    job_id=job.job_id,
                    status=ScriptStatus.FAILED,
                    error=job.error,
                )
                return
            self._running_count += 1

        try:
            yield from self._execute_script(job)
        finally:
            with self._lock:
                self._running_count -= 1

    def _execute_script(self, job: ScriptJob) -> Generator[ScriptProgress, None, None]:
        """执行脚本的核心逻辑"""
        job.status = ScriptStatus.RUNNING
        job.started_at = datetime.now()
        start_time = time.time()

        # 构建命令
        script_path = Path(job.script_path)
        if script_path.suffix == ".py":
            cmd = [self._python_executable, str(script_path)] + job.args
        elif script_path.suffix in (".sh", ".bash"):
            cmd = ["bash", str(script_path)] + job.args
        else:
            cmd = [str(script_path)] + job.args

        # 构建环境变量
        env = os.environ.copy()
        env.update(job.env)
        # 设置 Python 不缓冲输出
        env["PYTHONUNBUFFERED"] = "1"

        LOGGER.info("[%s] Starting script: %s", job.job_id, " ".join(cmd))

        try:
            # 启动子进程
            job.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=job.working_dir,
                text=True,
                bufsize=1,  # 行缓冲
            )

            stdout_lines: List[str] = []
            stderr_lines: List[str] = []
            metrics: Dict[str, Any] = {}

            # 确保 stdout/stderr 不为 None
            assert job.process.stdout is not None
            assert job.process.stderr is not None

            # 使用 select 或线程读取 stdout/stderr
            import selectors
            sel = selectors.DefaultSelector()
            sel.register(job.process.stdout, selectors.EVENT_READ)
            sel.register(job.process.stderr, selectors.EVENT_READ)

            last_yield_time = time.time()
            yield_interval = 0.5  # 每 0.5 秒发送一次进度

            while True:
                # 检查取消
                if job.cancel_event.is_set():
                    job.process.terminate()
                    job.status = ScriptStatus.CANCELLED
                    break

                # 检查超时
                elapsed = time.time() - start_time
                if elapsed > job.timeout_seconds:
                    job.process.terminate()
                    job.status = ScriptStatus.TIMEOUT
                    job.error = f"Timeout after {job.timeout_seconds}s"
                    break

                # 读取输出
                ready = sel.select(timeout=0.1)
                for key, _ in ready:
                    fileobj = key.fileobj  # type: ignore
                    line = fileobj.readline()
                    if line:
                        if fileobj is job.process.stdout:
                            stdout_lines.append(line.rstrip())
                            # 尝试解析 JSON 进度
                            parsed_metrics = self._parse_progress_line(line)
                            if parsed_metrics:
                                metrics.update(parsed_metrics)
                        else:
                            stderr_lines.append(line.rstrip())

                # 检查进程是否结束
                if job.process.poll() is not None:
                    # 读取剩余输出
                    remaining_stdout = job.process.stdout.read()  # type: ignore
                    remaining_stderr = job.process.stderr.read()  # type: ignore
                    if remaining_stdout:
                        stdout_lines.extend(remaining_stdout.strip().split('\n'))
                    if remaining_stderr:
                        stderr_lines.extend(remaining_stderr.strip().split('\n'))
                    break

                # 定期发送进度
                if time.time() - last_yield_time >= yield_interval:
                    yield ScriptProgress(
                        job_id=job.job_id,
                        status=ScriptStatus.RUNNING,
                        stdout_lines=stdout_lines[-100:],  # 最后 100 行
                        stderr_lines=stderr_lines[-50:],   # 最后 50 行
                        elapsed_seconds=elapsed,
                        metrics=metrics.copy(),
                    )
                    last_yield_time = time.time()

            sel.close()

            # 处理最终状态
            job.return_code = job.process.returncode
            job.finished_at = datetime.now()

            if job.status == ScriptStatus.RUNNING:
                if job.return_code == 0:
                    job.status = ScriptStatus.COMPLETED
                else:
                    job.status = ScriptStatus.FAILED
                    job.error = f"Exit code: {job.return_code}"

            # 发送最终进度
            yield ScriptProgress(
                job_id=job.job_id,
                status=job.status,
                stdout_lines=stdout_lines,
                stderr_lines=stderr_lines,
                return_code=job.return_code,
                elapsed_seconds=time.time() - start_time,
                metrics=metrics,
                error=job.error or "",
            )

        except Exception as e:
            LOGGER.exception("[%s] Script execution failed: %s", job.job_id, e)
            job.status = ScriptStatus.FAILED
            job.error = str(e)
            job.finished_at = datetime.now()
            yield ScriptProgress(
                job_id=job.job_id,
                status=ScriptStatus.FAILED,
                error=str(e),
            )

    def _validate_script_path(self, script_path: str) -> None:
        """验证脚本路径的安全性"""
        path = Path(script_path).resolve()
        
        # 检查文件是否存在
        if not path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")
        
        # 检查扩展名
        if path.suffix.lower() not in self.ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Invalid script extension: {path.suffix}. "
                f"Allowed: {self.ALLOWED_EXTENSIONS}"
            )
        
        # 检查路径白名单（如果配置了）
        if self._allowed_paths:
            allowed = False
            for allowed_path in self._allowed_paths:
                if str(path).startswith(str(Path(allowed_path).resolve())):
                    allowed = True
                    break
            if not allowed:
                raise PermissionError(
                    f"Script path not in allowed directories: {script_path}"
                )
        
        # 读取脚本内容进行安全检查
        try:
            content = path.read_text()
            for pattern in self.DANGEROUS_PATTERNS:
                if re.search(pattern, content):
                    raise PermissionError(
                        f"Script contains potentially dangerous pattern: {pattern}"
                    )
        except UnicodeDecodeError:
            raise ValueError("Script file is not valid UTF-8 text")

    def _parse_progress_line(self, line: str) -> Optional[Dict[str, Any]]:
        """
        解析脚本输出中的 JSON 进度行
        
        支持格式：
        1. {"ew_progress": {"epoch": 1, "loss": 0.5}}
        2. EW_PROGRESS: {"epoch": 1, "loss": 0.5}
        """
        line = line.strip()
        
        try:
            # 格式 1: 纯 JSON
            if line.startswith("{") and "ew_progress" in line.lower():
                data = json.loads(line)
                if "ew_progress" in data:
                    return data["ew_progress"]
            
            # 格式 2: 前缀标记
            if line.upper().startswith("EW_PROGRESS:"):
                json_str = line[12:].strip()
                return json.loads(json_str)
                
        except (json.JSONDecodeError, KeyError):
            pass
        
        return None


# 全局实例
_custom_script_runner: Optional[CustomScriptRunner] = None


def get_custom_script_runner(
    allowed_paths: Optional[List[str]] = None,
) -> CustomScriptRunner:
    """获取全局自定义脚本执行器"""
    global _custom_script_runner
    if _custom_script_runner is None:
        # 默认允许 worker 目录和用户主目录下的脚本
        default_paths = [
            str(Path(__file__).parent.parent),  # worker 目录
            str(Path.home()),  # 用户主目录
        ]
        _custom_script_runner = CustomScriptRunner(
            allowed_paths=allowed_paths or default_paths,
        )
    return _custom_script_runner
