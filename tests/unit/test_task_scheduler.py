"""
测试 TaskScheduler 的核心功能
"""

import pytest
import time
from concurrent.futures import Future
from cy_llm.worker.core.task_scheduler import TaskScheduler, SchedulerBusy, ScheduledTask


class TestTaskScheduler:
    """TaskScheduler 单元测试"""

    def test_submit_task_success(self):
        """测试成功提交任务"""
        scheduler = TaskScheduler(max_workers=2, queue_size=10)
        
        def sample_task(x: int, y: int) -> int:
            return x + y
        
        future = scheduler.submit(sample_task, 1, 2)
        assert isinstance(future, Future)
        result = future.result(timeout=1.0)
        assert result == 3
        
        scheduler.shutdown(wait=True)

    def test_queue_overflow_backpressure(self):
        """测试队列溢出时的背压机制"""
        scheduler = TaskScheduler(max_workers=1, queue_size=2)
        
        def slow_task():
            time.sleep(1.0)
            return True
        
        # 填满队列
        scheduler.submit(slow_task)
        scheduler.submit(slow_task)
        
        # 第三个任务应该触发背压
        with pytest.raises(SchedulerBusy):
            scheduler.submit(slow_task)
        
        scheduler.shutdown(wait=True)

    def test_priority_execution_order(self):
        """测试优先级调度（高优先级先执行）"""
        scheduler = TaskScheduler(max_workers=1, queue_size=10)
        results = []
        
        def record_task(task_id: str):
            results.append(task_id)
            time.sleep(0.01)
        
        # 提交不同优先级的任务（数字越小优先级越高）
        scheduler.submit(record_task, "low", priority=10)
        scheduler.submit(record_task, "high", priority=1)
        scheduler.submit(record_task, "medium", priority=5)
        
        time.sleep(0.5)  # 等待任务完成
        scheduler.shutdown(wait=True)
        
        # 高优先级任务应该先执行
        assert results[0] == "low"  # 第一个提交的先执行
        assert results[1] == "high"  # 然后是高优先级
        assert results[2] == "medium"

    def test_exception_handling(self):
        """测试任务异常被正确传播"""
        scheduler = TaskScheduler(max_workers=2)
        
        def failing_task():
            raise ValueError("Test exception")
        
        future = scheduler.submit(failing_task)
        
        with pytest.raises(ValueError, match="Test exception"):
            future.result(timeout=1.0)
        
        scheduler.shutdown(wait=True)

    def test_shutdown_cancels_pending_tasks(self):
        """测试关闭时未执行的任务被取消"""
        scheduler = TaskScheduler(max_workers=1, queue_size=10)
        
        def slow_task():
            time.sleep(2.0)
        
        # 提交多个任务
        scheduler.submit(slow_task)
        future = scheduler.submit(slow_task)
        
        # 立即关闭
        scheduler.shutdown(wait=False)
        
        # 未执行的任务应该包含异常
        with pytest.raises(RuntimeError, match="Scheduler stopped"):
            future.result(timeout=0.5)
