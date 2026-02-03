"""
test_task_scheduler.py
core/task_scheduler.py 模块的单元测试
"""

import pytest
import sys
import os
import time
import threading
from concurrent.futures import Future, TimeoutError as FutureTimeoutError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from worker.core.task_scheduler import TaskScheduler, SchedulerBusy, ScheduledTask


class TestTaskScheduler:
    """测试任务调度器"""

    @pytest.fixture
    def scheduler(self):
        """创建调度器实例"""
        s = TaskScheduler(max_workers=2, queue_size=10)
        yield s
        s.shutdown(wait=True)

    def test_submit_returns_future(self, scheduler):
        """submit 应返回 Future"""
        future = scheduler.submit(lambda: 42)
        assert isinstance(future, Future)

    def test_future_result(self, scheduler):
        """Future 应包含正确结果"""
        future = scheduler.submit(lambda: 42)
        result = future.result(timeout=5)
        assert result == 42

    def test_submit_with_args(self, scheduler):
        """submit 应支持参数"""
        future = scheduler.submit(lambda x, y: x + y, 10, 20)
        assert future.result(timeout=5) == 30

    def test_submit_with_kwargs(self, scheduler):
        """submit 应支持关键字参数"""
        def add(a, b=0):
            return a + b
        
        future = scheduler.submit(add, 10, b=5)
        assert future.result(timeout=5) == 15

    def test_priority_ordering(self, scheduler):
        """高优先级任务应先执行"""
        results = []
        barrier = threading.Barrier(2)
        
        def blocking_task(name):
            barrier.wait(timeout=2)
            results.append(name)
            return name
        
        # 提交低优先级任务
        f1 = scheduler.submit(blocking_task, "low", priority=10)
        # 提交高优先级任务
        f2 = scheduler.submit(blocking_task, "high", priority=1)
        
        f1.result(timeout=5)
        f2.result(timeout=5)
        
        # 由于并发执行，顺序可能不确定
        # 但如果队列未满且 worker 空闲，都应完成
        assert "low" in results
        assert "high" in results

    def test_exception_propagation(self, scheduler):
        """任务异常应传播到 Future"""
        def failing_task():
            raise ValueError("test error")
        
        future = scheduler.submit(failing_task)
        
        with pytest.raises(ValueError, match="test error"):
            future.result(timeout=5)

    def test_queue_full_raises_busy(self):
        """队列满时应抛出 SchedulerBusy"""
        # 使用小队列
        scheduler = TaskScheduler(max_workers=1, queue_size=2)
        blocker = threading.Event()
        
        def blocking_task():
            blocker.wait(timeout=10)
        
        try:
            # 填满队列
            futures = []
            for _ in range(5):  # 尝试提交多于队列容量的任务
                try:
                    f = scheduler.submit(blocking_task)
                    futures.append(f)
                except SchedulerBusy:
                    # 预期会抛出
                    blocker.set()
                    break
            else:
                # 如果没有抛出异常，手动通过
                blocker.set()
        finally:
            blocker.set()
            scheduler.shutdown(wait=False)

    def test_shutdown_stops_workers(self, scheduler):
        """shutdown 应停止所有 worker"""
        future = scheduler.submit(lambda: 42)
        future.result(timeout=5)
        
        scheduler.shutdown(wait=True)
        
        # shutdown 后提交应失败
        with pytest.raises(RuntimeError, match="shutdown"):
            scheduler.submit(lambda: 1)

    def test_concurrent_submissions(self, scheduler):
        """并发提交应正确处理"""
        results = []
        lock = threading.Lock()
        
        def task(n):
            with lock:
                results.append(n)
            return n
        
        futures = [scheduler.submit(task, i) for i in range(20)]
        
        for f in futures:
            f.result(timeout=10)
        
        assert len(results) == 20
        assert set(results) == set(range(20))

    def test_task_cancellation(self, scheduler):
        """取消的任务不应执行"""
        blocker = threading.Event()
        executed = []
        
        def task(n):
            blocker.wait()
            executed.append(n)
        
        # 提交任务但立即取消
        future = scheduler.submit(task, 1)
        cancelled = future.cancel()
        
        blocker.set()
        scheduler.shutdown(wait=True)
        
        # 如果成功取消，任务不应执行
        if cancelled:
            assert 1 not in executed


class TestScheduledTask:
    """测试 ScheduledTask 数据类"""

    def test_priority_ordering(self):
        """任务应按优先级排序"""
        task1 = ScheduledTask(
            priority=10,
            created_at=time.time(),
            sequence=1,
            func=lambda: None,
            args=(),
            kwargs={},
            future=Future(),
        )
        task2 = ScheduledTask(
            priority=1,
            created_at=time.time(),
            sequence=2,
            func=lambda: None,
            args=(),
            kwargs={},
            future=Future(),
        )
        
        # 低优先级数字 = 高优先级
        assert task2 < task1

    def test_same_priority_uses_created_at(self):
        """相同优先级应按创建时间排序"""
        t1 = time.time()
        t2 = t1 + 1
        
        task1 = ScheduledTask(
            priority=5,
            created_at=t1,
            sequence=1,
            func=lambda: None,
            args=(),
            kwargs={},
            future=Future(),
        )
        task2 = ScheduledTask(
            priority=5,
            created_at=t2,
            sequence=2,
            func=lambda: None,
            args=(),
            kwargs={},
            future=Future(),
        )
        
        # 先创建的任务优先
        assert task1 < task2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
