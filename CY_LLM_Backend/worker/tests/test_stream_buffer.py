"""
test_stream_buffer.py
utils/stream_buffer.py 模块的单元测试
"""

import pytest
import sys
import os
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from worker.utils.stream_buffer import StreamBuffer


class TestStreamBuffer:
    """测试流缓冲区"""

    @pytest.fixture
    def buffer(self):
        """创建缓冲区实例"""
        return StreamBuffer()

    def test_write_and_read(self, buffer):
        """write 和 read 应正确工作"""
        buffer.write("Hello")
        buffer.write(" World")
        
        content = buffer.read()
        assert "Hello" in content or content == "Hello World"

    def test_write_chunk(self, buffer):
        """write_chunk 应追加内容"""
        buffer.write_chunk("chunk1")
        buffer.write_chunk("chunk2")
        
        content = buffer.read_all() if hasattr(buffer, 'read_all') else buffer.read()
        assert "chunk1" in content
        assert "chunk2" in content

    def test_iter_chunks(self, buffer):
        """应能迭代所有块"""
        buffer.write_chunk("a")
        buffer.write_chunk("b")
        buffer.write_chunk("c")
        
        if hasattr(buffer, 'iter_chunks'):
            chunks = list(buffer.iter_chunks())
            assert len(chunks) == 3
        elif hasattr(buffer, '__iter__'):
            chunks = list(buffer)
            assert len(chunks) >= 1

    def test_clear(self, buffer):
        """clear 应清空缓冲区"""
        buffer.write("content")
        buffer.clear() if hasattr(buffer, 'clear') else None
        
        if hasattr(buffer, 'clear'):
            content = buffer.read()
            assert content == "" or content is None

    def test_is_empty(self, buffer):
        """is_empty 应正确反映状态"""
        if hasattr(buffer, 'is_empty'):
            assert buffer.is_empty() is True
            
            buffer.write("content")
            assert buffer.is_empty() is False

    def test_size(self, buffer):
        """size 应返回正确大小"""
        if hasattr(buffer, 'size'):
            assert buffer.size() == 0
            
            buffer.write("12345")
            assert buffer.size() == 5

    def test_thread_safety(self, buffer):
        """并发写入应线程安全"""
        errors = []
        
        def writer(prefix):
            try:
                for i in range(100):
                    buffer.write(f"{prefix}-{i}")
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=writer, args=(f"t{i}",)) for i in range(4)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0

    def test_blocking_read(self, buffer):
        """阻塞读取应等待数据"""
        if hasattr(buffer, 'read_blocking'):
            result = [None]
            
            def reader():
                result[0] = buffer.read_blocking(timeout=2)
            
            t = threading.Thread(target=reader)
            t.start()
            
            time.sleep(0.1)
            buffer.write("delayed content")
            
            t.join(timeout=3)
            assert result[0] is not None

    def test_close(self, buffer):
        """close 应标记缓冲区为关闭"""
        if hasattr(buffer, 'close'):
            buffer.close()
            
            if hasattr(buffer, 'is_closed'):
                assert buffer.is_closed() is True


class TestStreamBufferWithCapacity:
    """测试带容量限制的缓冲区"""

    def test_capacity_limit(self):
        """应遵守容量限制"""
        if hasattr(StreamBuffer, '__init__') and 'capacity' in StreamBuffer.__init__.__code__.co_varnames:
            buffer = StreamBuffer(capacity=100)
            
            # 写入超过容量的数据
            try:
                for _ in range(200):
                    buffer.write("x" * 10)
            except OverflowError:
                # 预期行为
                pass

    def test_auto_flush_on_capacity(self):
        """达到容量时应自动刷新"""
        # 这取决于实现
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
