"""
constants.py
统一管理 Worker 模块中的常量和默认值

使用方式：
    from worker.constants import GRPCDefaults, TrainingDefaults, CacheDefaults
"""

from typing import Final


class GRPCDefaults:
    """gRPC 服务相关常量"""
    # 默认端口
    PORT: Final[int] = 50051
    
    # 消息大小限制 (100MB)
    MAX_MESSAGE_SIZE_BYTES: Final[int] = 100 * 1024 * 1024
    
    # 兼容旧名称
    MAX_MESSAGE_SIZE: Final[int] = MAX_MESSAGE_SIZE_BYTES
    
    # Keepalive 配置 (毫秒)
    KEEPALIVE_TIME_MS: Final[int] = 30_000
    KEEPALIVE_TIMEOUT_MS: Final[int] = 10_000
    MIN_PING_INTERVAL_MS: Final[int] = 10_000
    
    # 并发工作线程数
    DEFAULT_WORKERS: Final[int] = 10
    MAX_WORKERS: Final[int] = DEFAULT_WORKERS
    
    # 兼容旧名称
    TIMEOUT_SECONDS: Final[int] = 30


class TrainingDefaults:
    """训练相关常量"""
    # 序列长度
    MAX_SEQ_LENGTH: Final[int] = 2048
    
    # 检查点保存频率
    SAVE_STEPS: Final[int] = 100
    EVAL_STEPS: Final[int] = 100
    
    # 作业清理间隔 (秒)
    JOB_CLEANUP_INTERVAL_SECONDS: Final[int] = 3600
    
    # 列表查询默认限制
    LIST_JOBS_DEFAULT_LIMIT: Final[int] = 100
    
    # 兼容旧测试
    BATCH_SIZE: Final[int] = 4
    LEARNING_RATE: Final[float] = 2e-4
    EPOCHS: Final[int] = 3
    LORA_RANK: Final[int] = 64
    LORA_ALPHA: Final[int] = 16


class CacheDefaults:
    """缓存相关常量"""
    # 提示缓存默认大小
    MAX_SIZE: Final[int] = 1000
    
    # TTL (秒)
    TTL_SECONDS: Final[int] = 3600
    
    # 默认 max_tokens
    DEFAULT_MAX_TOKENS: Final[int] = 512
    
    # Redis 扫描批量大小
    REDIS_SCAN_COUNT: Final[int] = 1000
    
    # Redis 默认 URL
    REDIS_DEFAULT_URL: Final[str] = "redis://localhost:6379"
    
    # 兼容旧测试
    ENABLED: Final[bool] = True


class MemoryDefaults:
    """显存管理相关常量"""
    # GPU 显存使用阈值
    GPU_MEMORY_THRESHOLD: Final[float] = 0.90
    GPU_THRESHOLD: Final[float] = GPU_MEMORY_THRESHOLD
    
    # 自动检查间隔 (秒)
    CHECK_INTERVAL_SECONDS: Final[int] = 60
    CLEANUP_INTERVAL: Final[int] = CHECK_INTERVAL_SECONDS
    
    # 锁清理触发阈值
    LOCK_CLEANUP_THRESHOLD: Final[int] = 100
    LOCK_CLEANUP_INTERVAL: Final[int] = LOCK_CLEANUP_THRESHOLD


class QueueDefaults:
    """队列相关常量"""
    # 默认队列大小
    DEFAULT_SIZE: Final[int] = 128


class UnitConversions:
    """单位换算常量"""
    # 字节转换
    BYTES_PER_KB: Final[int] = 1024
    BYTES_PER_MB: Final[int] = 1024 * 1024
    BYTES_PER_GB: Final[int] = 1024 * 1024 * 1024
    
    # 兼容旧测试
    KB: Final[int] = BYTES_PER_KB
    MB: Final[int] = BYTES_PER_MB
    GB: Final[int] = BYTES_PER_GB
    
    # 毫秒转换
    MS_PER_SECOND: Final[int] = 1000
    NS_PER_MS: Final[int] = 1_000_000


# 便捷访问
__all__ = [
    "GRPCDefaults",
    "TrainingDefaults",
    "CacheDefaults",
    "MemoryDefaults",
    "QueueDefaults",
    "UnitConversions",
]
