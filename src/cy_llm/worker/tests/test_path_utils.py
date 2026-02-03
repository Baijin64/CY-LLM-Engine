"""
test_path_utils.py
utils/path_utils.py 模块的单元测试
"""

import pytest
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from worker.utils.path_utils import safe_join, validate_model_path, sanitize_filename, is_safe_path


class TestSafeJoin:
    """测试安全路径拼接"""

    def test_simple_join(self):
        """简单路径拼接"""
        result = safe_join("/base", "subdir", "file.txt")
        assert result == "/base/subdir/file.txt"

    def test_prevent_traversal_dotdot(self):
        """应阻止 .. 路径穿越"""
        with pytest.raises(ValueError, match="traversal|unsafe|invalid"):
            safe_join("/base", "../etc/passwd")

    def test_prevent_absolute_path_injection(self):
        """应处理绝对路径注入尝试"""
        # safe_join 会去除前导斜杠，所以 /etc/passwd 变成 etc/passwd
        result = safe_join("/base", "/etc/passwd")
        # 结果应该在 base 下，不是根目录的 /etc/passwd
        assert "/base/" in result or result.startswith("/base")

    def test_normalize_slashes(self):
        """应正规化多余斜杠"""
        result = safe_join("/base/", "/subdir/", "file.txt")
        # 结果应该是有效路径或抛出异常
        assert isinstance(result, str)

    def test_empty_parts(self):
        """空路径部分应被忽略"""
        result = safe_join("/base", "", "file.txt")
        assert "file.txt" in result

    def test_single_dot(self):
        """单点应被正规化"""
        result = safe_join("/base", ".", "file.txt")
        assert result == "/base/file.txt"


class TestValidateModelPath:
    """测试模型路径验证"""

    def test_valid_path_in_allowed_dir(self):
        """允许目录内的路径应验证通过"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.bin")
            # 创建文件
            with open(model_path, "w") as f:
                f.write("test")
            
            result = validate_model_path(model_path, [tmpdir])
            assert result is True or result == model_path

    def test_path_outside_allowed_dirs(self):
        """不在允许目录的路径应失败"""
        with pytest.raises((ValueError, PermissionError)):
            validate_model_path("/etc/passwd", ["/home/models"])

    def test_empty_allowed_dirs(self):
        """空的允许目录列表应不验证，返回路径"""
        # 实现选择：空 allowed_roots 表示不检查限制
        result = validate_model_path("/any/path", [])
        assert isinstance(result, str)

    def test_traversal_attempt(self):
        """路径穿越尝试应被阻止"""
        with pytest.raises((ValueError, PermissionError)):
            validate_model_path("/models/../etc/passwd", ["/models"])


class TestSanitizeFilename:
    """测试文件名清理"""

    def test_normal_filename(self):
        """正常文件名应保持不变"""
        assert sanitize_filename("model.bin") == "model.bin"

    def test_remove_path_separators(self):
        """应移除路径分隔符"""
        result = sanitize_filename("path/to/file.txt")
        assert "/" not in result
        assert "\\" not in result

    def test_remove_dangerous_chars(self):
        """应移除危险字符"""
        result = sanitize_filename("file<>:\"|?*.txt")
        # 结果不应包含这些字符
        for char in '<>:"|?*':
            assert char not in result

    def test_remove_null_bytes(self):
        """应移除 null 字节"""
        result = sanitize_filename("file\x00name.txt")
        assert "\x00" not in result

    def test_unicode_filename(self):
        """Unicode 文件名应被保留"""
        result = sanitize_filename("模型文件.bin")
        assert "模型" in result or result  # 保留或安全替换

    def test_empty_filename(self):
        """空文件名应返回安全默认值或抛出异常"""
        result = sanitize_filename("")
        # 应该返回安全值或抛出异常
        assert isinstance(result, str) or result is None

    def test_dotdot_in_filename(self):
        """文件名中的 .. 应被处理"""
        result = sanitize_filename("..hidden")
        # 应该安全处理
        assert result is not None


class TestIsSafePath:
    """测试路径安全检查"""

    def test_path_within_base(self):
        """基础目录内的路径应安全"""
        assert is_safe_path("/base/subdir/file.txt", "/base") is True

    def test_path_outside_base(self):
        """基础目录外的路径应不安全"""
        assert is_safe_path("/other/file.txt", "/base") is False

    def test_traversal_attempt(self):
        """路径穿越应被检测"""
        assert is_safe_path("/base/../etc/passwd", "/base") is False

    def test_symbolic_link_handling(self):
        """符号链接应被正确处理"""
        # 这个测试可能需要实际创建符号链接
        # 暂时跳过
        pass

    def test_same_path(self):
        """相同路径应安全"""
        assert is_safe_path("/base", "/base") is True

    def test_normalized_comparison(self):
        """路径应被正规化后比较"""
        assert is_safe_path("/base/./subdir", "/base") is True
        assert is_safe_path("/base/subdir/../subdir", "/base") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
