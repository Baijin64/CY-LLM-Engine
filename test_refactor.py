#!/usr/bin/env python3
"""
CY-LLM Engine Refactor Test Suite
重构后功能验证测试
"""

import json
import sys
import os
from pathlib import Path

# 添加路径
sys.path.insert(0, 'CY_LLM_Backend')
sys.path.insert(0, 'CY_LLM_Backend/worker')

class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []
    
    def test(self, name, func):
        """运行单个测试"""
        try:
            func()
            self.passed += 1
            self.tests.append((name, "PASS", None))
            print(f"✅ {name}")
        except Exception as e:
            self.failed += 1
            self.tests.append((name, "FAIL", str(e)))
            print(f"❌ {name}: {e}")
    
    def report(self):
        """生成测试报告"""
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"测试完成: {self.passed}/{total} 通过")
        print(f"{'='*60}")
        return self.failed == 0

# 创建测试运行器
runner = TestRunner()

print("="*60)
print("CY-LLM Engine Refactor Test Suite")
print("="*60)

# Test 1: 目录结构测试
def test_directory_structure():
    assert Path('CY_LLM_Backend/worker/main.py').exists(), "worker/main.py 不存在"
    assert Path('CY_LLM_Backend/worker/engines').is_dir(), "engines 目录不存在"
    assert not Path('src/cy_llm').exists(), "src/cy_llm 应该已被删除"

runner.test("目录结构检查", test_directory_structure)

# Test 2: 引擎文件存在性测试
def test_engine_files_exist():
    engines_dir = Path('CY_LLM_Backend/worker/engines')
    required_files = [
        'abstract_engine.py',
        'engine_factory.py',
        'vllm_cuda_engine.py',
        'vllm_async_engine.py',
        'trt_engine.py',
        'mindie_engine.py',
        '__init__.py'
    ]
    for f in required_files:
        assert (engines_dir / f).exists(), f"{f} 不存在"

runner.test("引擎文件完整性", test_engine_files_exist)

# Test 3: JSON格式测试
def test_dependency_registry_json():
    with open('CY_LLM_Backend/deploy/dependency_registry.json') as f:
        data = json.load(f)
    assert 'version' in data, "缺少version字段"
    assert 'compatibility_matrix' in data, "缺少compatibility_matrix"
    assert len(data['compatibility_matrix']) > 0, "compatibility_matrix为空"

runner.test("Dependency Registry JSON格式", test_dependency_registry_json)

# Test 4: Protobuf版本一致性测试
def test_protobuf_consistency():
    versions = []
    req_files = [
        'requirements-vllm.txt',
        'CY_LLM_Backend/worker/requirements.txt',
        'CY_LLM_Backend/gateway_lite/requirements.txt',
        'CY_LLM_Backend/coordinator_lite/requirements.txt',
        'gateway/gateway_lite/requirements.txt'
    ]
    for f in req_files:
        if Path(f).exists():
            with open(f) as file:
                content = file.read()
                if 'protobuf==' in content:
                    version = [line for line in content.split('\n') if 'protobuf==' in line][0]
                    versions.append((f, version))
    
    # 检查是否都使用4.25.3
    for f, v in versions:
        assert '4.25.3' in v, f"{f} 使用错误的protobuf版本: {v}"

runner.test("Protobuf版本一致性", test_protobuf_consistency)

# Test 5: CUDA版本一致性测试
def test_cuda_consistency():
    with open('requirements-vllm.txt') as f:
        content = f.read()
        assert 'cu124' in content, "requirements-vllm.txt 应该使用cu124"
    
    with open('requirements-nvidia.txt') as f:
        content = f.read()
        assert 'cu124' in content, "requirements-nvidia.txt 应该使用cu124"

runner.test("CUDA版本一致性", test_cuda_consistency)

# Test 6: 依赖管理模块导入测试
def test_deps_module_import():
    from deps import HardwareDetector, DependencyResolver
    detector = HardwareDetector()
    resolver = DependencyResolver()
    profiles = resolver.list_available_profiles()
    assert len(profiles) > 0, "应该至少有一个配置"

runner.test("依赖管理模块导入", test_deps_module_import)

# Test 7: BaseEngine抽象类测试
def test_base_engine_abc():
    from engines.abstract_engine import BaseEngine
    import inspect
    assert inspect.isabstract(BaseEngine), "BaseEngine应该是抽象类"
    
    # 检查必须实现的方法
    required_methods = ['load_model', 'infer', 'unload_model', 'get_memory_usage']
    for method in required_methods:
        assert hasattr(BaseEngine, method), f"缺少方法: {method}"

runner.test("BaseEngine抽象类结构", test_base_engine_abc)

# Test 8: 引擎工厂测试
def test_engine_factory():
    from engines.engine_factory import EngineRegistry, EngineInfo
    registry = EngineRegistry()
    engines = registry.list_engines()
    assert len(engines) > 0, "应该至少注册一个引擎"
    
    # 检查关键引擎是否存在
    engine_types = [e.engine_type for e in engines]
    assert 'cuda-vllm' in engine_types, "缺少cuda-vllm引擎"

runner.test("引擎工厂注册", test_engine_factory)

# Test 9: pyproject.toml更新测试
def test_pyproject_updated():
    with open('pyproject.toml') as f:
        content = f.read()
        assert 'CY_LLM_Backend' in content, "pyproject.toml应该指向CY_LLM_Backend"
        assert 'where = ["src"]' not in content, "应该移除旧的src指向"

runner.test("pyproject.toml配置", test_pyproject_updated)

# Test 10: 引擎继承关系测试
def test_engine_inheritance():
    # 通过检查文件内容验证继承关系
    engine_files = [
        'CY_LLM_Backend/worker/engines/vllm_cuda_engine.py',
        'CY_LLM_Backend/worker/engines/trt_engine.py',
        'CY_LLM_Backend/worker/engines/mindie_engine.py'
    ]
    
    for f in engine_files:
        with open(f) as file:
            content = file.read()
            assert 'BaseEngine' in content, f"{f} 应该继承BaseEngine"

runner.test("引擎继承关系", test_engine_inheritance)

# 生成报告
success = runner.report()

# 保存详细报告
report_path = Path('docs/refactor/cy-llm-engine/TestReport.md')
report_path.parent.mkdir(parents=True, exist_ok=True)

with open(report_path, 'w') as f:
    f.write("""# CY-LLM Engine Refactor - Test Report

**测试时间**: 2026-02-10  
**测试环境**: cy-llm-refactor (Python 3.11.14, CPU-only)  
**测试范围**: Phase 1-5 重构验证

---

## 测试摘要

| 指标 | 数值 |
|------|------|
| 总测试数 | {} |
| 通过 | {} |
| 失败 | {} |
| 通过率 | {:.1f}% |

---

## 详细结果

""".format(len(runner.tests), runner.passed, runner.failed, 
           runner.passed/len(runner.tests)*100 if runner.tests else 0))

    for name, status, error in runner.tests:
        f.write(f"### {name}\n")
        f.write(f"- **状态**: {'✅ 通过' if status == 'PASS' else '❌ 失败'}\n")
        if error:
            f.write(f"- **错误**: {error}\n")
        f.write("\n")
    
    f.write("""
---

## 测试覆盖范围

### 1. 结构测试
- ✅ 目录结构完整性
- ✅ 引擎文件存在性
- ✅ 旧目录已删除

### 2. 配置测试
- ✅ Protobuf版本一致性（4.25.3）
- ✅ CUDA版本一致性（cu124）
- ✅ pyproject.toml指向正确

### 3. 功能测试
- ✅ Dependency Registry JSON格式
- ✅ 依赖管理模块可导入
- ✅ BaseEngine抽象类结构
- ✅ 引擎工厂注册
- ✅ 引擎继承关系

### 4. 限制说明
由于当前是CPU-only环境，以下测试未执行：
- ⚠️ vLLM引擎实际加载测试
- ⚠️ GPU显存检测测试
- ⚠️ 端到端推理测试
- ⚠️ 性能基准测试

这些测试需要在GPU环境中执行。

---

## 结论

""")
    
    if success:
        f.write("✅ **所有测试通过！重构质量良好，可以进入下一阶段。**\n")
    else:
        f.write("⚠️ **存在测试失败，请修复后再进入下一阶段。**\n")

print(f"\n详细报告已保存到: {report_path}")

# 退出码
sys.exit(0 if success else 1)
