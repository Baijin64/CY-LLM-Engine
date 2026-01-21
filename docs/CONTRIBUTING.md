# 贡献者规范

感谢您对本项目的兴趣！我们欢迎各种形式的贡献，包括但不限于：

- 报告 Bug
- 提出新功能建议
- 提交代码改进
- 完善文档

在开始贡献之前，请阅读以下规范。

## 目录

- [行为准则](#行为准则)
- [开始贡献](#开始贡献)
- [开发环境搭建](#开发环境搭建)
- [代码风格](#代码风格)
- [提交规范](#提交规范)
- [Pull Request 流程](#pull-request-流程)
- [测试要求](#测试要求)
- [版本号规范](#版本号规范)

## 行为准则

请尊重所有参与者，遵守以下准则：

1. **友善交流** - 使用友好、包容的语言
2. **接受不同意见** - 建设性讨论，避免人身攻击
3. **专注于项目** - 讨论与项目相关的内容
4. **协助他人** - 帮助新贡献者融入社区

## 开始贡献

### 1. Fork 仓库

```bash
# 在 GitHub 上点击 Fork 按钮

# 克隆你的 Fork
git clone https://github.com/YOUR_USERNAME/CY-LLM-Engine.git
cd CY-LLM-Engine

# 添加上游仓库
git remote add upstream https://github.com/Baijin64/CY-LLM-Engine.git
```

### 2. 创建分支

```bash
# 同步上游最新代码
git fetch upstream
git checkout main
git merge upstream/main

# 创建功能分支
git checkout -b feature/your-feature-name
# 或
git checkout -b fix/your-bug-fix
```

### 3. 开发与提交

```bash
# 进行代码修改
# ...

# 提交修改
git add .
git commit -m "[1.0.0.0-Alpha] feat(scope): description"
```

### 4. 推送并创建 PR

```bash
# 推送分支到你的 Fork
git push origin feature/your-feature-name

# 在 GitHub 上创建 Pull Request
```

## 开发环境搭建

### Python 环境 (Worker)

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
pip install -r requirements-nvidia.txt  # 如需 GPU 支持

# 安装开发依赖
pip install pytest pytest-asyncio pytest-cov
pip install ruff black mypy types-xxx
```

### Kotlin 环境 (Gateway/Coordinator)

```bash
# 安装 JDK 21
# Ubuntu/Debian
sudo apt-get install openjdk-21-jdk

# macOS
brew install openjdk@21

# 验证
java -version
```

### 运行测试

```bash
# Python 测试
cd CY_LLM_Backend/worker
pytest tests/ -v

# Kotlin 测试
cd CY_LLM_Backend/gateway
./gradlew test

cd CY_LLM_Backend/coordinator
./gradlew test
```

## 代码风格

### Python (Worker)

我们使用 `ruff`、`black` 和 `mypy` 来确保代码质量：

```bash
# 代码格式化
black .

# 代码检查
ruff check .

# 类型检查
mypy .
```

#### 风格规范

1. **PEP 8** - 遵循 Python 编码规范
2. **类型注解** - 所有公开函数必须有类型注解
3. **Docstrings** - 使用 Google 风格 docstrings
4. **导入排序** - 标准库 → 第三方库 → 本地模块

```python
"""模块文档字符串。

Description of the module.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from pydantic import BaseModel

from worker.config.config_loader import WorkerConfig


class MyClass(BaseModel):
    """类的文档字符串。

    Attributes:
        attr: 属性描述。
    """

    attr: str
    optional_attr: Optional[int] = None

    def method(self, param: str) -> bool:
        """方法的文档字符串。

        Args:
            param: 参数描述。

        Returns:
            返回值描述。

        Raises:
            ValueError: 异常描述。
        """
        if not param:
            raise ValueError("Parameter cannot be empty")
        return True
```

### Kotlin (Gateway/Coordinator)

使用 `ktlint` 和 `detekt`：

```bash
# 代码检查
./gradlew ktlintCheck
./gradlew detekt

# 自动修复
./gradlew ktlintFormat
```

#### 风格规范

1. **Kotlin 官方风格指南**
2. **命名规范** - 类名 PascalCase，变量/函数 camelCase
3. **可见性** - 默认 private，使用明确可见性修饰符
4. **空安全** - 使用 safe calls (?) 和 elvis operator (?)

```kotlin
/**
 * 类的文档注释。
 */
class MyService(
    private val config: Config,
    private val repository: Repository,
) {
    /**
     * 函数描述。
     *
     * @param param 参数描述
     * @return 返回值描述
     * @throws IllegalArgumentException 异常描述
     */
    fun doSomething(param: String): Result {
        require(param.isNotBlank()) { "Parameter cannot be blank" }
        return Result.Success
    }
}
```

## 提交规范

### 提交消息格式

本项目采用 **四段式版本号**，提交消息格式如下：

```
[version] <type>(scope): short description

English description of the change.

【中文】变更的中文描述
```

### 版本号格式

```
[major.minor.patch.build-SUFFIX]
```

| 段 | 含义 | 示例 |
|----|------|------|
| major | 重大不兼容变更 | 2 |
| minor | 向后兼容新功能 | 1 |
| patch | Bug 修复与优化 | 0 |
| build | 构建/测试次数 | 2 |
| SUFFIX | 稳定性后缀 | Alpha/Beta/RC |

### 后缀说明

| 后缀 | 含义 |
|------|------|
| `PreAlpha` | 功能不完整，设计早期 |
| `Alpha` | 大部分功能可用，开始测试 |
| `Beta` | 功能实现完整，广泛测试 |
| `RC` | 候选发布版本 |
| `Release` | 生产就绪版本 |

### 提交类型 (Type)

| 类型 | 描述 |
|------|------|
| `feat` | 新功能 |
| `fix` | Bug 修复 |
| `refactor` | 重构 (无功能变化) |
| `perf` | 性能优化 |
| `docs` | 文档更新 |
| `test` | 测试相关 |
| `chore` | 构建/工具/依赖 |
| `ci` | CI/CD 配置 |
| `style` | 代码格式 (不影响语义) |
| ` BREAKING` | 破坏性变更 (放在类型前) |

### 提交示例

```
[2.1.1.2-Alpha] feat(worker): add async inference support

Add async inference support for vLLM engine with improved throughput.

【中文】worker：添加异步推理支持

为 vLLM 引擎添加异步推理支持，提升吞吐量。
```

```
[1.5.0.0-Alpha] BREAKING refactor(api): redesign inference API

Redesigned inference API to support streaming with metadata.

【中文】重构推理 API

重新设计推理 API 以支持带元数据的流式输出。
```

```
[1.4.2.1-Alpha] fix(memory): prevent OOM on large models

Fixed memory leak in model unloading and improved VRAM estimation.

【中文】fix(memory): 防止大模型 OOM

修复模型卸载时的内存泄漏问题，提升 VRAM 估算准确性。
```

## Pull Request 流程

### 创建 PR 前

1. **同步上游** - 确保你的分支与主分支同步
2. **运行测试** - 所有测试必须通过
3. **代码检查** - 通过 linting 和格式化
4. **更新文档** - 如有必要，更新相关文档

### PR 描述模板

```markdown
## 描述

简要描述本次 PR 的内容。

## 变更类型

- [ ] Bug 修复
- [ ] 新功能
- [ ] 破坏性变更
- [ ] 文档更新
- [ ] 重构

## 测试

- [ ] 单元测试通过
- [ ] 集成测试通过
- [ ] 手动测试完成

## 兼容性

- [ ] 向后兼容
- [ ] 破坏性变更（需说明）

## 截图/示例 (如适用)

添加截图或代码示例来说明变更。

## 相关 Issue

Fixes #123
```

### PR 检查清单

- [ ] 代码符合项目风格规范
- [ ] 所有测试通过
- [ ] 文档已更新（如需要）
- [ ] 提交消息符合规范
- [ ] 分支已同步到最新主分支
- [ ] 无不必要的文件变更

## 测试要求

### 单元测试

所有新功能必须包含单元测试：

```python
# Python 单元测试示例
def test_feature():
    """Test description."""
    # Given
    input_data = ...
    expected = ...

    # When
    result = function(input_data)

    # Then
    assert result == expected
```

### 测试覆盖要求

- **核心逻辑** - 100% 覆盖
- **公共 API** - 100% 覆盖
- **错误处理** - 关键路径必须有测试
- **集成测试** - 新增组件需要集成测试

### 运行测试

```bash
# 所有测试
pytest tests/ -v --cov=worker --cov-report=html

# 单个测试文件
pytest tests/test_grpc_servicer.py -v

# 集成测试
pytest tests/test_integration.py -v
```

## 版本号规范

### 语义化版本

| 版本号 | 含义 |
|--------|------|
| `1.0.0` | 首次发布 |
| `1.0.1` | Bug 修复 |
| `1.1.0` | 新功能 (向后兼容) |
| `2.0.0` | 破坏性变更 |

### 版本发布流程

1. 更新 `CHANGELOG.md`
2. 更新版本号 (如需要)
3. 创建 Release Tag
4. 生成 Release Note

### Release Tag 格式

```
v{major}.{minor}.{patch}-{suffix}
```

示例：`v2.1.0-Alpha`

## 常用命令速查

```bash
# 代码检查
ruff check .
black .
mypy .

# 运行测试
pytest tests/ -v
./gradlew test

# 清理
git clean -fdx
```

## 获取帮助

- **Issue** - 在 GitHub Issues 中提问
- **Discussions** - 使用 GitHub Discussions 讨论
- **文档** - 查看 [docs/](../docs/) 目录

感谢您的贡献！
