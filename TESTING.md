# TESTING.md

... (existing content preserved)
# Dev environment quick setup (added by automation scripts)

If Gradle builds fail with messages like "Directory does not contain a Gradle build", don't run `./gradlew` at the repository root—run the subproject wrapper.

Scripts have been added to convenience your workflow:

- `scripts/setup-jdk.sh`: detect and suggest the correct JDK/JAVA_HOME path. Run with:

```bash
# Print suggestions
./scripts/setup-jdk.sh

# Attempt to export JAVA_HOME for this shell session (temporary):
./scripts/setup-jdk.sh --auto-export
```

- `scripts/gradle-build.sh`: runs gradle wrapper builds inside `CY_LLM_Backend/coordinator` and `CY_LLM_Backend/gateway` respectively. It avoids errors from running gradlew from repo root.

```bash
# Make sure JAVA_HOME is set, either manually or with script
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
# Or auto-export with the setup script
source ./scripts/setup-jdk.sh --auto-export

# Run builds (skips tests by default with -x test)
./scripts/gradle-build.sh
```

When the script detects no `gradlew` in a subproject, it will skip it and print a warning, which helps catch incomplete worktrees.

If you want to force a top-level Gradle build across multi-module project, make sure a `settings.gradle(.kts)` and top-level `gradlew` exist; otherwise prefer the `gradle-build.sh` approach.

# 项目测试说明 (TESTING.md)

本文件用来描述如何在本地与 CI 中运行单元测试、集成测试以及代码覆盖率工具。

## Python (Worker)

项目使用 `pytest` 运行 Python 单元测试：

```bash
# 进入 worker 目录
cd CY_LLM_Backend/worker

# 安装依赖（虚拟环境/conda 推荐）
pip install -r requirements.txt

# 运行所有单元测试
pytest tests/ -v

# 运行单个测试文件
pytest tests/test_prompt_cache.py -q

# 测试并生成覆盖率报告
pytest --cov=worker --cov-report=html tests/ -q
open htmlcov/index.html
```

### 测试分层
- 单元测试：`tests/test_*.py`（快速，模拟依赖）
- 集成测试：`tests/integration` 或 `tests/test_integration.py`（需要运行 Gateway/Coordinator/Worker）

## Kotlin (Gateway/Coordinator)

Gateway 和 Coordinator 使用 Gradle 的测试套件：

```bash
cd CY_LLM_Backend/gateway
./gradlew test

cd CY_LLM_Backend/coordinator
./gradlew test
```

Gradle 会输出报告到 `build/reports/tests/test/index.html`。

## Docker / CI

CI 流程（建议）:
1. 运行 Python lints（ruff/flake8/black）
2. 运行 `pytest --maxfail=1 --disable-warnings -q`
3. 运行 `./gradlew test` 在 Kotlin 项目中
4. 如果有需要，运行集成测试（`./cy test integration` / `./cy-llm test integration`）

## 运行本地端到端测试

在 `CY_LLM_Backend/deploy/docker-compose.yml` 或 `./cy start` 启动所有服务后，可运行 `CY_LLM_Backend/tests/test_integration.py`。

```bash
cd CY_LLM_Backend
# 启动服务
./cy start --engine cuda-vllm
python -m pytest tests/test_integration.py -q
```

## 新增测试建议
1. 每新增模块必须附带对应的 `test_*.py` 或 `.kt` 单元测试
2. 关键路径（net/http handlers、gRPC servicers、memory manager、train/engine、prompt cache）必须有单元或集成测试
3. 所有破坏性变更必须在 PR 中包含回归测试或兼容性说明
