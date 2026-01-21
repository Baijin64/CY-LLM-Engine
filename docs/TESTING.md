# 测试指南

本文档描述 CY-LLM Engine 的测试策略、测试方法以及如何在本地和 CI 环境中运行测试。

## 目录

- [测试概述](#测试概述)
- [Python 测试 (Worker)](#python-testing-worker)
- [Kotlin 测试 (Gateway/Coordinator)](#kotlin-testing-gatewaycoordinator)
- [集成测试](#integration-testing)
- [CI/CD 配置](#cicd-configuration)
- [测试覆盖率](#test-coverage)
- [最佳实践](#best-practices)

## 测试概述

### 测试分层

```
┌─────────────────────────────────────────────────────────────────┐
│                     端到端测试 (E2E)                             │
│              CY_LLM_Backend/tests/test_integration.py           │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    集成测试 (Integration)                        │
│     CY_LLM_Backend/worker/tests/integration/                    │
│     CY_LLM_Backend/gateway/src/test/kotlin/                     │
│     CY_LLM_Backend/coordinator/src/test/kotlin/                 │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     单元测试 (Unit)                              │
│           CY_LLM_Backend/worker/tests/test_*.py                 │
└─────────────────────────────────────────────────────────────────┘
```

### 测试矩阵

| 组件 | 测试类型 | 测试框架 | 位置 |
|------|----------|----------|------|
| Worker | 单元测试 | pytest | `worker/tests/` |
| Worker | 集成测试 | pytest | `worker/tests/integration/` |
| Gateway | 单元/集成测试 | JUnit 5 | `gateway/src/test/` |
| Coordinator | 单元/集成测试 | JUnit 5 | `coordinator/src/test/` |

---

## Python 测试 (Worker)

### 环境准备

```bash
# 进入 Worker 目录
cd CY_LLM_Backend/worker

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 安装测试依赖
pip install pytest pytest-asyncio pytest-cov pytest-mock httpx
```

### 运行测试

#### 所有测试

```bash
# 运行所有测试
pytest tests/ -v

# 快速运行 (安静模式)
pytest tests/ -q

# 带覆盖率
pytest tests/ --cov=worker --cov-report=html --cov-report=term-missing
```

#### 单个测试文件

```bash
# 运行单个测试文件
pytest tests/test_grpc_servicer.py -v

# 运行单个测试类
pytest tests/test_grpc_servicer.py::TestGrpcServicer -v

# 运行单个测试方法
pytest tests/test_grpc_servicer.py::TestGrpcServicer::test_stream_predict -v
```

#### 按标记运行

```bash
# 运行所有单元测试
pytest tests/ -m unit -v

# 运行所有集成测试
pytest tests/ -m integration -v

# 跳过 slow 测试
pytest tests/ -m "not slow" -v
```

### 测试标记 (Markers)

```python
import pytest

@pytest.mark.unit
def test_simple_function():
    """单元测试 - 不依赖外部服务"""
    pass

@pytest.mark.integration
def test_with_redis():
    """集成测试 - 需要 Redis"""
    pass

@pytest.mark.slow
def test_large_model():
    """慢速测试 - 需要 GPU"""
    pass

@pytest.mark.asyncio
async def test_async_function():
    """异步测试"""
    pass
```

### 测试示例

```python
"""测试示例 - test_prompt_cache.py"""

import pytest
from worker.cache.prompt_cache import PromptCache


class TestPromptCache:
    """PromptCache 单元测试"""

    def setup_method(self):
        """每个测试前初始化"""
        self.cache = PromptCache(max_size=100, ttl_seconds=3600)

    def test_set_and_get(self):
        """测试设置和获取"""
        # Given
        key = "test_key"
        value = "test_value"

        # When
        self.cache.set(key, value)
        result = self.cache.get(key)

        # Then
        assert result == value

    def test_get_nonexistent(self):
        """测试获取不存在的键"""
        result = self.cache.get("nonexistent")
        assert result is None

    def test_max_size(self):
        """测试最大容量"""
        # Fill cache to max size
        for i in range(100):
            self.cache.set(f"key_{i}", f"value_{i}")

        # Should evict oldest when over capacity
        self.cache.set("key_101", "value_101")

        # Oldest should be evicted
        assert self.cache.get("key_0") is None
        assert self.cache.get("key_101") == "value_101"

    def test_ttl_expiration(self):
        """测试 TTL 过期"""
        import time

        # Create cache with 1 second TTL
        cache = PromptCache(max_size=10, ttl_seconds=1)
        cache.set("key", "value")

        # Should exist
        assert cache.get("key") == "value"

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired
        assert cache.get("key") is None

    @pytest.mark.asyncio
    async def test_async_set(self):
        """测试异步设置"""
        await self.cache.async_set("async_key", "async_value")
        result = await self.cache.async_get("async_key")
        assert result == "async_value"
```

### Mock 使用

```python
"""使用 pytest-mock 进行 Mock"""

import pytest
from unittest.mock import MagicMock, patch

class TestWithMocks:
    @patch('worker.engines.engine_factory.create_engine')
    def test_engine_creation(self, mock_create_engine):
        """测试引擎创建 (Mock)"""
        # Given
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        # When
        from worker.engines.engine_factory import create_engine
        engine = create_engine("cuda-vllm")

        # Then
        mock_create_engine.assert_called_once_with("cuda-vllm")
        assert engine is mock_engine

    def test_with_service(self):
        """测试依赖外部服务"""
        with patch('redis.Redis') as mock_redis:
            mock_client = MagicMock()
            mock_redis.return_value = mock_client

            # Test that uses Redis
            from worker.cache.prompt_cache import PromptCache
            cache = PromptCache(redis_client=mock_client)
            # ...
```

### 异步测试

```python
"""异步测试示例"""

import pytest
from worker.core.server import InferenceServer


class TestInferenceServer:
    @pytest.mark.asyncio
    async def test_stream_predict(self):
        """测试流式推理"""
        # Given
        server = InferenceServer()
        # ... setup

        # When
        chunks = []
        async for chunk in server.stream_predict("test-prompt"):
            chunks.append(chunk)

        # Then
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """测试并发请求"""
        import asyncio

        # Given
        server = InferenceServer()

        # When - 并发多个请求
        tasks = [
            server.stream_predict(f"prompt-{i}")
            for i in range(10)
        ]
        results = await asyncio.gather(*tasks)

        # Then
        assert len(results) == 10
```

---

## Kotlin 测试 (Gateway/Coordinator)

### 运行测试

```bash
# Gateway 测试
cd CY_LLM_Backend/gateway
./gradlew test

# Coordinator 测试
cd CY_LLM_Backend/coordinator
./gradlew test

# 查看测试报告
open build/reports/tests/test/index.html
```

### 测试示例

```kotlin
// InferenceControllerTest.kt
@SpringBootTest
@AutoConfigureMockMvc
class InferenceControllerTest {

    @Autowired
    private lateinit var mockMvc: MockMvc

    @Test
    fun `test inference endpoint`() {
        // Given
        val request = InferenceRequest(
            modelId = "qwen2.5-7b",
            prompt = "Hello"
        )

        // When & Then
        mockMvc.perform(post("/api/v1/inference")
            .contentType(MediaType.APPLICATION_JSON)
            .content(ObjectMapper().writeValueAsString(request)))
            .andExpect(status().isOk)
            .andExpect(jsonPath("$.choices[0].text").exists())
    }

    @Test
    fun `test invalid model returns 404`() {
        // Given
        val request = InferenceRequest(
            modelId = "nonexistent-model",
            prompt = "Hello"
        )

        // When & Then
        mockMvc.perform(post("/api/v1/inference")
            .contentType(MediaType.APPLICATION_JSON)
            .content(ObjectMapper().writeValueAsString(request)))
            .andExpect(status().isNotFound)
    }
}
```

---

## 集成测试

### 环境准备

集成测试需要运行完整的服务栈：

```bash
# 启动所有服务
cd CY_LLM_Backend/deploy
docker compose up -d

# 等待服务就绪
./cy-llm status
```

### 运行集成测试

```bash
# Python 集成测试
cd CY_LLM_Backend
pytest tests/test_integration.py -v

# 或使用 CLI
./cy-llm test integration
```

### 集成测试示例

```python
"""test_integration.py"""

import pytest
import requests


class TestInferenceIntegration:
    """推理集成测试"""

    BASE_URL = "http://localhost:8080/api/v1"

    @pytest.fixture(scope="class")
    def setup_services(self):
        """确保服务运行"""
        # 这里可以检查服务状态
        yield
        # 清理

    def test_health_check(self, setup_services):
        """测试健康检查"""
        response = requests.get(f"{self.BASE_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "UP"

    def test_inference_request(self, setup_services):
        """测试推理请求"""
        request = {
            "modelId": "qwen2.5-7b",
            "prompt": "你好",
            "generation": {
                "maxNewTokens": 100,
                "temperature": 0.7
            }
        }

        response = requests.post(
            f"{self.BASE_URL}/inference",
            json=request
        )

        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0

    def test_streaming_inference(self, setup_services):
        """测试流式推理"""
        request = {
            "modelId": "qwen2.5-7b",
            "prompt": "讲一个短故事"
        }

        with requests.post(
            f"{self.BASE_URL}/inference/stream",
            json=request,
            stream=True
        ) as response:
            chunks = []
            for line in response.iter_lines():
                if line:
                    chunks.append(line)

            assert len(chunks) > 0
            # 验证 SSE 格式
            assert any(b"data:" in chunk for chunk in chunks)
```

---

## CI/CD 配置

### GitHub Actions 工作流

```yaml
# .github/workflows/tests.yml
name: Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  python-tests:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          
      - name: Install dependencies
        run: |
          cd CY_LLM_Backend/worker
          python -m venv .venv
          source .venv/bin/activate
          pip install -r requirements.txt
          pip install pytest pytest-cov
          
      - name: Run unit tests
        run: |
          source .venv/bin/activate
          pytest tests/ -m unit -v --cov=worker
          
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml

  kotlin-tests:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up JDK
        uses: actions/setup-java@v4
        with:
          java-version: '21'
          distribution: 'temurin'
          
      - name: Gateway tests
        run: |
          cd CY_LLM_Backend/gateway
          ./gradlew test
          
      - name: Coordinator tests
        run: |
          cd CY_LLM_Backend/coordinator
          ./gradlew test

  integration-tests:
    runs-on: ubuntu-latest
    needs: [python-tests, kotlin-tests]
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Start services
        run: |
          cd CY_LLM_Backend/deploy
          docker compose up -d
          sleep 30
          
      - name: Run integration tests
        run: |
          cd CY_LLM_Backend
          pip install pytest requests
          pytest tests/test_integration.py -v
          
      - name: Cleanup
        run: |
          cd CY_LLM_Backend/deploy
          docker compose down
```

---

## 测试覆盖率

### 生成覆盖率报告

```bash
# Python 覆盖率
cd CY_LLM_Backend/worker
pytest tests/ --cov=worker --cov-report=html --cov-report=term-missing

# 查看 HTML 报告
open htmlcov/index.html
```

### 覆盖率要求

| 组件 | 最低覆盖率 |
|------|------------|
| 核心逻辑 | 90% |
| 公共 API | 100% |
| 错误处理 | 80% |

### 覆盖率报告示例

```
Name                              Stmts   Miss  Cover
-----------------------------------------------------
worker/core/server.py               150    10    93%
worker/core/task_scheduler.py       100     5    95%
worker/engines/                     300    20    93%
worker/cache/prompt_cache.py         80     2    97%
worker/config/                      60     5    92%
-----------------------------------------------------
TOTAL                               690    42    94%
```

---

## 最佳实践

### 1. 测试命名

```python
# Good
def test_user_cannot_login_with_wrong_password():
    pass

def test_model_load_returns_correct_signature():
    pass

# Bad
def test_login():
    pass

def test_model():
    pass
```

### 2. 测试结构 (Given-When-Then)

```python
def test_feature():
    # Given - Setup
    input_data = create_test_data()
    expected = {...}

    # When - Action
    result = function(input_data)

    # Then - Assert
    assert result == expected
```

### 3. 避免测试耦合

```python
# Bad - Tests depend on execution order
def test_1_first():
    ...

def test_2_second():
    ...

# Good - Independent tests
def test_feature_one():
    ...

def test_feature_two():
    ...
```

### 4. 使用 Fixtures

```python
@pytest.fixture
def sample_model():
    """创建测试用模型"""
    return Model(id="test", name="Test Model")

@pytest.fixture(scope="session")
def app_config():
    """会话级配置"""
    return {"debug": True}
```

### 5. Mock 外部依赖

```python
# Mock API calls
@patch('requests.post')
def test_api_call(mock_post):
    mock_post.return_value.json.return_value = {"result": "success"}
    # ... test
```

### 6. 测试边界条件

```python
def test_empty_input():
    assert handle_input("") == DefaultResponse

def test_max_length_input():
    max_input = "a" * 10000
    assert handle_input(max_input) == TruncatedResponse
```

### 7. 定期运行完整测试

```bash
# 在开发过程中
pytest tests/ -v

# 在提交前
pytest tests/ -v --cov=worker

# 在 PR 前
pytest tests/ -v --cov=worker && ./gradlew test
```
