# 自部署模块测试指南

## 测试环境要求

- Java 21+
- Conda 已安装和配置
- Worker 目录在 Gateway 相对位置或绝对路径可访问
- Python 3.10+ 环境

## 测试方式一：本地开发环境（推荐）

### 1. 启用自部署模块

编辑 `application.yml` 或设置环境变量：

```bash
export CY_LLM_SELF_DEPLOY_ENABLED=true
export CY_LLM_SELF_DEPLOY_ENABLED=true
export CY_LLM_WORKER_DIR=../worker
export CY_LLM_WORKER_DIR=../worker
export CY_LLM_CONDA_PATH=conda
export CY_LLM_CONDA_PATH=conda
export CY_LLM_CONDA_ENV=cy_llm_worker
export CY_LLM_PYTHON_VERSION=3.10
export CY_LLM_PYTHON_VERSION=3.10
export CY_LLM_WORKER_PORT=50051
export CY_LLM_WORKER_PORT=50051
export CY_LLM_AUTO_START=true
export CY_LLM_AUTO_START=true
```

### 2. 启动 Gateway

```bash
cd gateway
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
./gradlew bootRun
```

或者使用 JAR：

```bash
java -jar build/libs/gateway-0.0.1-SNAPSHOT.jar \
  --self-deploy.enabled=true \
  --self-deploy.workerDir=../worker
```

### 3. 观察日志

启动时应该看到：

```
========================================
  开始自动部署 AI Worker
========================================
[1/3] 配置 Python 环境...
检测到 Conda: conda (版本 23.x.x)
创建 Conda 环境: cy_llm_worker (Python 3.10)
...
[2/3] 启动 Worker 进程...
启动 Worker: id=default, port=50051
Worker 已启动，PID=xxxxx
...
[3/3] 等待 Worker 就绪...
Worker 健康检查通过 (xxxx ms)
========================================
  自动部署完成!
========================================
```

## 测试方式二：Docker 环境

### 1. 使用 docker-compose

```bash
cd deploy
docker-compose up -d
```

注意：设置 `CY_LLM_SELF_DEPLOY_ENABLED=false`，让 docker-compose 管理服务。

### 2. 检查服务状态

```bash
curl -X GET http://localhost:8080/api/deploy/status
```

预期响应：

```json
{
  "enabled": false,
  "initialized": false,
  "workerPort": 50051,
  "condaEnvName": "cy_llm_worker",
  "workerDir": "../worker"
}
```

## 测试方式三：API 测试

### 获取部署状态

```bash
curl -X GET http://localhost:8080/api/deploy/status
```

### 手动启动 Worker（如果自动启动禁用）

```bash
curl -X POST http://localhost:8080/api/deploy/start
```

响应示例：

```json
{
  "success": true,
  "message": "部署成功",
  "steps": [
    {
      "name": "environment",
      "success": true,
      "message": "环境设置成功"
    },
    {
      "name": "start_worker",
      "success": true,
      "message": "Worker 进程已启动 (PID=12345)"
    },
    {
      "name": "health_check",
      "success": true,
      "message": "Worker 健康检查通过 (1234ms)"
    }
  ],
  "durationMs": 5000
}
```

### 重启 Worker

```bash
curl -X POST http://localhost:8080/api/deploy/restart
```

### 停止 Worker

```bash
curl -X POST http://localhost:8080/api/deploy/stop
```

### 重新安装依赖

```bash
curl -X POST http://localhost:8080/api/deploy/reinstall
```

## 预期的自部署流程

### 阶段 1：环境配置（30秒内）

1. ✅ 检测 Conda 安装
2. ✅ 检查 Conda 环境是否存在
3. ✅ 创建 Python 环境（如果不存在）
4. ✅ 安装 requirements.txt 依赖

### 阶段 2：启动 Worker（5秒内）

1. ✅ 构建启动命令
2. ✅ 启动 Worker 进程
3. ✅ 重定向日志到 Gateway 日志系统
4. ✅ 等待进程启动

### 阶段 3：健康检查（10秒内）

1. ✅ 尝试连接 Worker gRPC 端口
2. ✅ 验证连接成功

## 故障排查

### 问题 1：找不到 Conda

**症状：** `未检测到 Conda 安装`

**解决方案：**

```bash
# 检查 Conda 是否已安装
which conda
# 或设置完整路径
export CY_LLM_CONDA_PATH=/opt/conda/bin/conda
export CY_LLM_CONDA_PATH=/opt/conda/bin/conda
```

### 问题 2：环境创建失败

**症状：** `Conda 环境创建失败`

**解决方案：**

```bash
# 手动创建环境测试
conda create -n cy_llm_worker python=3.10 -y
# 检查是否成功
conda env list | grep cy_llm_worker
```

### 问题 3：Worker 进程启动失败

**症状：** `Worker 启动失败`

**解决方案：**

```bash
# 检查 worker 目录中的 requirements.txt
ls -la ../worker/requirements.txt
# 手动尝试启动
cd ../worker
conda run -n cy_llm_worker python -m worker.main --port 50051
```

### 问题 4：健康检查超时

**症状：** `健康检查超时`

**解决方案：**

```bash
# 检查 Worker 进程是否在运行
ps aux | grep "python.*worker"
# 检查日志中的错误信息
tail -f gateway.log | grep Worker
```

## 验证完整工作流

```bash
#!/bin/bash

echo "测试 1: 获取状态"
curl http://localhost:8080/api/deploy/status | jq .

echo "测试 2: 重启 Worker"
curl -X POST http://localhost:8080/api/deploy/restart | jq .

echo "测试 3: 获取状态（应该显示已初始化）"
curl http://localhost:8080/api/deploy/status | jq .

echo "测试 4: 停止 Worker"
curl -X POST http://localhost:8080/api/deploy/stop | jq .

echo "测试 5: 再次启动"
curl -X POST http://localhost:8080/api/deploy/start | jq .
```

## 性能指标

- **环境配置耗时**：取决于 pip install（通常 2-5 分钟）
- **Worker 启动耗时**：通常 1-3 秒
- **健康检查耗时**：通常 100-500ms
- **总部署时间**：首次 3-10 分钟，后续 5-10 秒

## 生产环境建议

1. **禁用自部署模块**（`enabled: false`）
2. **使用 Docker Compose** 来管理服务生命周期
3. **使用 Kubernetes** 来管理多个 Worker 副本
4. **配置合适的资源限制** （CPU、内存、GPU）
5. **设置监控和告警** 来跟踪 Worker 状态