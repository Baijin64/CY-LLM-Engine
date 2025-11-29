# AI Inference Gateway

Element Warfare AI 后端的核心网关服务。负责接收来自客户端的 HTTP/REST 请求，将其转换为 gRPC 流式请求转发给后端的 AI Worker 节点，并将推理结果通过 Server-Sent Events (SSE) 实时流式返回给客户端。

## 🛠 技术栈 (Tech Stack)

本项目基于 **Kotlin** 和 **Spring Boot** 构建，采用响应式编程模型。

| 组件 | 版本 | 说明 |
| --- | --- | --- |
| **Java** | **21** | 基础运行环境 (LTS) |
| **Kotlin** | **1.9.24** | 主要开发语言 |
| **Spring Boot** | **3.3.2** | WebFlux 响应式框架 |
| **Gradle** | **8.14** | 构建工具 |
| **gRPC** | **1.65.1** | 高性能 RPC 框架 (Protobuf) |
| **Resilience4j** | **2.2.0** | 熔断器与重试机制 |
| **Project Reactor** | - | 响应式流处理 |
| **Coroutines** | - | Kotlin 协程支持 |

## 🚀 功能特性

- **协议转换**: 将前端的 RESTful 请求转换为内部的 gRPC 双向流/服务器流。
- **流式响应**: 支持 `text/event-stream` (SSE)，实现打字机效果的 AI 回复。
- **高可用性**: 集成 Resilience4j 实现服务熔断 (Circuit Breaker) 和自动重试 (Retry)。
- **负载均衡**: (规划中) 支持多 Worker 节点的负载分发。

## 📂 项目结构

```
src/main/kotlin/com/genshin/ai/
├── config/             # 配置类 (Resilience4j, gRPC Channel)
├── controller/         # WebFlux 控制器 (对外 REST 接口)
├── model/              # 数据模型 (Request/Response DTOs)
├── service/            # 业务逻辑 (InferenceService, WorkerStreamClient)
└── GatewayApplication.kt # 启动入口
```

## ⚡ 快速开始 (Quick Start)

### 1. 环境准备
确保本地已安装 **JDK 21**。
```bash
java -version
```

### 2. 构建项目
使用 Gradle Wrapper 进行构建（推荐使用国内镜像源配置）：
```bash
./gradlew clean build
```

### 3. 运行测试
本项目包含集成测试，会启动一个模拟的 gRPC Server 进行端到端验证。
```bash
./gradlew test
```

### 4. 启动服务
```bash
./gradlew bootRun
```
服务默认运行在 `8080` 端口。

## 🔌 接口说明

### 流式推理接口
- **URL**: `/api/v1/inference/stream`
- **Method**: `POST`
- **Content-Type**: `application/json`
- **Accept**: `text/event-stream`

**请求体示例**:
```json
{
  "modelId": "deepseek-v3",
  "prompt": "你好，请介绍一下你自己。",
  "parameters": {
    "temperature": 0.7,
    "maxTokens": 1024
  }
}
```

## ⚙️ 配置说明

主要配置文件位于 `src/main/resources/application.yml` (如未创建则使用默认配置)。

关键配置项：
- `server.port`: 服务端口
- `grpc.client.host`: AI Worker 地址
- `resilience4j.circuitbreaker`: 熔断策略配置
