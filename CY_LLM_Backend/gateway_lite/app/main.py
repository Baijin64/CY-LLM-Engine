import json
import os
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import grpc
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from worker.proto_gen import ai_service_pb2, ai_service_pb2_grpc


app = FastAPI(title="Gateway Lite", version="0.1.0")


class GatewayLiteConfig(BaseModel):
    api_token: str = ""
    coordinator_grpc_addr: str = "unix:///tmp/cy_coordinator.sock"
    request_timeout_sec: int = 60


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    repetition_penalty: Optional[float] = 1.0


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config.json"


@lru_cache(maxsize=1)
def load_config() -> GatewayLiteConfig:
    config_path = os.getenv("GATEWAY_LITE_CONFIG", "").strip()
    path = Path(config_path) if config_path else _default_config_path()
    if not path.exists():
        return GatewayLiteConfig()

    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return GatewayLiteConfig(**payload)
    except Exception as exc:
        print(f"[GatewayLite] Failed to load config from {path}: {exc}")
        return GatewayLiteConfig()


def get_api_token() -> str:
    env_token = os.getenv("GATEWAY_API_TOKEN", "").strip()
    if env_token:
        return env_token
    return load_config().api_token.strip()


def get_coordinator_addr() -> str:
    env_addr = os.getenv("COORDINATOR_UDS_PATH", "").strip()
    if env_addr:
        return f"unix://{env_addr}" if not env_addr.startswith("unix://") else env_addr
    cfg_addr = load_config().coordinator_grpc_addr.strip()
    return cfg_addr or "unix:///tmp/cy_coordinator.sock"


def get_request_timeout() -> Optional[float]:
    env_timeout = os.getenv("GATEWAY_REQUEST_TIMEOUT", "").strip()
    if env_timeout:
        try:
            value = float(env_timeout)
            return value if value > 0 else None
        except ValueError:
            return None
    cfg_timeout = load_config().request_timeout_sec
    return float(cfg_timeout) if cfg_timeout and cfg_timeout > 0 else None


def build_prompt(messages: List[ChatMessage]) -> str:
    """
    构建对话提示词。
    对于简单模型（如 OPT），使用清晰的对话格式。
    """
    # 检查是否只有一条用户消息（最简单的情况）
    if len(messages) == 1 and messages[0].role == "user":
        # 直接返回用户内容，让模型自由回复
        return messages[0].content
    
    # 多轮对话：构建对话历史
    lines = []
    for msg in messages:
        if msg.role == "system":
            lines.append(f"System: {msg.content}")
        elif msg.role == "user":
            lines.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            lines.append(f"Assistant: {msg.content}")
    
    # 添加 Assistant 前缀，引导模型生成回复
    lines.append("Assistant:")
    return "\n".join(lines)


def require_token(authorization: Optional[str]) -> None:
    token = get_api_token()
    if not token:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    incoming = authorization.replace("Bearer ", "", 1).strip()
    if incoming != token:
        raise HTTPException(status_code=401, detail="Unauthorized")


async def stream_predict(prompt: str, model_id: str, params: ChatCompletionRequest) -> dict:
    target = get_coordinator_addr()
    trace_id = str(uuid.uuid4())
    base_timeout = get_request_timeout()
    # 模型加载可能需要较长时间，使用更长的超时时间
    # 加载阶段不计入正常超时，使用一个很长的超时时间（30分钟）
    # 这样可以确保模型加载不会因为超时而失败
    loading_timeout = 1800.0  # 30分钟，足够模型加载
    timeout = loading_timeout if base_timeout is None else max(base_timeout, loading_timeout)

    async def request_iter():
        yield ai_service_pb2.StreamPredictRequest(
            model_id=model_id,
            prompt=prompt,
            generation=ai_service_pb2.GenerationParameters(
                max_new_tokens=params.max_tokens or 256,
                temperature=params.temperature or 0.7,
                top_p=params.top_p or 0.9,
                repetition_penalty=params.repetition_penalty or 1.0,
            ),
            metadata=ai_service_pb2.StreamMetadata(trace_id=trace_id),
        )

    try:
        async with grpc.aio.insecure_channel(target) as channel:
            stub = ai_service_pb2_grpc.AiInferenceStub(channel)
            response_text = []
            ttft_ms = 0.0
            tps = 0.0
            
            async for resp in stub.StreamPredict(request_iter(), timeout=timeout):
                chunk = resp.chunk
                
                # 过滤掉模型加载进度信息（以"[模型加载]"开头的消息）
                # 这些信息只在服务端日志中显示，不返回给客户端
                if "[模型加载]" in chunk:
                    continue  # 跳过加载进度信息
                
                response_text.append(chunk)
                if resp.end_of_stream:
                    ttft_ms = resp.ttft_ms
                    tps = resp.tokens_per_sec
                    break
            
            # 只返回实际的模型响应，不包含加载进度信息
            return {
                "text": "".join(response_text),
                "ttft_ms": ttft_ms,
                "tps": tps,
                "trace_id": trace_id
            }
    except grpc.aio.AioRpcError as e:
        # 如果是超时错误，提供更详细的错误信息
        if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
            detail = f"Request timeout: {e.details()}. Model loading may take several minutes for the first request."
            raise HTTPException(status_code=504, detail=detail)
        detail = f"Coordinator unavailable or error: {e.details()}"
        raise HTTPException(status_code=503, detail=detail)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(payload: ChatCompletionRequest, authorization: Optional[str] = Header(default=None)):
    require_token(authorization)
    if payload.stream:
        raise HTTPException(status_code=400, detail="Streaming not supported in lite gateway")

    prompt = build_prompt(payload.messages)
    model_id = payload.model or "default"
    result = await stream_predict(prompt, model_id, payload)
    completion = result["text"]

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": completion},
                "finish_reason": "stop",
            }
        ],
        "performance": {
            "ttft_ms": result["ttft_ms"],
            "tps": result["tps"],
            "trace_id": result["trace_id"]
        }
    }
