import os
import time
import uuid
from typing import List, Optional

import grpc
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from worker.proto_gen import ai_service_pb2, ai_service_pb2_grpc


app = FastAPI(title="Gateway Lite", version="0.1.0")


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


def build_prompt(messages: List[ChatMessage]) -> str:
    lines = []
    for msg in messages:
        lines.append(f"{msg.role}: {msg.content}")
    return "\n".join(lines)


def require_token(authorization: Optional[str]) -> None:
    token = os.getenv("GATEWAY_API_TOKEN", "").strip()
    if not token:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    incoming = authorization.replace("Bearer ", "", 1).strip()
    if incoming != token:
        raise HTTPException(status_code=401, detail="Unauthorized")


async def stream_predict(prompt: str, model_id: str, params: ChatCompletionRequest) -> str:
    target = os.getenv("COORDINATOR_GRPC_ADDR", "127.0.0.1:50051")
    trace_id = str(uuid.uuid4())

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
            async for resp in stub.StreamPredict(request_iter()):
                response_text.append(resp.chunk)
                if resp.end_of_stream:
                    break
            return "".join(response_text)
    except grpc.aio.AioRpcError as e:
        detail = f"Coordinator unavailable or error: {e.details()}"
        raise HTTPException(status_code=503, detail=detail)
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
    completion = await stream_predict(prompt, model_id, payload)

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
    }
