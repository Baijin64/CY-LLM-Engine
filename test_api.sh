#!/bin/bash
# 测试 API 响应格式

echo "测试 1: 简单问候"
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "hello"}],
    "temperature": 0.7
  }' | jq -r '.choices[0].message.content'

echo -e "\n---\n"

echo "测试 2: 简单问题"
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "temperature": 0.7
  }' | jq -r '.choices[0].message.content'

echo -e "\n---\n"

echo "测试 3: 多轮对话"
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {"role": "user", "content": "My name is Alice"},
      {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
      {"role": "user", "content": "What is my name?"}
    ],
    "temperature": 0.7
  }' | jq -r '.choices[0].message.content'
