#!/bin/bash
# 自部署模块 API 测试脚本

set -e

BASE_URL="http://localhost:8080"

echo "=========================================="
echo "  自部署模块 API 集成测试"
echo "=========================================="
echo ""

# 测试 1: 获取状态
echo "[测试 1] 获取部署状态"
echo "命令: curl -X GET $BASE_URL/api/deploy/status"
curl -X GET "$BASE_URL/api/deploy/status" | jq . || echo "Gateway 未运行或未启用自部署"
echo ""

# 测试 2: 启动 Worker
echo "[测试 2] 启动 Worker"
echo "命令: curl -X POST $BASE_URL/api/deploy/start"
RESPONSE=$(curl -X POST "$BASE_URL/api/deploy/start" 2>/dev/null | jq . || echo '{"success":false}')
echo "$RESPONSE"
echo ""

# 测试 3: 再次获取状态
echo "[测试 3] 获取更新后的部署状态"
echo "命令: curl -X GET $BASE_URL/api/deploy/status"
sleep 2
curl -X GET "$BASE_URL/api/deploy/status" | jq . || echo "Gateway 未运行"
echo ""

# 测试 4: 重启 Worker
echo "[测试 4] 重启 Worker"
echo "命令: curl -X POST $BASE_URL/api/deploy/restart"
curl -X POST "$BASE_URL/api/deploy/restart" | jq . || echo "重启失败"
echo ""

# 测试 5: 停止 Worker
echo "[测试 5] 停止 Worker"
echo "命令: curl -X POST $BASE_URL/api/deploy/stop"
curl -X POST "$BASE_URL/api/deploy/stop" | jq . || echo "停止失败"
echo ""

# 测试 6: 重新安装依赖
echo "[测试 6] 重新安装依赖"
echo "命令: curl -X POST $BASE_URL/api/deploy/reinstall"
curl -X POST "$BASE_URL/api/deploy/reinstall" | jq . || echo "重新安装失败"
echo ""

echo "=========================================="
echo "  测试完成"
echo "=========================================="
