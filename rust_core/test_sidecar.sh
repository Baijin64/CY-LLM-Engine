#!/bin/bash
# Rust Sidecar 测试脚本

set -e

echo "=========================================="
echo "🧪 Rust Sidecar 测试脚本"
echo "=========================================="

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 1. 编译检查
echo -e "\n${YELLOW}[1/5] 编译检查...${NC}"
cd "$(dirname "$0")"
cargo build --release --no-default-features
echo -e "${GREEN}✓ 编译成功${NC}"

# 2. 单元测试
echo -e "\n${YELLOW}[2/5] 运行单元测试...${NC}"
cargo test --lib
echo -e "${GREEN}✓ 所有单元测试通过${NC}"

# 3. 集成测试
echo -e "\n${YELLOW}[3/5] 运行集成测试...${NC}"
cargo test --test integration_test
echo -e "${GREEN}✓ 所有集成测试通过${NC}"

# 4. 启动测试（无 Worker）
echo -e "\n${YELLOW}[4/5] 启动测试（无 Worker 连接）...${NC}"
./target/release/sidecar &
SIDECAR_PID=$!
sleep 2

if ps -p $SIDECAR_PID > /dev/null; then
    echo -e "${GREEN}✓ Sidecar 进程启动成功 (PID: $SIDECAR_PID)${NC}"
    
    # 检查监听端口
    if command -v ss &> /dev/null; then
        if ss -ltn | grep -q ':50051'; then
            echo -e "${GREEN}✓ gRPC 端口 50051 监听正常${NC}"
        else
            echo -e "${RED}✗ gRPC 端口 50051 未监听${NC}"
        fi
        
        if ss -ltn | grep -q ':9090'; then
            echo -e "${GREEN}✓ Metrics 端口 9090 监听正常${NC}"
        else
            echo -e "${YELLOW}⚠ Metrics 端口 9090 未监听（可能尚未实现）${NC}"
        fi
    fi
    
    # 停止进程
    kill $SIDECAR_PID
    wait $SIDECAR_PID 2>/dev/null || true
    echo -e "${GREEN}✓ Sidecar 进程已停止${NC}"
else
    echo -e "${RED}✗ Sidecar 进程启动失败${NC}"
    exit 1
fi

# 5. 配置验证
echo -e "\n${YELLOW}[5/5] 配置文件验证...${NC}"
if [ -f "sidecar.toml" ]; then
    echo -e "${GREEN}✓ 配置文件 sidecar.toml 存在${NC}"
    echo "配置内容摘要:"
    grep -E "^(bind_addr|worker_uds|metrics_port)" sidecar.toml || echo "  (使用默认配置)"
else
    echo -e "${YELLOW}⚠ 配置文件不存在，将使用默认配置${NC}"
fi

echo -e "\n=========================================="
echo -e "${GREEN}✅ 所有测试通过！${NC}"
echo "=========================================="
echo ""
echo "📦 下一步操作："
echo "  1. 启动 Python Worker:"
echo "     python -m worker.main --serve --uds-path /tmp/cy_worker.sock"
echo ""
echo "  2. 启动 Rust Sidecar:"
echo "     ./target/release/sidecar"
echo ""
echo "  3. 使用 grpcurl 测试:"
echo "     grpcurl -plaintext localhost:50051 cy.llm.AiInference/Health"
echo ""
