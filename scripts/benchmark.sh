#!/usr/bin/env bash
# =============================================================================
# 压力测试脚本 - 使用 k6
# 安装: https://k6.io/docs/getting-started/installation/
# 用法: ./benchmark.sh [endpoint] [duration] [vus]
# =============================================================================

set -e

# 配置
ENDPOINT="${1:-http://localhost:8000}"
DURATION="${2:-30s}"
VUS="${3:-10}"

# 颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  CY-LLM Lite 压力测试${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""
echo "Endpoint: $ENDPOINT"
echo "Duration: $DURATION"
echo "VUs:      $VUS"
echo ""

# 检查 k6 是否安装
if ! command -v k6 &> /dev/null; then
    echo -e "${YELLOW}k6 未安装，尝试安装...${NC}"
    if command -v apt &> /dev/null; then
        sudo gpg -k
        sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
        echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
        sudo apt update && sudo apt install k6 -y
    else
        echo "请手动安装 k6: https://k6.io/docs/getting-started/installation/"
        exit 1
    fi
fi

# 创建临时 k6 脚本
K6_SCRIPT=$(mktemp /tmp/k6_test_XXXXXX.js)

cat > "$K6_SCRIPT" << 'EOF'
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// 自定义指标
const errorRate = new Rate('errors');
const latency = new Trend('latency');

// 从环境变量读取配置
const ENDPOINT = __ENV.ENDPOINT || 'http://localhost:8000';

export const options = {
    vus: parseInt(__ENV.VUS) || 10,
    duration: __ENV.DURATION || '30s',
    thresholds: {
        http_req_duration: ['p(95)<5000'],  // 95% 请求 < 5秒
        errors: ['rate<0.1'],                // 错误率 < 10%
    },
};

export default function () {
    // 测试 1: 健康检查
    const healthRes = http.get(`${ENDPOINT}/health`);
    check(healthRes, {
        'health check status is 200': (r) => r.status === 200,
    });

    // 测试 2: OpenAI 兼容推理接口
    const inferencePayload = JSON.stringify({
        model: 'default',
        messages: [
            { role: 'user', content: '你好，请简短回复。' }
        ],
        temperature: 0.7,
        max_tokens: 50,
    });

    const inferenceRes = http.post(
        `${ENDPOINT}/v1/chat/completions`,
        inferencePayload,
        {
            headers: {
                'Content-Type': 'application/json',
            },
            timeout: '60s',
        }
    );

    const success = check(inferenceRes, {
        'inference status is 200': (r) => r.status === 200,
        'inference has choices': (r) => {
            try {
                const body = JSON.parse(r.body);
                return body.choices && body.choices.length > 0;
            } catch {
                return false;
            }
        },
    });

    errorRate.add(!success);
    latency.add(inferenceRes.timings.duration);

    sleep(1);  // 1秒间隔，避免过载
}

export function handleSummary(data) {
    return {
        'stdout': textSummary(data, { indent: ' ', enableColors: true }),
    };
}

function textSummary(data, options) {
    const indent = options.indent || '  ';
    let output = '\n';
    
    output += '█ SUMMARY\n';
    output += `${indent}Total Requests: ${data.metrics.http_reqs.values.count}\n`;
    output += `${indent}Failed Requests: ${data.metrics.http_req_failed.values.passes || 0}\n`;
    output += `${indent}Avg Duration: ${(data.metrics.http_req_duration.values.avg || 0).toFixed(2)}ms\n`;
    output += `${indent}P95 Duration: ${(data.metrics.http_req_duration.values['p(95)'] || 0).toFixed(2)}ms\n`;
    output += `${indent}P99 Duration: ${(data.metrics.http_req_duration.values['p(99)'] || 0).toFixed(2)}ms\n`;
    output += `${indent}Error Rate: ${((data.metrics.errors?.values?.rate || 0) * 100).toFixed(2)}%\n`;
    
    return output;
}
EOF

echo -e "${GREEN}开始测试...${NC}"
echo ""

# 运行 k6
k6 run \
    -e ENDPOINT="$ENDPOINT" \
    -e VUS="$VUS" \
    -e DURATION="$DURATION" \
    "$K6_SCRIPT"

# 清理
rm -f "$K6_SCRIPT"

echo ""
echo -e "${GREEN}测试完成！${NC}"
