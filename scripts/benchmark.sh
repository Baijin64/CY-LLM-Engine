#!/usr/bin/env bash
# =============================================================================
# 压力测试脚本 - 使用 k6
# 安装: https://k6.io/docs/getting-started/installation/
# 用法: ./benchmark.sh [endpoint] [api_key]
# =============================================================================

set -e

# 配置
ENDPOINT="${1:-http://localhost:8080}"
API_KEY="${2:-your-test-api-key}"
DURATION="${3:-30s}"
VUS="${4:-50}"

# 颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  EW AI Gateway 压力测试${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""
echo "Endpoint: $ENDPOINT"
echo "API Key:  ${API_KEY:0:10}..."
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
const ENDPOINT = __ENV.ENDPOINT || 'http://localhost:8080';
const API_KEY = __ENV.API_KEY || '';

export const options = {
    vus: parseInt(__ENV.VUS) || 50,
    duration: __ENV.DURATION || '30s',
    thresholds: {
        http_req_duration: ['p(95)<2000'],  // 95% 请求 < 2秒
        errors: ['rate<0.1'],                // 错误率 < 10%
    },
};

export default function () {
    // 测试 1: 健康检查
    const healthRes = http.get(`${ENDPOINT}/actuator/health`);
    check(healthRes, {
        'health check status is 200': (r) => r.status === 200,
    });

    // 测试 2: 推理接口 (同步)
    const inferencePayload = JSON.stringify({
        modelId: 'default',
        prompt: '你好，请简短回复。',
        generation: {
            maxNewTokens: 50,
            temperature: 0.7,
        },
    });

    const inferenceRes = http.post(
        `${ENDPOINT}/api/v1/inference`,
        inferencePayload,
        {
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': API_KEY,
            },
            timeout: '30s',
        }
    );

    const success = check(inferenceRes, {
        'inference status is 200': (r) => r.status === 200,
        'inference has content': (r) => {
            try {
                const body = JSON.parse(r.body);
                return body.content && body.content.length > 0;
            } catch {
                return false;
            }
        },
    });

    errorRate.add(!success);
    latency.add(inferenceRes.timings.duration);

    // 检查速率限制响应头
    if (inferenceRes.headers['X-RateLimit-Remaining']) {
        const remaining = parseInt(inferenceRes.headers['X-RateLimit-Remaining']);
        if (remaining < 10) {
            console.log(`Warning: Rate limit remaining: ${remaining}`);
        }
    }

    // 如果被限流，等待
    if (inferenceRes.status === 429) {
        const retryAfter = parseInt(inferenceRes.headers['Retry-After']) || 60;
        console.log(`Rate limited, waiting ${retryAfter}s`);
        sleep(retryAfter);
    }

    sleep(0.1);  // 100ms 间隔
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
    -e API_KEY="$API_KEY" \
    -e VUS="$VUS" \
    -e DURATION="$DURATION" \
    "$K6_SCRIPT"

# 清理
rm -f "$K6_SCRIPT"

echo ""
echo -e "${GREEN}测试完成！${NC}"
