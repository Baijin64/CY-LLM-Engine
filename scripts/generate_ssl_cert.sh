#!/usr/bin/env bash
# =============================================================================
# 生成 SSL 证书脚本
# 用法: ./generate_ssl_cert.sh [domain] [output_dir]
# =============================================================================

set -e

DOMAIN="${1:-localhost}"
OUTPUT_DIR="${2:-./certs}"
KEYSTORE_PASSWORD="${SSL_KEYSTORE_PASSWORD:-changeit}"

# 颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  SSL 证书生成工具${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Domain: $DOMAIN"
echo "Output: $OUTPUT_DIR"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 方式 1: 自签名证书 (开发/测试环境)
generate_self_signed() {
    echo -e "${YELLOW}生成自签名证书...${NC}"
    
    # 生成私钥和证书
    openssl req -x509 -newkey rsa:4096 \
        -keyout "$OUTPUT_DIR/privkey.pem" \
        -out "$OUTPUT_DIR/fullchain.pem" \
        -sha256 -days 365 -nodes \
        -subj "/CN=$DOMAIN" \
        -addext "subjectAltName=DNS:$DOMAIN,DNS:localhost,IP:127.0.0.1"
    
    # 转换为 PKCS12 格式 (Java/Spring Boot 使用)
    openssl pkcs12 -export \
        -in "$OUTPUT_DIR/fullchain.pem" \
        -inkey "$OUTPUT_DIR/privkey.pem" \
        -out "$OUTPUT_DIR/keystore.p12" \
        -name gateway \
        -passout "pass:$KEYSTORE_PASSWORD"
    
    echo -e "${GREEN}✓ 自签名证书已生成${NC}"
    echo ""
    echo "文件列表:"
    echo "  - $OUTPUT_DIR/privkey.pem    (私钥)"
    echo "  - $OUTPUT_DIR/fullchain.pem  (证书)"
    echo "  - $OUTPUT_DIR/keystore.p12   (PKCS12 密钥库)"
    echo ""
    echo "Spring Boot 配置:"
    echo "  server.ssl.key-store: file:$OUTPUT_DIR/keystore.p12"
    echo "  server.ssl.key-store-password: $KEYSTORE_PASSWORD"
    echo "  server.ssl.key-store-type: PKCS12"
}

# 方式 2: Let's Encrypt 证书 (生产环境)
generate_letsencrypt() {
    echo -e "${YELLOW}使用 Let's Encrypt 生成证书...${NC}"
    
    if ! command -v certbot &> /dev/null; then
        echo "certbot 未安装，安装中..."
        sudo apt update && sudo apt install -y certbot
    fi
    
    # 生成证书 (需要域名指向服务器)
    sudo certbot certonly --standalone \
        -d "$DOMAIN" \
        --non-interactive \
        --agree-tos \
        --email "admin@$DOMAIN"
    
    # 复制证书
    sudo cp "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" "$OUTPUT_DIR/"
    sudo cp "/etc/letsencrypt/live/$DOMAIN/privkey.pem" "$OUTPUT_DIR/"
    sudo chown "$USER:$USER" "$OUTPUT_DIR"/*.pem
    
    # 转换为 PKCS12
    openssl pkcs12 -export \
        -in "$OUTPUT_DIR/fullchain.pem" \
        -inkey "$OUTPUT_DIR/privkey.pem" \
        -out "$OUTPUT_DIR/keystore.p12" \
        -name gateway \
        -passout "pass:$KEYSTORE_PASSWORD"
    
    echo -e "${GREEN}✓ Let's Encrypt 证书已生成${NC}"
    echo ""
    echo "注意: Let's Encrypt 证书每 90 天需要续期"
    echo "自动续期: sudo certbot renew --quiet"
}

# 根据域名选择方式
if [[ "$DOMAIN" == "localhost" || "$DOMAIN" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    generate_self_signed
else
    echo "检测到公网域名: $DOMAIN"
    echo ""
    echo "请选择证书类型:"
    echo "  1) 自签名证书 (开发/测试)"
    echo "  2) Let's Encrypt (生产环境)"
    read -p "选择 [1/2]: " choice
    
    case $choice in
        2) generate_letsencrypt ;;
        *) generate_self_signed ;;
    esac
fi

echo ""
echo -e "${GREEN}完成！${NC}"
