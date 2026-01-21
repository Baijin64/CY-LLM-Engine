"""
tls.py
[gRPC TLS] 加载 gRPC 服务端 TLS 证书与密钥
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import grpc

LOGGER = logging.getLogger("cy_llm.worker.tls")


@dataclass
class GRPCTLSConfig:
    certificate_chain: bytes
    private_key: bytes
    credentials: grpc.ServerCredentials


ENV_TLS_CERT_PATH = "CY_LLM_GRPC_TLS_CERT"
ENV_TLS_KEY_PATH = "CY_LLM_GRPC_TLS_KEY"
ENV_TLS_CA_PATH = "CY_LLM_GRPC_TLS_CA"
ENV_TLS_ALLOW_INSECURE = "CY_LLM_ALLOW_INSECURE_GRPC"
ENV_APP_ENV = "CY_LLM_ENV"


def load_grpc_tls_config() -> Optional[GRPCTLSConfig]:
    cert_path, key_path, ca_path = resolve_grpc_tls_paths()
    if not cert_path or not key_path:
        if is_insecure_grpc_allowed():
            LOGGER.warning("gRPC TLS 未配置，允许以明文模式运行（开发模式）。")
            return None
        raise RuntimeError("gRPC TLS cert/key is required in production")

    certificate_chain = _read_file(cert_path)
    private_key = _read_file(key_path)
    root_certificates = _read_file(ca_path) if ca_path else None

    credentials = grpc.ssl_server_credentials(
        [(private_key, certificate_chain)],
        root_certificates=root_certificates,
    )
    return GRPCTLSConfig(
        certificate_chain=certificate_chain,
        private_key=private_key,
        credentials=credentials,
    )


def resolve_grpc_tls_paths() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    cert_path = os.getenv(ENV_TLS_CERT_PATH)
    key_path = os.getenv(ENV_TLS_KEY_PATH)
    ca_path = os.getenv(ENV_TLS_CA_PATH)

    if cert_path and key_path:
        return cert_path, key_path, ca_path

    for cert, key in _default_tls_candidates():
        if Path(cert).is_file() and Path(key).is_file():
            return cert, key, ca_path

    return None, None, ca_path


def _default_tls_candidates() -> Tuple[Tuple[str, str], ...]:
    return (
        ("./certs/fullchain.pem", "./certs/privkey.pem"),
        ("./certs/server.crt", "./certs/server.key"),
        ("/etc/cy-llm/certs/fullchain.pem", "/etc/cy-llm/certs/privkey.pem"),
        ("/etc/ssl/certs/server.crt", "/etc/ssl/private/server.key"),
    )


def _read_file(path: Optional[str]) -> Optional[bytes]:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"TLS file not found: {file_path}")
    return file_path.read_bytes()


def is_insecure_grpc_allowed() -> bool:
    env_value = os.getenv(ENV_APP_ENV, "development").lower().strip()
    allow_flag = os.getenv(ENV_TLS_ALLOW_INSECURE, "false").lower().strip() in {"1", "true", "yes"}
    return env_value != "production" or allow_flag
