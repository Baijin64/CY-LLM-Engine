from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

try:
	from worker.core.telemetry import Telemetry  # type: ignore
except ImportError:
	Telemetry = None


class _HealthHandler(BaseHTTPRequestHandler):
	telemetry = Telemetry() if Telemetry is not None else None

	def do_GET(self) -> None:  # noqa: N802
		if self.path not in ("/healthz", "/metrics"):
			self.send_response(404)
			self.end_headers()
			return

		if self.path == "/healthz":
			payload = {"status": "ok"}
			self.send_response(200)
			self.send_header("Content-Type", "application/json")
			self.end_headers()
			self.wfile.write(json.dumps(payload, ensure_ascii=True).encode("utf-8"))
			return

		metrics = self.telemetry.export_prometheus() if self.telemetry else ""
		self.send_response(200)
		self.send_header("Content-Type", "text/plain; version=0.0.4")
		self.end_headers()
		self.wfile.write(metrics.encode("utf-8"))

	def log_message(self, format: str, *args) -> None:  # noqa: A003
		return


def start_health_server() -> HTTPServer:
	port = int(os.getenv("CY_LLM_HEALTH_PORT", "9090"))
	server = HTTPServer(("0.0.0.0", port), _HealthHandler)
	return server
