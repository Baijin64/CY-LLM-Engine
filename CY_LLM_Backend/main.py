"""Compatibility wrapper for legacy tests."""

import importlib
import sys

if "worker" not in sys.modules:
    sys.path.insert(0, __file__.rsplit("/", 1)[0])

_worker_main = importlib.import_module("worker.main")

parse_args = _worker_main.parse_args
_preload_models = _worker_main._preload_models
create_server = _worker_main.create_server
register_servicers = _worker_main.register_servicers
setup_reflection = _worker_main.setup_reflection
setup_signal_handlers = _worker_main.setup_signal_handlers
graceful_shutdown = _worker_main.graceful_shutdown
