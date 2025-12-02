"""
ascend_engine.py
# [实现] AscendEngine：华为 Ascend / ACL / torch_npu 实现占位
# 说明：负责在 Ascend 硬件上加载模型与推理，处理特定运行时差异。
"""

from __future__ import annotations

from threading import Thread
from typing import Dict, Generator, Optional
import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel

from .abstract_engine import BaseEngine


def _ascend_available() -> bool:
	return bool(getattr(torch, "npu", None)) and getattr(torch.npu, "is_available", lambda: False)()


class AscendEngine(BaseEngine):
	"""Thin wrapper around Hugging Face models running on Ascend NPUs."""

	# 说明：该引擎封装了在 Ascend NPU 上加载与推理的常见流程。
	# - 使用 torch_npu / npu 设备（若可用）
	# - 支持可选的 LoRA 适配器加载
	# - 提供流式生成接口以兼容 gRPC/streamer 用法

	def __init__(self) -> None:
		self.model = None
		self.tokenizer = None
		self.device = torch.device("cpu")

	def _resolve_device(self) -> torch.device:
		if _ascend_available():
			return torch.device("npu")
		raise RuntimeError("Ascend device is not available or torch_npu is missing.")

	# 检查并返回 NPU 设备，否则抛出异常提示环境缺失

	def load_model(self, model_path: str, adapter_path: Optional[str] = None, **kwargs) -> None:
		"""Load a base model (and optional LoRA adapter) onto Ascend."""

		# 将模型移动到 NPU 并完成 tokenizer 加载
		self.device = self._resolve_device()
		print(f"[AscendEngine] Loading base model on {self.device}: {model_path}")

		self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
		if self.tokenizer.pad_token is None:
			self.tokenizer.pad_token = self.tokenizer.eos_token

		model_kwargs = {"trust_remote_code": True}
		dtype = kwargs.get("torch_dtype", torch.float16)

		self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, **model_kwargs)
		self.model.to(self.device)

		if adapter_path:
			print(f"[AscendEngine] Loading LoRA adapter: {adapter_path}")
			self.model = PeftModel.from_pretrained(self.model, adapter_path)

		self.model.eval()
		print("[AscendEngine] Model loaded successfully.")

	# 说明：load_model 在 Ascend 上不做量化处理（由上层决定），仅确保模型迁移到 NPU

	def infer(self, prompt: str, **kwargs) -> Generator[str, None, None]:
		if self.model is None or self.tokenizer is None:
			raise RuntimeError("Model not loaded. Please call load_model() first.")

		generation_kwargs = {
			"max_new_tokens": kwargs.get("max_new_tokens", 512),
			"temperature": kwargs.get("temperature", 0.7),
			"top_p": kwargs.get("top_p", 0.9),
			"repetition_penalty": kwargs.get("repetition_penalty", 1.1),
			"do_sample": kwargs.get("do_sample", True),
		}

		inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

		streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
		generation_kwargs.update({
			"input_ids": inputs.input_ids,
			"attention_mask": inputs.attention_mask,
			"streamer": streamer,
			"pad_token_id": self.tokenizer.pad_token_id,
			"eos_token_id": self.tokenizer.eos_token_id,
		})

		thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
		thread.start()

		for new_text in streamer:
			yield new_text

	# 说明：使用 TextIteratorStreamer 实现流式推理，适合边读边回传的场景

	def unload_model(self) -> None:
		if self.model is not None:
			del self.model
			self.model = None
		if self.tokenizer is not None:
			del self.tokenizer
			self.tokenizer = None

		gc.collect()
		if _ascend_available():
			torch.npu.empty_cache()  # type: ignore[attr-defined]
		print("[AscendEngine] Model unloaded and NPU memory cleared.")

	# 说明：卸载时执行 GC 并清理 NPU 缓存，释放内存供其他模型使用

	def get_memory_usage(self) -> Dict[str, float]:
		if not _ascend_available():
			return {"used": 0.0, "total": 0.0}

		try:
			device = torch.npu.current_device()  # type: ignore[attr-defined]
			used = torch.npu.memory_allocated(device) / 1024**3  # type: ignore[attr-defined]
			total = getattr(torch.npu.get_device_properties(device), "total_memory", 0) / 1024**3  # type: ignore[attr-defined]
		except AttributeError:
			return {"used": 0.0, "total": 0.0}

		return {"used": used, "total": total}

