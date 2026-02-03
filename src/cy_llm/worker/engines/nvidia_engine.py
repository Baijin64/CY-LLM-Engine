"""Nvidia GPU inference engine implementation."""

from threading import Thread
from typing import Any, Dict, Generator, Optional
import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from peft import PeftModel

# 尝试导入抽象基类，如果是在包内运行
try:
    from .abstract_engine import BaseEngine
except ImportError:
    # 如果是单独运行此脚本测试
    from abstract_engine import BaseEngine


class NvidiaEngine(BaseEngine):
    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self, model_path: str, adapter_path: Optional[str] = None, **kwargs) -> None:
        """加载模型并根据需要应用 LoRA 适配器。"""

        try:
            # === 阶段 0: 模型路径预处理 (检测/下载) ===
            try:
                from worker.utils.model_manager import ModelManager, LoadMode
                manager = ModelManager(
                    engine_type="nvidia",
                    stall_threshold_seconds=600.0,
                )
                # 这将触发本地检测或带进度条的下载
                print(f"[NvidiaEngine] Preparing model: {model_path}...")
                resolved_path, _ = manager.prepare_model(
                    model_id=model_path,
                    mode=LoadMode.CHECK_AND_LOAD,
                )
                model_path = str(resolved_path)
            except Exception as e:
                print(f"[NvidiaEngine] ModelManager scan skipped: {e}")

            print(f"[NvidiaEngine] Stage 1/3: Initializing tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # 兼容两种写法：use_4bit 或 quantization
            use_4bit = kwargs.pop("use_4bit", True)
            quantization = kwargs.pop("quantization", None)
            if quantization == "bitsandbytes":
                use_4bit = True

            device_map = kwargs.pop("device_map", "auto")
            quantization_config = kwargs.pop("quantization_config", None)

            if quantization_config is None and use_4bit and torch.cuda.is_available():
                print("[NvidiaEngine] Stage 1.1: Configuring 4-bit quantization (BNB)...")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )

            model_kwargs: Dict[str, Any] = {
                "trust_remote_code": True,
                "device_map": device_map,
            }

            if quantization_config is not None:
                model_kwargs["quantization_config"] = quantization_config
            else:
                if torch.cuda.is_available():
                    model_kwargs["torch_dtype"] = torch.float16
                else:
                    model_kwargs["torch_dtype"] = torch.float32
                    model_kwargs["device_map"] = "cpu"

            print(f"[NvidiaEngine] Stage 2/3: Loading model weights (this may take several minutes)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs,
            )

            if adapter_path:
                print(f"[NvidiaEngine] Stage 2.2: Applying LoRA adapter: {adapter_path}...")
                self.model = PeftModel.from_pretrained(self.model, adapter_path)

            print("[NvidiaEngine] Stage 3/3: Finalizing model setup...")
            self.model.eval()
            self.device = next(self.model.parameters()).device
            print("[NvidiaEngine] ✓ Model load complete. Ready for inference.")

        except Exception as e:
            print(f"[NvidiaEngine] ✗ Error loading model: {e}")
            raise e

    def infer(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Please call load_model() first.")

        # 默认生成参数
        generation_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", 512),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        # 构建输入
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # 使用 Streamer 实现流式输出
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs["input_ids"] = inputs.input_ids
        generation_kwargs["attention_mask"] = inputs.attention_mask
        generation_kwargs["streamer"] = streamer
        
        # 在独立线程中运行 generate，主线程读取 streamer
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text

    def unload_model(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        gc.collect()
        torch.cuda.empty_cache()
        print("[NvidiaEngine] Model unloaded and GPU memory cleared.")

    def get_memory_usage(self) -> Dict[str, float]:
        if not torch.cuda.is_available():
            return {"used": 0.0, "total": 0.0}
        
        free, total = torch.cuda.mem_get_info()
        used = (total - free) / 1024**3
        total_gb = total / 1024**3
        return {"used": used, "total": total_gb}

# 简单的测试代码
if __name__ == "__main__":
    engine = NvidiaEngine()
    # 注意：这里需要替换为你实际的模型路径
    # engine.load_model("deepseek-ai/deepseek-llm-7b-chat")
    # for chunk in engine.infer("你好"):
    #     print(chunk, end="", flush=True)
