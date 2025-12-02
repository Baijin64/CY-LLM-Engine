"""
full_finetune.py
[训练] 全参数微调脚本（从零训练）
说明：支持 DeepSpeed ZeRO、混合精度训练、分布式训练。

使用示例：
    # 单卡训练
    python full_finetune.py \
        --model_name deepseek-ai/deepseek-llm-7b-base \
        --dataset_path ./data/train.jsonl \
        --output_dir ./checkpoints/full_ft

    # 多卡 DeepSpeed ZeRO-3
    deepspeed --num_gpus 4 full_finetune.py \
        --model_name deepseek-ai/deepseek-llm-7b-base \
        --dataset_path ./data/train.jsonl \
        --output_dir ./checkpoints/full_ft \
        --deepspeed ds_config_zero3.json

进度输出格式（供 CustomScriptRunner 解析）：
    {"ew_progress": {"epoch": 1, "step": 100, "loss": 2.5, "lr": 1e-5}}
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger("ew.training.full_finetune")

# 延迟导入重型依赖
def import_training_deps():
    """延迟导入训练依赖"""
    global torch, AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    global Trainer, TrainerCallback, TrainerState, TrainerControl
    global load_dataset, DataCollatorForLanguageModeling
    
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        TrainerCallback,
        TrainerState,
        TrainerControl,
        DataCollatorForLanguageModeling,
    )
    from datasets import load_dataset


class ProgressCallback(TrainerCallback):
    """进度回调，输出 JSON 格式的进度信息"""
    
    def __init__(self, job_id: str = ""):
        self.job_id = job_id
        self.start_time = None
    
    def on_train_begin(self, args, state, control, **kwargs):
        import time
        self.start_time = time.time()
        self._emit_progress(
            status="running",
            message="Training started",
            epoch=0,
            step=0,
            total_steps=state.max_steps,
        )
    
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is None:
            return
        
        import time
        elapsed = time.time() - self.start_time if self.start_time else 0
        eta = 0
        if state.global_step > 0 and state.max_steps > 0:
            eta = (elapsed / state.global_step) * (state.max_steps - state.global_step)
        
        self._emit_progress(
            status="running",
            epoch=int(state.epoch) if state.epoch else 0,
            total_epochs=int(args.num_train_epochs),
            step=state.global_step,
            total_steps=state.max_steps,
            loss=logs.get("loss", 0.0),
            lr=logs.get("learning_rate", 0.0),
            elapsed_seconds=elapsed,
            eta_seconds=eta,
        )
    
    def on_train_end(self, args, state, control, **kwargs):
        import time
        elapsed = time.time() - self.start_time if self.start_time else 0
        self._emit_progress(
            status="completed",
            message="Training completed",
            epoch=int(args.num_train_epochs),
            total_epochs=int(args.num_train_epochs),
            step=state.global_step,
            total_steps=state.max_steps,
            elapsed_seconds=elapsed,
        )
    
    def _emit_progress(self, **kwargs):
        """输出 JSON 格式的进度信息"""
        progress = {"ew_progress": {**kwargs, "job_id": self.job_id}}
        print(json.dumps(progress), flush=True)


def create_deepspeed_config(
    stage: int = 2,
    offload_optimizer: bool = False,
    offload_param: bool = False,
) -> Dict[str, Any]:
    """
    创建 DeepSpeed 配置
    
    Args:
        stage: ZeRO 阶段 (1, 2, 3)
        offload_optimizer: 是否将优化器状态卸载到 CPU
        offload_param: 是否将参数卸载到 CPU (仅 ZeRO-3)
    """
    config = {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": 1.0,
        "bf16": {
            "enabled": True,
        },
        "zero_optimization": {
            "stage": stage,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
        },
        "zero_allow_untested_optimizer": True,
        "wall_clock_breakdown": False,
    }
    
    # ZeRO-3 特定配置
    if stage >= 3:
        config["zero_optimization"]["stage3_prefetch_bucket_size"] = 5e7
        config["zero_optimization"]["stage3_param_persistence_threshold"] = 1e5
        config["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"] = True
    
    # 优化器卸载
    if offload_optimizer:
        config["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True,
        }
    
    # 参数卸载 (仅 ZeRO-3)
    if offload_param and stage >= 3:
        config["zero_optimization"]["offload_param"] = {
            "device": "cpu",
            "pin_memory": True,
        }
    
    return config


def prepare_dataset(
    tokenizer,
    dataset_path: str,
    max_seq_length: int = 2048,
    text_column: str = "text",
    num_proc: int = 4,
):
    """
    准备训练数据集
    
    支持格式：
    1. JSONL 文件，每行包含 "text" 字段
    2. JSONL 文件，每行包含 "instruction", "input", "output" 字段（Alpaca 格式）
    3. HuggingFace 数据集名称
    """
    LOGGER.info("Loading dataset from: %s", dataset_path)
    
    # 加载数据集
    if dataset_path.endswith((".json", ".jsonl")):
        dataset = load_dataset("json", data_files=dataset_path, split="train")
    else:
        dataset = load_dataset(dataset_path, split="train")
    
    LOGGER.info("Dataset size: %d samples", len(dataset))
    
    # 检测数据格式
    columns = dataset.column_names
    
    def tokenize_function(examples):
        # Alpaca 格式
        if "instruction" in columns:
            texts = []
            for i in range(len(examples["instruction"])):
                instruction = examples["instruction"][i]
                input_text = examples.get("input", [""] * len(examples["instruction"]))[i]
                output_text = examples["output"][i]
                
                if input_text:
                    text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"
                else:
                    text = f"### Instruction:\n{instruction}\n\n### Response:\n{output_text}"
                texts.append(text)
        # 纯文本格式
        elif text_column in columns:
            texts = examples[text_column]
        else:
            raise ValueError(f"Dataset must have '{text_column}' or 'instruction/output' columns")
        
        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            return_tensors=None,
        )
        
        # 对于 Causal LM，labels = input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    # 处理数据集
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=columns,
        desc="Tokenizing",
    )
    
    return tokenized_dataset


def train(args):
    """主训练函数"""
    import_training_deps()
    
    LOGGER.info("=" * 60)
    LOGGER.info("Full Fine-Tuning Configuration")
    LOGGER.info("=" * 60)
    LOGGER.info("Model: %s", args.model_name)
    LOGGER.info("Dataset: %s", args.dataset_path)
    LOGGER.info("Output: %s", args.output_dir)
    LOGGER.info("Batch size: %d x %d (gradient accumulation)", args.batch_size, args.grad_accum)
    LOGGER.info("Learning rate: %s", args.learning_rate)
    LOGGER.info("Epochs: %d", args.epochs)
    LOGGER.info("Max sequence length: %d", args.max_seq_length)
    LOGGER.info("=" * 60)
    
    # 1. 加载 Tokenizer
    LOGGER.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # 2. 加载模型
    LOGGER.info("Loading model...")
    model_kwargs = {
        "trust_remote_code": True,
    }
    
    # 混合精度
    if args.bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif args.fp16:
        model_kwargs["torch_dtype"] = torch.float16
    
    # 设备映射
    if args.deepspeed:
        # DeepSpeed 自己处理设备映射
        pass
    elif torch.cuda.device_count() > 1:
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = {"": 0}
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **model_kwargs,
    )
    
    # 启用梯度检查点以节省显存
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    
    LOGGER.info("Model parameters: %.2fB", model.num_parameters() / 1e9)
    
    # 3. 准备数据集
    train_dataset = prepare_dataset(
        tokenizer=tokenizer,
        dataset_path=args.dataset_path,
        max_seq_length=args.max_seq_length,
        num_proc=args.num_workers,
    )
    
    # 4. 配置训练参数
    training_args_dict = {
        "output_dir": args.output_dir,
        "overwrite_output_dir": True,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "max_grad_norm": args.max_grad_norm,
        "dataloader_num_workers": args.num_workers,
        "optim": args.optimizer,
        "report_to": "none",
        "ddp_find_unused_parameters": False,
        "remove_unused_columns": False,
    }
    
    # DeepSpeed 配置
    if args.deepspeed:
        if os.path.isfile(args.deepspeed):
            training_args_dict["deepspeed"] = args.deepspeed
        else:
            # 自动生成配置
            ds_config = create_deepspeed_config(
                stage=args.zero_stage,
                offload_optimizer=args.offload_optimizer,
                offload_param=args.offload_param,
            )
            ds_config_path = os.path.join(args.output_dir, "ds_config_auto.json")
            os.makedirs(args.output_dir, exist_ok=True)
            with open(ds_config_path, "w") as f:
                json.dump(ds_config, f, indent=2)
            training_args_dict["deepspeed"] = ds_config_path
            LOGGER.info("Auto-generated DeepSpeed config: %s", ds_config_path)
    
    # 恢复训练
    if args.resume_from_checkpoint:
        training_args_dict["resume_from_checkpoint"] = args.resume_from_checkpoint
        LOGGER.info("Resuming from checkpoint: %s", args.resume_from_checkpoint)
    
    training_args = TrainingArguments(**training_args_dict)
    
    # 5. 创建进度回调
    progress_callback = ProgressCallback(job_id=args.job_id)
    
    # 6. 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[progress_callback],
    )
    
    # 7. 开始训练
    LOGGER.info("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    # 8. 保存模型
    LOGGER.info("Saving model to: %s", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    LOGGER.info("Training completed!")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Full Fine-Tuning Script for Large Language Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # 必需参数
    parser.add_argument(
        "--model_name", type=str, required=True,
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True,
        help="Path to training dataset (JSONL) or HuggingFace dataset name",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./checkpoints/full_ft",
        help="Output directory for model checkpoints",
    )
    
    # 训练超参数
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--lr_scheduler", type=str, default="cosine",
                        choices=["linear", "cosine", "constant", "constant_with_warmup"],
                        help="Learning rate scheduler")
    parser.add_argument("--optimizer", type=str, default="adamw_torch",
                        choices=["adamw_torch", "adamw_hf", "sgd", "adafactor"],
                        help="Optimizer")
    
    # 混合精度
    parser.add_argument("--bf16", action="store_true", help="Use BF16 precision")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 precision")
    
    # 内存优化
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                        help="Enable gradient checkpointing to save memory")
    
    # DeepSpeed
    parser.add_argument("--deepspeed", type=str, default=None,
                        help="DeepSpeed config file path or 'auto' to generate")
    parser.add_argument("--zero_stage", type=int, default=2, choices=[1, 2, 3],
                        help="ZeRO optimization stage (for auto config)")
    parser.add_argument("--offload_optimizer", action="store_true",
                        help="Offload optimizer state to CPU")
    parser.add_argument("--offload_param", action="store_true",
                        help="Offload parameters to CPU (ZeRO-3 only)")
    
    # 保存和日志
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging frequency")
    parser.add_argument("--save_steps", type=int, default=500, help="Checkpoint save frequency")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Max checkpoints to keep")
    
    # 恢复训练
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    # 其他
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--job_id", type=str, default="", help="Job ID for progress tracking")
    
    args = parser.parse_args()
    
    # 默认启用 bf16
    if not args.bf16 and not args.fp16:
        args.bf16 = True
    
    return train(args)


if __name__ == "__main__":
    sys.exit(main())
