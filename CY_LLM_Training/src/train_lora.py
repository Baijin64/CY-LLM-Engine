import os
import sys
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from trl import SFTTrainer
from datasets import load_dataset, concatenate_datasets
import argparse

def train(args):
    # 1. 配置模型加载参数 (QLoRA 4-bit)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading base model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True, # DeepSeek 可能需要
        # attn_implementation="flash_attention_2" # 可以不用 flash_attn 库，自动使用 PyTorch 内置加速
    )
    
    model.config.use_cache = False # 训练时关闭 cache
    model = prepare_model_for_kbit_training(model)

    # 2. 配置 LoRA
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # 针对所有线性层
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 3. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # 训练时通常 right padding

    # 4. 加载数据集
    if args.settings_path:
        print(f"Loading dialogue dataset from {args.dataset_path}")
        dialogue_ds = load_dataset("json", data_files=args.dataset_path, split="train")
        
        print(f"Loading settings dataset from {args.settings_path}")
        settings_ds = load_dataset("json", data_files=args.settings_path, split="train")
        
        # 提高 settings 权重 (过采样)
        # 设定数据通常较少，为了让模型记住设定，我们将其复制多份
        # 这里设置为 5 倍权重，你可以根据实际数据量比例调整
        settings_weight = 5
        print(f"Oversampling settings dataset {settings_weight} times...")
        settings_ds_oversampled = concatenate_datasets([settings_ds] * settings_weight)
        
        # 合并数据集
        dataset = concatenate_datasets([dialogue_ds, settings_ds_oversampled])
        dataset = dataset.shuffle(seed=42) # 这一点很重要，必须打乱
        print(f"Combined dataset size: {len(dataset)} (Dialogue: {len(dialogue_ds)}, Settings x{settings_weight}: {len(settings_ds_oversampled)})")
    else:
        print(f"Loading dataset from {args.dataset_path}")
        dataset = load_dataset("json", data_files=args.dataset_path, split="train")

    # 5. 定义格式化函数
    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['instruction'])):
            # 构建 Prompt 格式:
            # DeepSeek V3/V2 推荐使用 ChatML 格式或 Alpaca 格式，这里使用 Alpaca 变体
            # 如果模型有特定的 chat_template，最好使用 tokenizer.apply_chat_template
            
            instruction = example['instruction'][i]
            input_str = example['input'][i]
            output_str = example['output'][i]
            
            if input_str:
                text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_str}\n\n### Response:\n{output_str}<|endoftext|>"
            else:
                text = f"### Instruction:\n{instruction}\n\n### Response:\n{output_str}<|endoftext|>"
            
            output_texts.append(text)
        return output_texts

    # 6. 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=10,
        num_train_epochs=args.epochs,
        save_steps=100,
        fp16=False,
        bf16=True, # DeepSeek 建议使用 bf16
        optim="paged_adamw_32bit",
        report_to="none", # 可以暂时不用 tensorboard 日志
        # 针对 DeepSeek V3 可能的长文本优化
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
    )

    # 7. 初始化 Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        args=training_args,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        packing=False, # 对于短对话，packing=False 可能更稳定
    )

    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 更新默认模型为 DeepSeek V3.2-7B (假设路径或ID)
    parser.add_argument("--model_name", type=str, default="deepseek-ai/deepseek-llm-7b-chat", help="HuggingFace model ID or local path")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to processed JSONL dataset (e.g. furina_train.jsonl)")
    parser.add_argument("--settings_path", type=str, default=None, help="Path to settings JSONL dataset (e.g. settings_train.jsonl) to be oversampled")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/adapter", help="Output directory for LoRA weights")
    parser.add_argument("--batch_size", type=int, default=2) # 4bit 显存优化
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=64) # 增加秩以提升表现
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--max_seq_length", type=int, default=4096) # DeepSeek 支持长窗口
    
    args = parser.parse_args()
    train(args)
