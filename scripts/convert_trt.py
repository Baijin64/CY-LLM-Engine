#!/usr/bin/env python3
"""TensorRT-LLM 模型转换工具"""

import argparse
import subprocess
from pathlib import Path


def convert_checkpoint(args):
    """步骤 1: 转换 checkpoint"""
    cmd = [
        "python", "-m", "tensorrt_llm.commands.convert_checkpoint",
        "--model_type", args.model_type,
        "--model_dir", args.model,
        "--output_dir", args.checkpoint_dir,
        "--dtype", args.dtype,
    ]

    if args.tp_size > 1:
        cmd.extend(["--tp_size", str(args.tp_size)])

    print(f"运行: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def build_engine(args):
    """步骤 2: 构建 TRT 引擎"""
    cmd = [
        "trtllm-build",
        "--checkpoint_dir", args.checkpoint_dir,
        "--output_dir", args.output,
        "--gemm_plugin", args.dtype,
        "--max_batch_size", str(args.max_batch_size),
        "--max_input_len", str(args.max_input_len),
        "--max_output_len", str(args.max_output_len),
    ]

    print(f"运行: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="转换模型为 TensorRT-LLM 引擎")
    parser.add_argument("--model", required=True, help="HuggingFace 模型路径")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--model-type", default="llama", help="模型类型")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--tp-size", type=int, default=1, help="张量并行度")
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--max-input-len", type=int, default=4096)
    parser.add_argument("--max-output-len", type=int, default=2048)

    args = parser.parse_args()
    args.checkpoint_dir = f"{args.output}_checkpoint"

    print("🔧 步骤 1: 转换 Checkpoint")
    convert_checkpoint(args)

    print("\n🔨 步骤 2: 构建 TRT 引擎")
    build_engine(args)

    print(f"\n✅ 转换完成！引擎保存在: {args.output}")
    print(f"   在 config.json 中使用: \"model_path\": \"{args.output}\"")


if __name__ == "__main__":
    main()
