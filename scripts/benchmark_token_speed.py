#!/usr/bin/env python3
"""
benchmark_token_speed.py
Token速度基准测试脚本

用法:
    python scripts/benchmark_token_speed.py --model MODEL_NAME --engine ENGINE_TYPE --output RESULT.json

示例:
    python scripts/benchmark_token_speed.py --model deepseek-ai/deepseek-llm-7b-chat --engine cuda-vllm --output baseline.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_hardware_info() -> Dict[str, Any]:
    """获取硬件信息"""
    info = {
        "python_version": sys.version,
        "timestamp": datetime.now().isoformat(),
    }

    # CUDA信息
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_names"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
            info["gpu_memory_gb"] = [
                torch.cuda.get_device_properties(i).total_memory / (1024**3)
                for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        info["cuda_available"] = False
        info["note"] = "PyTorch not installed"

    # vLLM版本
    try:
        import vllm

        info["vllm_version"] = vllm.__version__
    except ImportError:
        info["vllm_version"] = "not_installed"

    return info


def benchmark_engine(
    model_name: str,
    engine_type: str,
    prompt: str,
    max_tokens: int = 256,
    warmup: bool = True,
) -> Dict[str, Any]:
    """
    基准测试指定引擎的token生成速度

    Returns:
        包含性能指标的字典
    """
    print(f"\n{'=' * 60}")
    print(f"Benchmarking: model={model_name}, engine={engine_type}")
    print(f"{'=' * 60}\n")

    # 导入引擎工厂
    try:
        from CY_LLM_Backend.worker.engines.engine_factory import create_engine
    except ImportError:
        try:
            from worker.engines.engine_factory import create_engine
        except ImportError as e:
            print(f"Error: Cannot import engine_factory: {e}")
            sys.exit(1)

    # 创建引擎
    print(f"Creating engine: {engine_type}...")
    start_init = time.time()
    try:
        engine = create_engine(engine_type)
    except Exception as e:
        print(f"Error creating engine: {e}")
        return {"error": f"Engine creation failed: {e}"}
    init_time = (time.time() - start_init) * 1000
    print(f"Engine created in {init_time:.1f}ms")

    # 加载模型
    print(f"Loading model: {model_name}...")
    print("(This may take a few minutes for first download)")
    start_load = time.time()
    try:
        engine.load_model(model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        return {"error": f"Model loading failed: {e}"}
    load_time = (time.time() - start_load) * 1000
    print(f"Model loaded in {load_time:.1f}ms")

    # Warmup run (不计入统计)
    if warmup:
        print("\nWarmup run...")
        warmup_prompt = "Hello"
        list(engine.infer(warmup_prompt, max_new_tokens=10))
        print("Warmup complete")

    # 正式测试
    print(
        f"\nRunning benchmark with prompt: '{prompt[:50]}...' "
        if len(prompt) > 50
        else f"\nRunning benchmark with prompt: '{prompt}'"
    )
    print(f"Target tokens: {max_tokens}")

    tokens = []
    ttft_recorded = False
    ttft_ms = 0.0
    start_time = time.time()
    first_token_time = None

    try:
        for i, chunk in enumerate(engine.infer(prompt, max_new_tokens=max_tokens)):
            tokens.append(chunk)

            # 记录首token时间 (TTFT)
            if not ttft_recorded:
                first_token_time = time.time()
                ttft_ms = (first_token_time - start_time) * 1000
                ttft_recorded = True
                print(f"First token received after {ttft_ms:.1f}ms")

        end_time = time.time()

    except Exception as e:
        print(f"Error during inference: {e}")
        engine.unload_model()
        return {"error": f"Inference failed: {e}"}

    # 计算指标
    total_time_ms = (end_time - start_time) * 1000
    generation_time_ms = (end_time - first_token_time) * 1000 if first_token_time else total_time_ms

    # 估算token数（简单按空格和字符估算，实际应用应使用tokenizer）
    full_text = "".join(tokens)
    # 简单估算：英文约4字符/token，中文约1.5字符/token
    estimated_tokens = len(full_text) / 3.0  # 粗略估算

    tokens_per_sec = (estimated_tokens / generation_time_ms) * 1000 if generation_time_ms > 0 else 0

    # 卸载模型
    engine.unload_model()

    result = {
        "model": model_name,
        "engine": engine_type,
        "prompt": prompt,
        "prompt_length_chars": len(prompt),
        "max_tokens_setting": max_tokens,
        "metrics": {
            "tokens_per_second": round(tokens_per_sec, 2),
            "ttft_ms": round(ttft_ms, 2),
            "total_time_ms": round(total_time_ms, 2),
            "generation_time_ms": round(generation_time_ms, 2),
            "estimated_output_tokens": round(estimated_tokens, 0),
            "output_length_chars": len(full_text),
            "init_time_ms": round(init_time, 2),
            "load_time_ms": round(load_time, 2),
        },
        "output_preview": full_text[:200] + "..." if len(full_text) > 200 else full_text,
        "hardware_info": get_hardware_info(),
    }

    # 打印结果
    print(f"\n{'=' * 60}")
    print("BENCHMARK RESULTS")
    print(f"{'=' * 60}")
    print(f"Token Speed:       {result['metrics']['tokens_per_second']:.2f} tokens/s")
    print(f"TTFT:              {result['metrics']['ttft_ms']:.2f} ms")
    print(f"Total Time:        {result['metrics']['total_time_ms']:.2f} ms")
    print(f"Output Tokens:     {result['metrics']['estimated_output_tokens']:.0f} (estimated)")
    print(f"Output Preview:    {result['output_preview'][:100]}...")
    print(f"{'=' * 60}\n")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark token generation speed for CY-LLM Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark with default settings
  python benchmark_token_speed.py --model deepseek-ai/deepseek-llm-7b-chat
  
  # Benchmark specific engine
  python benchmark_token_speed.py --model deepseek-ai/deepseek-llm-7b-chat --engine cuda-vllm-async
  
  # Save results to file
  python benchmark_token_speed.py --model deepseek-ai/deepseek-llm-7b-chat --output results.json
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/deepseek-llm-7b-chat",
        help="Model name or path (default: deepseek-ai/deepseek-llm-7b-chat)",
    )

    parser.add_argument(
        "--engine",
        type=str,
        default="cuda-vllm",
        choices=["cuda-vllm", "cuda-vllm-async", "nvidia", "cuda-trt"],
        help="Engine type (default: cuda-vllm)",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="请详细解释什么是人工智能，包括其历史、现状和未来发展趋势。",
        help="Test prompt (default: Chinese AI explanation request)",
    )

    parser.add_argument(
        "--max-tokens", type=int, default=256, help="Maximum tokens to generate (default: 256)"
    )

    parser.add_argument("--output", type=str, help="Output JSON file path to save results")

    parser.add_argument("--no-warmup", action="store_true", help="Skip warmup run")

    parser.add_argument("--runs", type=int, default=1, help="Number of benchmark runs (default: 1)")

    args = parser.parse_args()

    # 执行基准测试
    all_results = []
    for run in range(args.runs):
        if args.runs > 1:
            print(f"\n### Run {run + 1}/{args.runs} ###")

        result = benchmark_engine(
            model_name=args.model,
            engine_type=args.engine,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            warmup=(not args.no_warmup) and (run == 0),
        )

        all_results.append(result)

        if "error" in result:
            print(f"ERROR: {result['error']}")
            sys.exit(1)

    # 多轮测试统计
    if args.runs > 1:
        tps_values = [r["metrics"]["tokens_per_second"] for r in all_results]
        ttft_values = [r["metrics"]["ttft_ms"] for r in all_results]

        import statistics

        summary = {
            "runs": args.runs,
            "tokens_per_second": {
                "mean": round(statistics.mean(tps_values), 2),
                "stdev": round(statistics.stdev(tps_values), 2) if len(tps_values) > 1 else 0,
                "min": round(min(tps_values), 2),
                "max": round(max(tps_values), 2),
            },
            "ttft_ms": {
                "mean": round(statistics.mean(ttft_values), 2),
                "stdev": round(statistics.stdev(ttft_values), 2) if len(ttft_values) > 1 else 0,
                "min": round(min(ttft_values), 2),
                "max": round(max(ttft_values), 2),
            },
        }

        print(f"\n{'=' * 60}")
        print("SUMMARY STATISTICS")
        print(f"{'=' * 60}")
        print(
            f"Token Speed (t/s):  mean={summary['tokens_per_second']['mean']:.2f}, "
            f"stdev={summary['tokens_per_second']['stdev']:.2f}, "
            f"range=[{summary['tokens_per_second']['min']:.2f}, {summary['tokens_per_second']['max']:.2f}]"
        )
        print(
            f"TTFT (ms):          mean={summary['ttft_ms']['mean']:.2f}, "
            f"stdev={summary['ttft_ms']['stdev']:.2f}, "
            f"range=[{summary['ttft_ms']['min']:.2f}, {summary['ttft_ms']['max']:.2f}]"
        )
        print(f"{'=' * 60}\n")

        final_result = {
            "summary": summary,
            "runs": all_results,
        }
    else:
        final_result = all_results[0]

    # 保存结果
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_result, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {output_path}")

    # 打印最终JSON到stdout
    print("\nFinal Result (JSON):")
    print(json.dumps(final_result, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
