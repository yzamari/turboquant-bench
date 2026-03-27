"""
CLI entry point for tq-bench.

Usage:
    tq-bench --model mlx-community/Qwen2.5-3B-Instruct-4bit --prompt "Explain quantum computing"
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Compare LLM inference with and without TurboQuant KV cache compression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  tq-bench --model mlx-community/Qwen2.5-3B-Instruct-4bit --prompt "Explain quantum computing"
  tq-bench --model mlx-community/Llama-3.2-3B-Instruct-4bit --max-tokens 300 --key-bits 3
  tq-bench --include-mlx-quantized --prompt "Write a haiku about AI"
        """,
    )
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen2.5-3B-Instruct-4bit",
        help="MLX model path or HuggingFace ID (default: Qwen2.5-3B-Instruct-4bit)",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        default="Explain quantum computing in simple terms",
        help="Input prompt",
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=200,
        help="Maximum tokens to generate (default: 200)",
    )
    parser.add_argument(
        "--key-bits",
        type=int,
        default=3,
        choices=[2, 3, 4],
        help="TurboQuant key compression bits (default: 3)",
    )
    parser.add_argument(
        "--value-bits",
        type=int,
        default=2,
        choices=[2, 4],
        help="TurboQuant value compression bits (default: 2)",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=128,
        help="TurboQuant uncompressed buffer size (default: 128)",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="Sampling temperature, 0 = greedy for deterministic comparison (default: 0.0)",
    )
    parser.add_argument(
        "--include-mlx-quantized",
        action="store_true",
        help="Also test MLX-LM's built-in QuantizedKVCache (4-bit)",
    )

    args = parser.parse_args()

    from tq_bench.compare import run_comparison
    from tq_bench.report import print_report

    modes = ["standard", "turboquant"]
    if args.include_mlx_quantized:
        modes.insert(1, "mlx-quantized")

    results = run_comparison(
        model_path=args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        key_bits=args.key_bits,
        value_bits=args.value_bits,
        buffer_size=args.buffer_size,
        temp=args.temp,
        modes=modes,
    )

    print_report(results, args.prompt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
