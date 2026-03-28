# turboquant-bench

Compare LLM inference with and without TurboQuant KV cache compression on Apple Silicon.

One repo, one command: see side-by-side speed, memory, and output quality.

## Benchmark Results (M4 Pro 48GB)

### Qwen2.5-32B-Instruct-4bit

| Context | Standard (FP16) | TurboQuant (3-bit) | Speed | Quality |
|---------|----------------|-------------------|-------|---------|
| Short (36 tok) | 6.7 tok/s | 6.8 tok/s | 101% | **100% match** |
| Medium (690 tok) | 7.0 tok/s | 7.2 tok/s | 103% | **100% match** |
| Long (1130 tok) | 6.9 tok/s | **9.6 tok/s** | **139%** | **100% match** |
| Long gen (500 tok) | 6.9 tok/s | 6.7 tok/s | 97% | **100% match** |

**39% faster on long context** with the 32B model. Larger models are more memory-bandwidth-constrained, so TurboQuant's compressed KV cache gives a bigger advantage.

### Qwen2.5-3B-Instruct-4bit

| Context | Standard (FP16) | TurboQuant (3-bit) | Speed | Quality |
|---------|----------------|-------------------|-------|---------|
| Short (36 tok) | 70.6 tok/s | 69.1 tok/s | 98% | **100% match** |
| Medium (690 tok) | 61.6 tok/s | **69.4 tok/s** | **113%** | **100% match** |
| Long (1130 tok) | 59.4 tok/s | **63.2 tok/s** | **106%** | **100% match** |
| Long gen (500 tok) | 56.0 tok/s | 54.7 tok/s | 98% | **100% match** |

TurboQuant is **faster than standard on medium and long contexts** because the compressed KV cache uses less memory bandwidth. Output is **100% identical** to standard FP16 inference at 3-bit.

### Memory Savings (theoretical, at scale)

| Compressed Tokens | FP16 Cache | TurboQuant 3-bit | Compression |
|------------------|-----------|-----------------|-------------|
| 4,096 | 512 MB | 113 MB | 4.5x |
| 16,384 | 2,048 MB | 413 MB | 5.0x |

## Quick Start

```bash
git clone https://github.com/yzamari/turboquant-bench.git
cd turboquant-bench
python3.12 -m venv venv && source venv/bin/activate
pip install -e ".[dev]"

# Run comparison
tq-bench --model mlx-community/Qwen2.5-3B-Instruct-4bit \
         --prompt "Explain quantum computing"
```

### Full multi-config benchmark

```bash
python benchmarks/bench_compare.py mlx-community/Qwen2.5-3B-Instruct-4bit
```

### Test with larger models

```bash
# 7B
tq-bench --model mlx-community/Qwen2.5-7B-Instruct-4bit --prompt "Write a web scraper"

# 32B (needs ~20GB)
tq-bench --model mlx-community/Qwen2.5-32B-Instruct-4bit --prompt "Design a REST API"

# 72B (needs ~38GB, for 48GB+ Macs)
python benchmarks/bench_compare.py mlx-community/Qwen2.5-72B-Instruct-4bit
```

## What It Compares

| Mode | Keys | Values | Description |
|------|------|--------|-------------|
| **standard** | FP16 | FP16 | Default MLX-LM KV cache, no compression |
| **mlx-quantized** | 4-bit | 4-bit | MLX-LM built-in QuantizedKVCache (opt-in via `--include-mlx-quantized`) |
| **turboquant** | 3-bit | 2-bit | TurboQuant: Metal kernel scores + fused value weighted sum |

## Metrics Collected

- **Gen tok/s** -- token generation throughput
- **Peak Mem MB** -- peak unified memory usage
- **Token match %** -- percentage of tokens identical to standard mode
- **Char match %** -- character-level text similarity

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `Qwen2.5-3B-Instruct-4bit` | MLX model path or HuggingFace ID |
| `--prompt` / `-p` | `Explain quantum computing in simple terms` | Input prompt |
| `--max-tokens` / `-m` | `200` | Max tokens to generate |
| `--key-bits` | `3` | Key compression bits (2, 3, or 4) |
| `--value-bits` | `2` | Value compression bits (2 or 4) |
| `--buffer-size` | `128` | Recent tokens kept uncompressed |
| `--temp` | `0.0` | Sampling temperature (0 = greedy for deterministic comparison) |
| `--include-mlx-quantized` | off | Also benchmark MLX-LM QuantizedKVCache |

## How It Works

TurboQuant ([ICLR 2026](https://arxiv.org/abs/2504.19874)) compresses the KV cache during inference using three optimizations:

1. **Zero-decompression Metal kernels** -- attention scores and value weighted sums computed directly from packed 2-3 bit data. No FP16 intermediate tensors ever created.
2. **Batch flush** -- tokens compressed 128 at a time (GPU-saturated) instead of 1 at a time (underutilized).
3. **Prefill bypass** -- standard MLX attention during prompt processing, Metal kernels only during decode.

**Keys**: random rotation + Lloyd-Max codebook (MSE-optimal) + QJL sign sketching (unbiased inner products).
**Values**: asymmetric group quantization (2-bit, min-max per group of 32).
**Buffer**: recent 128 tokens kept in FP16 for quality.

## Project Ecosystem

| Repo | Purpose | URL |
|------|---------|-----|
| **turboQuantPlayground** | Core algorithm, Metal kernels, notebooks | https://github.com/yzamari/turboQuantPlayground |
| **mlx-turboquant** | MLX-LM integration for real inference | https://github.com/yzamari/mlx-turboquant |
| **turboquant-bench** | This repo: comparison benchmarks | https://github.com/yzamari/turboquant-bench |

## References

- [TurboQuant paper (ICLR 2026)](https://arxiv.org/abs/2504.19874)
- [Google Research Blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- [0xSero/turboquant](https://github.com/0xSero/turboquant) -- upstream NVIDIA Triton implementation

## License

MIT
