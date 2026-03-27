# turboquant-bench

Compare LLM inference with and without TurboQuant KV cache compression on Apple Silicon.

One repo, one command: see side-by-side speed, memory, and output quality differences between standard FP16 KV cache and TurboQuant compressed KV cache.

## Quick Start

```bash
git clone https://github.com/yzamari/turboquant-bench.git
cd turboquant-bench
python3.12 -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
tq-bench --model mlx-community/Qwen2.5-3B-Instruct-4bit --prompt "Explain quantum computing"
```

## What It Compares

| Mode | Keys | Values | Description |
|------|------|--------|-------------|
| **standard** | FP16 | FP16 | Default MLX-LM KV cache, no compression |
| **mlx-quantized** | 4-bit | 4-bit | MLX-LM built-in QuantizedKVCache (opt-in via `--include-mlx-quantized`) |
| **turboquant** | 3-bit | 2-bit | TurboQuant product quantization for keys + group quantization for values |

## Metrics Collected

- **Gen tok/s** -- Token generation throughput
- **Peak Mem MB** -- Peak unified memory usage
- **Token match %** -- Percentage of generated tokens identical to standard mode
- **Char match %** -- Character-level text similarity to standard mode

## Usage

```bash
# Basic comparison (standard vs turboquant)
tq-bench --model mlx-community/Qwen2.5-3B-Instruct-4bit --prompt "Explain quantum computing"

# Include MLX-LM built-in quantized cache
tq-bench --include-mlx-quantized --prompt "Write a poem about the ocean"

# Custom TurboQuant settings
tq-bench --key-bits 4 --value-bits 4 --buffer-size 256 --max-tokens 300

# Different model
tq-bench --model mlx-community/Llama-3.2-3B-Instruct-4bit --prompt "What is gravity?"
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `mlx-community/Qwen2.5-3B-Instruct-4bit` | MLX model path or HuggingFace ID |
| `--prompt` / `-p` | `Explain quantum computing in simple terms` | Input prompt |
| `--max-tokens` / `-m` | `200` | Max tokens to generate |
| `--key-bits` | `3` | Key compression bits (2, 3, or 4) |
| `--value-bits` | `2` | Value compression bits (2 or 4) |
| `--buffer-size` | `128` | Recent tokens kept uncompressed |
| `--temp` | `0.0` | Sampling temperature (0 = greedy) |
| `--include-mlx-quantized` | off | Also benchmark MLX-LM QuantizedKVCache |

## How TurboQuant Works

TurboQuant ([ICLR 2026 paper](https://arxiv.org/abs/2504.19874)) compresses the KV cache during inference:

- **Keys**: Product quantization using Lloyd-Max codebooks (MSE component) plus QJL random projection for residual inner-product estimation. Achieves 3-bit compression while preserving attention score accuracy.
- **Values**: Asymmetric group quantization down to 2 bits per element.
- **Buffer**: The most recent N tokens (default 128) are kept uncompressed in FP16 for quality, since the model attends most heavily to recent context.

The result is roughly 5x KV cache memory reduction with minimal quality degradation -- critical for fitting longer contexts on memory-constrained Apple Silicon devices.

## Related Repositories

- [turboQuantPlayground](https://github.com/yzamari/turboQuantPlayground) -- Core TurboQuant library (codebooks, quantizer, rotation, Metal kernels)
- [mlx-turboquant](https://github.com/yzamari/mlx-turboquant) -- MLX-LM integration layer (TurboQuantCache, model patching, generation)

## License

MIT
