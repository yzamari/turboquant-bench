"""
Core comparison engine: run inference in multiple KV cache modes and collect metrics.

Modes:
  - standard:       Default FP16 KV cache (mlx_lm default)
  - mlx-quantized:  Built-in QuantizedKVCache (4-bit)
  - turboquant:     TurboQuantCache (3-bit keys + 2-bit values)
"""

import time
from dataclasses import dataclass, field

import mlx.core as mx


@dataclass
class BenchmarkResult:
    """Metrics from a single inference run."""

    mode: str
    text: str
    tokens: list[int] = field(repr=False)
    num_tokens: int = 0
    prefill_ms: float = 0.0
    gen_tok_s: float = 0.0
    peak_mem_mb: float = 0.0
    cache_mem_mb: float = 0.0


def run_comparison(
    model_path: str,
    prompt: str,
    max_tokens: int = 200,
    key_bits: int = 3,
    value_bits: int = 2,
    buffer_size: int = 128,
    temp: float = 0.0,
    modes: list[str] | None = None,
) -> list[BenchmarkResult]:
    """
    Run inference in multiple KV cache modes and return results.

    Args:
        model_path: HuggingFace model ID or local path.
        prompt: User prompt string.
        max_tokens: Maximum tokens to generate per mode.
        key_bits: TurboQuant key compression bits (2-4).
        value_bits: TurboQuant value compression bits (2 or 4).
        buffer_size: TurboQuant uncompressed buffer size.
        temp: Sampling temperature (0 = greedy for deterministic comparison).
        modes: List of modes to run. Default: ["standard", "turboquant"].

    Returns:
        List of BenchmarkResult, one per mode.
    """
    if modes is None:
        modes = ["standard", "turboquant"]

    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_turboquant.patch import make_turboquant_cache

    print(f"Loading model: {model_path}")
    model, tokenizer = load(model_path)

    # Format prompt with chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        except Exception:
            prompt_text = prompt
    else:
        prompt_text = prompt

    prompt_tokens = mx.array(tokenizer.encode(prompt_text))
    print(f"Prompt tokens: {prompt_tokens.size}")

    results = []

    for mode in modes:
        result = _run_single_mode(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            prompt_tokens=prompt_tokens,
            mode=mode,
            max_tokens=max_tokens,
            temp=temp,
            key_bits=key_bits,
            value_bits=value_bits,
            buffer_size=buffer_size,
        )
        results.append(result)

    return results


def _run_single_mode(
    model,
    tokenizer,
    prompt_text: str,
    prompt_tokens: mx.array,
    mode: str,
    max_tokens: int,
    temp: float,
    key_bits: int,
    value_bits: int,
    buffer_size: int,
) -> BenchmarkResult:
    """Run inference for a single mode and return metrics."""
    import mlx_lm
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_turboquant.patch import make_turboquant_cache

    print(f"\n{'='*60}")
    print(f"Mode: {mode}")
    print(f"{'='*60}")

    # Build generation kwargs based on mode
    gen_kwargs: dict = dict(max_tokens=max_tokens)

    if mode == "standard":
        pass  # Default cache
    elif mode == "mlx-quantized":
        gen_kwargs["kv_bits"] = 4
        gen_kwargs["kv_group_size"] = 64
    elif mode == "turboquant":
        cache = make_turboquant_cache(
            model,
            key_bits=key_bits,
            value_bits=value_bits,
            buffer_size=buffer_size,
        )
        gen_kwargs["prompt_cache"] = cache
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Reset memory tracking
    mx.synchronize()
    mx.reset_peak_memory()

    # Run generation via stream_generate (handles all modes cleanly)
    text = ""
    start = time.perf_counter()

    response = None
    for response in mlx_lm.stream_generate(
        model, tokenizer, prompt_text, **gen_kwargs
    ):
        text += response.text

    elapsed = time.perf_counter() - start

    mx.synchronize()
    peak_mem = mx.get_peak_memory() / 1024 / 1024  # MB

    # Extract metrics from the final response object
    if response is None:
        print("  WARNING: No tokens generated")
        return BenchmarkResult(
            mode=mode, text="", tokens=[], num_tokens=0,
        )

    gen_tokens = response.generation_tokens
    gen_tps = response.generation_tps
    prompt_tps = response.prompt_tps

    # Get cache memory for TurboQuant mode
    cache_mem_mb = 0.0
    if mode == "turboquant" and "prompt_cache" in gen_kwargs:
        cache_mem_mb = sum(
            c.nbytes for c in gen_kwargs["prompt_cache"]
        ) / 1024 / 1024

    # Tokenize generated text for quality comparison
    generated_token_ids = tokenizer.encode(text)

    print(f"  Tokens:    {gen_tokens}")
    print(f"  Prefill:   {prompt_tps:.1f} tok/s")
    print(f"  Generate:  {gen_tps:.1f} tok/s")
    print(f"  Peak mem:  {peak_mem:.1f} MB")
    if cache_mem_mb > 0:
        print(f"  Cache mem: {cache_mem_mb:.1f} MB")
    print(f"  Elapsed:   {elapsed:.2f}s")

    return BenchmarkResult(
        mode=mode,
        text=text,
        tokens=generated_token_ids,
        num_tokens=gen_tokens,
        prefill_ms=0,  # prompt_tps gives throughput, not latency
        gen_tok_s=gen_tps,
        peak_mem_mb=peak_mem,
        cache_mem_mb=cache_mem_mb,
    )
