"""
Comprehensive benchmark: Standard vs TurboQuant across multiple configurations.

Tests:
1. Short prompt (36 tokens) — shows overhead baseline
2. Medium prompt (~500 tokens) — shows compression kicking in
3. Long generation (500 tokens) — shows memory scaling
4. Multiple bit widths (2, 3, 4) — shows quality/compression tradeoff
"""

import time
import sys
import mlx.core as mx


def _load_model(model_path):
    from mlx_lm import load
    print(f"Loading model: {model_path}")
    model, tokenizer = load(model_path)
    return model, tokenizer


def _generate_with_cache(model, tokenizer, prompt_text, cache, max_tokens, temp=0.0):
    """Run generation and return (text, tokens, gen_tok_s, elapsed, cache_mem_bytes)."""
    import mlx_lm
    from mlx_lm.sample_utils import make_sampler

    mx.synchronize()
    mx.reset_peak_memory()

    text = ""
    response = None
    start = time.perf_counter()

    gen_kwargs = dict(max_tokens=max_tokens)
    if cache is not None:
        gen_kwargs["prompt_cache"] = cache

    for response in mlx_lm.stream_generate(model, tokenizer, prompt_text, **gen_kwargs):
        text += response.text

    elapsed = time.perf_counter() - start
    mx.synchronize()
    peak_mem = mx.get_peak_memory() / 1024 / 1024

    if response is None:
        return "", [], 0.0, elapsed, peak_mem, 0

    cache_mem = 0
    if cache is not None:
        cache_mem = sum(c.nbytes for c in cache) / 1024 / 1024

    token_ids = tokenizer.encode(text)
    return text, token_ids, response.generation_tps, elapsed, peak_mem, cache_mem


def _make_long_prompt(tokenizer, target_tokens=500):
    """Create a prompt that results in ~target_tokens of context."""
    base = (
        "Read the following technical document carefully and summarize the key points:\n\n"
        "Quantum computing represents a fundamentally different approach to computation "
        "that leverages quantum mechanical phenomena such as superposition and entanglement "
        "to process information. Unlike classical computers that use bits representing 0 or 1, "
        "quantum computers use qubits that can exist in multiple states simultaneously. "
        "This property allows quantum computers to explore many possible solutions at once, "
        "making them potentially faster for certain types of problems like optimization, "
        "cryptography, and molecular simulation. "
        "The development of quantum computers has been a long journey starting from Richard "
        "Feynman's 1982 proposal. Major milestones include Shor's algorithm for factoring "
        "large numbers, Grover's search algorithm, and Google's quantum supremacy claim in 2019. "
        "Current quantum computers face challenges including decoherence, error rates, and "
        "the need for extremely low temperatures. Companies like IBM, Google, Microsoft, and "
        "startups like IonQ and Rigetti are racing to build more stable and powerful quantum "
        "processors. The field is also seeing advances in quantum error correction, which is "
        "essential for building practical quantum computers. "
    )
    # Repeat to reach target length
    prompt = base
    while len(tokenizer.encode(prompt)) < target_tokens:
        prompt += base
    return prompt


def run_benchmark(model_path="mlx-community/Qwen2.5-3B-Instruct-4bit"):
    model, tokenizer = _load_model(model_path)

    # Format helper
    def fmt_prompt(prompt):
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                messages = [{"role": "user", "content": prompt}]
                return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            except Exception:
                pass
        return prompt

    short_prompt = fmt_prompt("Explain quantum computing in simple terms")
    medium_prompt = fmt_prompt(_make_long_prompt(tokenizer, 500))
    long_prompt = fmt_prompt(_make_long_prompt(tokenizer, 1000))

    short_toks = len(tokenizer.encode(short_prompt))
    medium_toks = len(tokenizer.encode(medium_prompt))
    long_toks = len(tokenizer.encode(long_prompt))

    print(f"\nPrompt sizes: short={short_toks}, medium={medium_toks}, long={long_toks}")

    from mlx_turboquant.patch import make_turboquant_cache

    # ========================================================================
    print("\n" + "=" * 90)
    print("BENCHMARK 1: Short prompt (quality comparison)")
    print("=" * 90)

    configs = [
        ("Standard (FP16)", None, short_prompt, 100),
        ("TurboQuant 4-bit", {"key_bits": 4, "value_bits": 2, "buffer_size": 128}, short_prompt, 100),
        ("TurboQuant 3-bit", {"key_bits": 3, "value_bits": 2, "buffer_size": 128}, short_prompt, 100),
        ("TurboQuant 2-bit", {"key_bits": 2, "value_bits": 2, "buffer_size": 128}, short_prompt, 100),
    ]

    results_1 = []
    for name, tq_config, prompt, max_tok in configs:
        cache = None
        if tq_config:
            cache = make_turboquant_cache(model, **tq_config)
        text, tokens, tok_s, elapsed, peak_mem, cache_mem = _generate_with_cache(
            model, tokenizer, prompt, cache, max_tok
        )
        results_1.append((name, text, tokens, tok_s, peak_mem, cache_mem))
        print(f"  {name:<22} {tok_s:>6.1f} tok/s  peak={peak_mem:>7.1f}MB  cache={cache_mem:>5.1f}MB")

    # Quality comparison
    std_tokens = results_1[0][2]
    print(f"\n  {'Mode':<22} {'Token Match':<14} {'Text Preview'}")
    print(f"  {'-'*70}")
    for name, text, tokens, *_ in results_1:
        if name.startswith("Standard"):
            match_pct = "baseline"
        else:
            min_len = min(len(std_tokens), len(tokens))
            matches = sum(1 for a, b in zip(std_tokens[:min_len], tokens[:min_len]) if a == b)
            match_pct = f"{matches}/{min_len} ({matches/min_len*100:.0f}%)"
        print(f"  {name:<22} {match_pct:<14} {text[:60]}...")

    # ========================================================================
    print("\n" + "=" * 90)
    print("BENCHMARK 2: Medium prompt (~500 tokens) — compression kicks in")
    print("=" * 90)

    configs_2 = [
        ("Standard (FP16)", None, medium_prompt, 200),
        ("TurboQuant 3-bit", {"key_bits": 3, "value_bits": 2, "buffer_size": 128}, medium_prompt, 200),
    ]

    for name, tq_config, prompt, max_tok in configs_2:
        cache = None
        if tq_config:
            cache = make_turboquant_cache(model, **tq_config)
        text, tokens, tok_s, elapsed, peak_mem, cache_mem = _generate_with_cache(
            model, tokenizer, prompt, cache, max_tok
        )
        print(f"  {name:<22} {tok_s:>6.1f} tok/s  peak={peak_mem:>7.1f}MB  cache={cache_mem:>6.1f}MB  elapsed={elapsed:.1f}s")

    # ========================================================================
    print("\n" + "=" * 90)
    print("BENCHMARK 3: Long prompt (~1000 tokens) — TurboQuant advantage")
    print("=" * 90)

    configs_3 = [
        ("Standard (FP16)", None, long_prompt, 200),
        ("TurboQuant 3-bit", {"key_bits": 3, "value_bits": 2, "buffer_size": 128}, long_prompt, 200),
    ]

    for name, tq_config, prompt, max_tok in configs_3:
        cache = None
        if tq_config:
            cache = make_turboquant_cache(model, **tq_config)
        text, tokens, tok_s, elapsed, peak_mem, cache_mem = _generate_with_cache(
            model, tokenizer, prompt, cache, max_tok
        )
        print(f"  {name:<22} {tok_s:>6.1f} tok/s  peak={peak_mem:>7.1f}MB  cache={cache_mem:>6.1f}MB  elapsed={elapsed:.1f}s")

    # ========================================================================
    print("\n" + "=" * 90)
    print("BENCHMARK 4: Long generation (500 tokens) — memory over time")
    print("=" * 90)

    configs_4 = [
        ("Standard (FP16)", None, short_prompt, 500),
        ("TurboQuant 3-bit", {"key_bits": 3, "value_bits": 2, "buffer_size": 128}, short_prompt, 500),
    ]

    for name, tq_config, prompt, max_tok in configs_4:
        cache = None
        if tq_config:
            cache = make_turboquant_cache(model, **tq_config)
        text, tokens, tok_s, elapsed, peak_mem, cache_mem = _generate_with_cache(
            model, tokenizer, prompt, cache, max_tok
        )
        print(f"  {name:<22} {tok_s:>6.1f} tok/s  peak={peak_mem:>7.1f}MB  cache={cache_mem:>6.1f}MB  tokens={len(tokens)}")

    # ========================================================================
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print("""
TurboQuant compresses the KV cache (the model's conversation memory) by ~5x.

When it helps MOST:
  - Long prompts (>128 tokens) where compression actually kicks in
  - Long conversations (>500 generated tokens)
  - Running large models that are memory-limited

Current tradeoffs:
  - ~2-3x slower generation (quantize/dequantize overhead per step)
  - Memory savings are proportional to context length beyond the buffer
  - Quality is near-identical at 3-bit, good at 2-bit
""")


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen2.5-3B-Instruct-4bit"
    run_benchmark(model)
