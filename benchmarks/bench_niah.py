"""
Needle-in-a-Haystack (NIAH) benchmark for TurboQuant.

Tests whether the model can retrieve a specific fact hidden in a long context,
with and without TurboQuant KV cache compression.

This is the community-standard quality test used by Prince Canuma and others.
"""
import time
import mlx.core as mx
import mlx_lm
from mlx_turboquant import patch_model

NEEDLE = "The secret password is 'rainbow-dolphin-42'."
QUESTION = "What is the secret password mentioned in the text?"
EXPECTED = "rainbow-dolphin-42"

FILLER = (
    "The history of artificial intelligence spans decades of research and development. "
    "Early pioneers like Alan Turing and John McCarthy laid the groundwork for what would "
    "become one of the most transformative technologies of the 21st century. Machine learning, "
    "a subset of AI, has seen remarkable progress with the advent of deep neural networks. "
    "Natural language processing has evolved from simple rule-based systems to sophisticated "
    "transformer architectures capable of understanding and generating human-like text. "
    "Computer vision has similarly advanced, with models now able to identify objects, faces, "
    "and scenes with superhuman accuracy. Reinforcement learning has produced agents that can "
    "master complex games and optimize real-world systems. The ethical implications of AI "
    "continue to be debated as these systems become more powerful and pervasive. "
)


def build_haystack(tokenizer, target_tokens: int) -> str:
    """Build a haystack of approximately target_tokens with a needle in the middle."""
    # Estimate tokens per filler block
    filler_tokens = len(tokenizer.encode(FILLER))
    blocks_needed = target_tokens // filler_tokens + 1

    half = blocks_needed // 2
    before = FILLER * half
    after = FILLER * (blocks_needed - half)

    haystack = before + "\n\n" + NEEDLE + "\n\n" + after
    return haystack


def run_niah(model, tokenizer, context_tokens: int, use_tq: bool, key_bits: int = 3):
    """Run a single NIAH test. Returns (passed, response_text, tok/s, peak_mb)."""
    haystack = build_haystack(tokenizer, context_tokens)
    prompt = f"{haystack}\n\nQuestion: {QUESTION}\nAnswer:"

    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    prompt_ids = tokenizer.encode(prompt_text)
    actual_tokens = len(prompt_ids)

    mx.reset_peak_memory()
    start = time.time()

    if use_tq:
        from mlx_turboquant import generate
        result = generate(
            model_path=None,  # model already loaded
            prompt=prompt_text,
            max_tokens=50,
            key_bits=key_bits,
            verbose=False,
            _model=model,
            _tokenizer=tokenizer,
        )
        text = result["text"] if isinstance(result, dict) else result
    else:
        response = mlx_lm.generate(
            model, tokenizer, prompt=prompt_text, max_tokens=50, verbose=False
        )
        text = response

    elapsed = time.time() - start
    peak_mb = mx.get_peak_memory() / 1024 / 1024

    passed = EXPECTED.lower() in text.lower()
    return passed, text, actual_tokens, elapsed, peak_mb


def main():
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen2.5-3B-Instruct-4bit"

    print(f"Loading model: {model_path}")
    model, tokenizer = mlx_lm.load(model_path)

    context_sizes = [2048, 4096, 8192]
    # Adjust for model capability
    if "32B" in model_path or "35B" in model_path:
        context_sizes = [4096, 8192, 16384, 32768]

    print(f"\n{'='*70}")
    print(f"NEEDLE-IN-A-HAYSTACK BENCHMARK: {model_path}")
    print(f"{'='*70}")
    print(f"Needle: {NEEDLE}")
    print(f"Expected answer contains: '{EXPECTED}'")
    print()

    results = []
    for ctx in context_sizes:
        print(f"--- Context: ~{ctx:,} tokens ---")

        # Standard
        mx.metal.clear_cache()
        passed_std, text_std, actual, elapsed_std, peak_std = run_niah(
            model, tokenizer, ctx, use_tq=False
        )
        print(f"  Standard:    {'PASS' if passed_std else 'FAIL'}  peak={peak_std:.0f}MB  {elapsed_std:.1f}s  ({actual:,} tok)")
        if not passed_std:
            print(f"    Response: {text_std[:100]}...")

        # TurboQuant 3-bit
        mx.metal.clear_cache()
        patch_model(model, key_bits=3)
        try:
            # Use generate directly since patch_model modifies make_cache
            from mlx_turboquant import generate as tq_gen
            mx.reset_peak_memory()
            start = time.time()
            r = tq_gen(
                model_path=model_path,
                prompt=f"{build_haystack(tokenizer, ctx)}\n\nQuestion: {QUESTION}\nAnswer:",
                max_tokens=50, key_bits=3, verbose=False,
            )
            elapsed_tq = time.time() - start
            peak_tq = mx.get_peak_memory() / 1024 / 1024
            text_tq = r["text"] if isinstance(r, dict) else r
            passed_tq = EXPECTED.lower() in text_tq.lower()
        except Exception as e:
            passed_tq, text_tq, elapsed_tq, peak_tq = False, str(e), 0, 0

        print(f"  TurboQuant:  {'PASS' if passed_tq else 'FAIL'}  peak={peak_tq:.0f}MB  {elapsed_tq:.1f}s")
        if not passed_tq:
            print(f"    Response: {text_tq[:100]}...")

        results.append({
            "context": ctx,
            "actual_tokens": actual,
            "std_pass": passed_std,
            "tq_pass": passed_tq,
            "std_peak": peak_std,
            "tq_peak": peak_tq,
        })
        print()

    # Summary
    print(f"{'='*70}")
    print("NIAH SUMMARY")
    print(f"{'='*70}")
    std_total = sum(1 for r in results if r["std_pass"])
    tq_total = sum(1 for r in results if r["tq_pass"])
    print(f"Standard:    {std_total}/{len(results)} passed")
    print(f"TurboQuant:  {tq_total}/{len(results)} passed")


if __name__ == "__main__":
    main()
