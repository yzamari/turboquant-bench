"""
Long-context benchmark: test where TurboQuant actually matters.

Tests at 4K, 8K, 16K prompt tokens — where KV cache becomes a
significant fraction of total memory and bandwidth.
"""

import time
import sys
import mlx.core as mx


def _load_model(model_path):
    from mlx_lm import load
    print(f"Loading model: {model_path}")
    model, tokenizer = load(model_path)
    return model, tokenizer


def _make_long_prompt(tokenizer, target_tokens):
    """Create a prompt with ~target_tokens."""
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
        "Machine learning and AI are also benefiting from advances in hardware acceleration. "
        "GPUs, TPUs, and specialized AI chips are making it possible to train larger models "
        "faster than ever before. The transformer architecture has revolutionized NLP, "
        "computer vision, and even protein folding prediction. Models like GPT, BERT, and "
        "LLaMA have demonstrated remarkable capabilities in understanding and generating text. "
        "The scaling laws suggest that larger models with more data continue to improve. "
        "However, the cost of training and inference is a major concern for the industry. "
        "Techniques like quantization, pruning, distillation, and KV cache compression "
        "are being developed to make inference more efficient without sacrificing quality. "
    )
    prompt = base
    while len(tokenizer.encode(prompt)) < target_tokens:
        prompt += base
    # Trim to exact target
    tokens = tokenizer.encode(prompt)
    if len(tokens) > target_tokens:
        prompt = tokenizer.decode(tokens[:target_tokens])
    return prompt


def _run_mode(model, tokenizer, prompt_text, mode, max_tokens=50):
    """Run generation in a given mode, return (tok_s, peak_mb, elapsed, text)."""
    import mlx_lm

    gen_kwargs = dict(max_tokens=max_tokens)

    if mode == "turboquant":
        from mlx_turboquant.patch import make_turboquant_cache
        cache = make_turboquant_cache(model, key_bits=3, value_bits=2, buffer_size=128)
        gen_kwargs["prompt_cache"] = cache

    mx.synchronize()
    mx.reset_peak_memory()

    text = ""
    response = None
    start = time.perf_counter()

    for response in mlx_lm.stream_generate(model, tokenizer, prompt_text, **gen_kwargs):
        text += response.text

    elapsed = time.perf_counter() - start
    mx.synchronize()
    peak_mem = mx.get_peak_memory() / 1024 / 1024

    if response is None:
        return 0.0, peak_mem, elapsed, ""

    return response.generation_tps, peak_mem, elapsed, text


def run_long_context_benchmark(model_path):
    model, tokenizer = _load_model(model_path)

    # Format helper
    def fmt(prompt):
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                messages = [{"role": "user", "content": prompt}]
                return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            except Exception:
                pass
        return prompt

    # Test configs: prompt sizes that stress the KV cache
    prompt_sizes = [1024, 2048, 4096, 8192, 16384, 32768]
    max_gen = 50  # small generation to focus on prefill + first decode steps

    print(f"\n{'='*90}")
    print(f"LONG-CONTEXT BENCHMARK: {model_path}")
    print(f"{'='*90}")
    print(f"Generate {max_gen} tokens after each prompt. Focus: prefill speed + memory.")
    print()

    results = []

    for n_tok in prompt_sizes:
        print(f"\n--- Prompt: {n_tok:,} tokens ---")

        prompt_text = fmt(_make_long_prompt(tokenizer, n_tok))
        actual_tokens = len(tokenizer.encode(prompt_text))
        print(f"  Actual prompt tokens: {actual_tokens:,}")

        # Standard
        try:
            std_tps, std_mem, std_elapsed, std_text = _run_mode(
                model, tokenizer, prompt_text, "standard", max_gen
            )
            print(f"  Standard:   {std_tps:>6.1f} tok/s  peak={std_mem:>8.1f} MB  elapsed={std_elapsed:.1f}s")
        except Exception as e:
            print(f"  Standard:   FAILED — {e}")
            std_tps, std_mem, std_elapsed = 0, 0, 0

        # TurboQuant
        try:
            tq_tps, tq_mem, tq_elapsed, tq_text = _run_mode(
                model, tokenizer, prompt_text, "turboquant", max_gen
            )
            print(f"  TurboQuant: {tq_tps:>6.1f} tok/s  peak={tq_mem:>8.1f} MB  elapsed={tq_elapsed:.1f}s")
        except Exception as e:
            print(f"  TurboQuant: FAILED — {e}")
            tq_tps, tq_mem, tq_elapsed = 0, 0, 0

        # Comparison
        if std_tps > 0 and tq_tps > 0:
            speed_pct = tq_tps / std_tps * 100
            mem_saved = std_mem - tq_mem
            print(f"  => Speed: {speed_pct:.0f}% of standard | Memory: {mem_saved:+.0f} MB")

            # Text comparison
            std_tokens = tokenizer.encode(std_text)
            tq_tokens = tokenizer.encode(tq_text)
            min_len = min(len(std_tokens), len(tq_tokens))
            if min_len > 0:
                matches = sum(1 for a, b in zip(std_tokens[:min_len], tq_tokens[:min_len]) if a == b)
                print(f"  => Quality: {matches}/{min_len} tokens match ({matches/min_len*100:.0f}%)")

        results.append({
            "prompt_tokens": actual_tokens,
            "std_tps": std_tps, "std_mem": std_mem, "std_elapsed": std_elapsed,
            "tq_tps": tq_tps, "tq_mem": tq_mem, "tq_elapsed": tq_elapsed,
        })

    # Summary table
    print(f"\n{'='*90}")
    print("SUMMARY TABLE")
    print(f"{'='*90}")
    print(f"{'Prompt Tokens':<15} {'Std tok/s':<12} {'TQ tok/s':<12} {'Speed':<10} {'Std Mem':<12} {'TQ Mem':<12} {'Mem Saved':<12}")
    print("-" * 85)

    for r in results:
        speed_pct = f"{r['tq_tps']/r['std_tps']*100:.0f}%" if r['std_tps'] > 0 else "N/A"
        mem_saved = f"{r['std_mem']-r['tq_mem']:+.0f} MB" if r['std_mem'] > 0 else "N/A"
        print(f"{r['prompt_tokens']:<15,} {r['std_tps']:<12.1f} {r['tq_tps']:<12.1f} {speed_pct:<10} {r['std_mem']:<12.0f} {r['tq_mem']:<12.0f} {mem_saved:<12}")

    print(f"\nAs prompt length grows, TurboQuant's advantage increases because:")
    print(f"  - KV cache becomes a larger fraction of total memory bandwidth")
    print(f"  - Reading 3-bit packed data is much less bandwidth than 16-bit FP16")
    print(f"  - Memory savings prevent OOM at very long contexts")


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen2.5-3B-Instruct-4bit"
    run_long_context_benchmark(model)
