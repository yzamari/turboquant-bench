"""
Format and display comparison results.
"""

from tq_bench.compare import BenchmarkResult


def print_report(results: list[BenchmarkResult], prompt: str) -> None:
    """Print a formatted comparison report to stdout."""

    print()
    print("=" * 80)
    print("TURBOQUANT BENCHMARK REPORT")
    print("=" * 80)
    prompt_display = prompt[:80] + ("..." if len(prompt) > 80 else "")
    print(f"Prompt: {prompt_display}")
    print()

    # --- Performance table ---
    header = f"{'Mode':<20} {'Tokens':<8} {'Gen tok/s':<12} {'Peak Mem MB':<14}"
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r.mode:<20} {r.num_tokens:<8} "
            f"{r.gen_tok_s:<12.1f} {r.peak_mem_mb:<14.1f}"
        )

    # --- Quality comparison ---
    standard = next((r for r in results if r.mode == "standard"), None)

    if standard and len(standard.tokens) > 0:
        print()
        print("Quality vs Standard:")
        for r in results:
            if r.mode == "standard":
                continue
            _print_quality_comparison(standard, r)

    # --- Memory savings ---
    _print_memory_comparison(results)

    # --- Output text ---
    print()
    print("=" * 80)
    print("GENERATED TEXT COMPARISON")
    print("=" * 80)
    for r in results:
        print(f"\n--- {r.mode} ({r.num_tokens} tokens) ---")
        print(r.text[:500])

    print()


def _print_quality_comparison(
    standard: BenchmarkResult, other: BenchmarkResult
) -> None:
    """Print token match rate between standard and another mode."""
    min_len = min(len(standard.tokens), len(other.tokens))
    if min_len == 0:
        print(f"  {other.mode}: no tokens to compare")
        return

    matches = sum(
        1
        for a, b in zip(standard.tokens[:min_len], other.tokens[:min_len])
        if a == b
    )
    match_rate = matches / min_len * 100

    # Character-level similarity on the text
    min_chars = min(len(standard.text), len(other.text))
    if min_chars > 0:
        char_matches = sum(
            1
            for a, b in zip(standard.text[:min_chars], other.text[:min_chars])
            if a == b
        )
        char_rate = char_matches / min_chars * 100
    else:
        char_rate = 0.0

    print(
        f"  {other.mode}: {match_rate:.1f}% token match ({matches}/{min_len}), "
        f"{char_rate:.1f}% char match"
    )


def _print_memory_comparison(results: list[BenchmarkResult]) -> None:
    """Print peak memory savings relative to standard mode."""
    standard = next((r for r in results if r.mode == "standard"), None)
    if standard is None or standard.peak_mem_mb <= 0:
        return

    others = [r for r in results if r.mode != "standard" and r.peak_mem_mb > 0]
    if not others:
        return

    print()
    print("Peak Memory Comparison:")
    for r in others:
        diff = standard.peak_mem_mb - r.peak_mem_mb
        pct = diff / standard.peak_mem_mb * 100 if standard.peak_mem_mb > 0 else 0
        direction = "saved" if diff > 0 else "more"
        print(f"  {r.mode}: {r.peak_mem_mb:.1f} MB ({abs(pct):.1f}% {direction})")
