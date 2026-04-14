"""
Sanity-check a features .pt file produced by step01_extract_features.py.

Prints:
  - Basic structure (num examples, keys, shapes)
  - Label distribution (faithful vs hallucinated)
  - Per-dataset / per-source-model breakdowns
  - Feature statistics: mean / std of lookback ratio, attention entropy,
    and all three logit features
  - NaN/Inf checks
  - The critical sanity test: do hallucinated examples show LOWER lookback
    ratio and LOWER chosen_prob than faithful ones?  If not, something is
    wrong with feature extraction or the prompt format.
  - A sample example printed in full

Usage:
    python inspect_features.py --features features_aggrefact_sota.pt
    python inspect_features.py --features features_aggrefact_sota.pt --sample-idx 5
"""

import argparse
import torch
from collections import Counter


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def mean_over_tokens(tensor):
    """Collapse the token-time dimension: (L, H, T) -> (L, H) or (T,) -> scalar."""
    if tensor.ndim == 3:
        return tensor.mean(dim=-1)            # (L, H)
    return tensor.mean().item()                 # scalar


def example_summary_stats(ex):
    """Return a dict of scalar summaries for one example."""
    return {
        "lookback_ratio_mean": ex["lookback_ratio"].mean().item(),
        "attn_entropy_mean": ex["attn_entropy"].mean().item(),
        "chosen_prob_mean": ex["logit_chosen_prob"].mean().item(),
        "output_entropy_mean": ex["logit_output_entropy"].mean().item(),
        "top_margin_mean": ex["logit_top_margin"].mean().item(),
        "num_tokens": ex["lookback_ratio"].shape[-1],
    }


def fmt(x, n=4):
    if isinstance(x, float):
        return f"{x:.{n}f}"
    return str(x)


def section(title):
    print()
    print("═" * 70)
    print(f"  {title}")
    print("═" * 70)


# ═══════════════════════════════════════════════════════════════════════
#  Checks
# ═══════════════════════════════════════════════════════════════════════

def check_structure(results):
    section("1. Basic Structure")

    if not results:
        print("  ❌  Empty file — no examples found.")
        return False

    print(f"  Num examples: {len(results)}")
    r0 = results[0]
    print(f"  Keys in first example: {sorted(r0.keys())}")

    required = [
        "lookback_ratio", "attn_entropy",
        "logit_chosen_prob", "logit_output_entropy", "logit_top_margin",
        "data_index", "context_length",
    ]
    missing = [k for k in required if k not in r0]
    if missing:
        print(f"  ❌  Missing required keys: {missing}")
        return False

    L, H, T = r0["lookback_ratio"].shape
    print(f"  First example tensor shapes:")
    print(f"    lookback_ratio       : ({L}, {H}, {T})  [layers, heads, tokens]")
    print(f"    attn_entropy         : {tuple(r0['attn_entropy'].shape)}")
    print(f"    logit_chosen_prob    : {tuple(r0['logit_chosen_prob'].shape)}")
    print(f"    logit_output_entropy : {tuple(r0['logit_output_entropy'].shape)}")
    print(f"    logit_top_margin     : {tuple(r0['logit_top_margin'].shape)}")

    if (L, H) != (32, 32):
        print(f"  ⚠️   Expected (32, 32) for LLaMA-2-7B — got ({L}, {H})")
    else:
        print(f"  ✅  Dimensions match LLaMA-2-7B (32 layers, 32 heads)")

    # Consistency across examples
    all_shapes_ok = all(
        r["lookback_ratio"].shape[:2] == (L, H) for r in results
    )
    if all_shapes_ok:
        print(f"  ✅  All {len(results)} examples have consistent (L, H) dimensions")
    else:
        print(f"  ❌  Shape mismatch across examples!")

    return True


def check_nans(results):
    section("2. NaN / Inf Checks")

    tensor_keys = [
        "lookback_ratio", "attn_entropy",
        "logit_chosen_prob", "logit_output_entropy", "logit_top_margin",
    ]
    any_bad = False
    for key in tensor_keys:
        n_nan = 0
        n_inf = 0
        for r in results:
            t = r[key]
            n_nan += torch.isnan(t).sum().item()
            n_inf += torch.isinf(t).sum().item()
        status = "✅" if (n_nan == 0 and n_inf == 0) else "❌"
        print(f"  {status}  {key:22s}  NaN={n_nan}  Inf={n_inf}")
        if n_nan > 0 or n_inf > 0:
            any_bad = True

    if any_bad:
        print("  ⚠️   Non-finite values found — check the 1e-10 clamps in features.py")


def check_labels(results):
    section("3. Label Distribution")

    labels = [r.get("label") for r in results if "label" in r]
    if not labels:
        print("  ⚠️   No labels found on any example.")
        return None

    c = Counter(labels)
    total = sum(c.values())
    print(f"  Total labeled: {total}/{len(results)}")
    for lbl, n in sorted(c.items()):
        name = "faithful" if lbl == 1 else ("hallucinated" if lbl == 0 else f"label={lbl}")
        print(f"    {name:15s} (label={lbl}): {n:5d}  ({100*n/total:.1f}%)")

    return labels


def check_metadata_breakdown(results):
    section("4. Source Dataset / Model Breakdown")

    datasets = Counter(r.get("source_dataset", "?") for r in results)
    models = Counter(r.get("source_model", "?") for r in results)
    splits = Counter(r.get("split", "?") for r in results)

    print(f"  Source datasets:")
    for name, n in datasets.most_common():
        print(f"    {name:15s}: {n}")
    print(f"  Source models:")
    for name, n in models.most_common():
        print(f"    {name:20s}: {n}")
    print(f"  Splits:")
    for name, n in splits.most_common():
        print(f"    {name:15s}: {n}")


def check_token_lengths(results):
    section("5. Token Length Distribution")

    lengths = torch.tensor([r["lookback_ratio"].shape[-1] for r in results]).float()
    ctx_lens = torch.tensor([r["context_length"] for r in results]).float()

    print(f"  Summary token length (T):")
    print(f"    min={int(lengths.min())}  max={int(lengths.max())}  "
          f"mean={lengths.mean():.1f}  median={int(lengths.median())}")
    print(f"  Context length (prompt before #Summary#:):")
    print(f"    min={int(ctx_lens.min())}  max={int(ctx_lens.max())}  "
          f"mean={ctx_lens.mean():.1f}  median={int(ctx_lens.median())}")


def check_feature_stats(results):
    section("6. Global Feature Statistics")

    stats = [example_summary_stats(r) for r in results]
    keys = ["lookback_ratio_mean", "attn_entropy_mean",
            "chosen_prob_mean", "output_entropy_mean", "top_margin_mean"]

    print(f"  {'Feature':25s}  {'mean':>10s}  {'std':>10s}  {'min':>10s}  {'max':>10s}")
    print(f"  {'-'*25}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for k in keys:
        vals = torch.tensor([s[k] for s in stats])
        print(f"  {k:25s}  {vals.mean():>10.4f}  {vals.std():>10.4f}  "
              f"{vals.min():>10.4f}  {vals.max():>10.4f}")


def check_faithful_vs_hallucinated(results):
    """
    THE most important sanity check.

    If feature extraction is correct, hallucinated summaries (label=0)
    should on average show:
      - LOWER  lookback ratio  (less attention on context)
      - LOWER  chosen_prob      (model surprised by forced token)
      - HIGHER output_entropy   (model uncertain)
      - LOWER  top_margin       (less confident)

    If the direction is reversed or the gap is zero, something is wrong.
    """
    section("7. Faithful vs Hallucinated Feature Comparison  ⭐ KEY SANITY CHECK")

    faith = [r for r in results if r.get("label") == 1]
    hallu = [r for r in results if r.get("label") == 0]

    if not faith or not hallu:
        print("  ⚠️   Need both labels to compare. Found: "
              f"{len(faith)} faithful, {len(hallu)} hallucinated.")
        return

    print(f"  Comparing {len(faith)} faithful vs {len(hallu)} hallucinated examples.")
    print()

    def mean_of(examples, key):
        return torch.tensor([
            example_summary_stats(e)[key] for e in examples
        ]).mean().item()

    keys = [
        ("lookback_ratio_mean", "higher for faithful"),
        ("chosen_prob_mean",    "higher for faithful"),
        ("top_margin_mean",     "higher for faithful"),
        ("output_entropy_mean", "lower for faithful"),
        ("attn_entropy_mean",   "lower for faithful"),
    ]

    print(f"  {'Feature':25s}  {'faithful':>12s}  {'hallu':>12s}  "
          f"{'gap':>10s}  {'expected':>22s}  {'status'}")
    print(f"  {'-'*25}  {'-'*12}  {'-'*12}  {'-'*10}  {'-'*22}  {'-'*6}")

    for k, expectation in keys:
        f_mean = mean_of(faith, k)
        h_mean = mean_of(hallu, k)
        gap = f_mean - h_mean

        if "higher for faithful" in expectation:
            correct = gap > 0
        else:
            correct = gap < 0

        status = "✅" if correct else "❌"
        print(f"  {k:25s}  {f_mean:>12.4f}  {h_mean:>12.4f}  "
              f"{gap:>+10.4f}  {expectation:>22s}  {status}")

    print()
    print("  If any feature has ❌, the signal direction is wrong — either")
    print("  feature extraction is buggy, or that feature is not useful for")
    print("  this dataset.  Small gaps (near 0) mean weak individual signal")
    print("  — this is normal and why we train a classifier over many heads.")


def print_sample(results, idx):
    section(f"8. Sample Example (index {idx})")

    if idx >= len(results):
        print(f"  Index {idx} out of range (only {len(results)} examples)")
        return

    r = results[idx]
    print(f"  data_index      : {r.get('data_index')}")
    print(f"  label           : {r.get('label', 'N/A')}  "
          f"({'faithful' if r.get('label') == 1 else 'hallucinated' if r.get('label') == 0 else '?'})")
    print(f"  source_dataset  : {r.get('source_dataset', 'N/A')}")
    print(f"  source_model    : {r.get('source_model', 'N/A')}")
    print(f"  split           : {r.get('split', 'N/A')}")
    print(f"  context_length  : {r.get('context_length')}")
    print(f"  summary text    : {str(r.get('summary_text', ''))[:200]}...")

    print(f"\n  Per-token feature stats (averaged over L, H for attention):")
    print(f"    lookback_ratio       : mean={r['lookback_ratio'].mean():.4f}")
    print(f"    attn_entropy         : mean={r['attn_entropy'].mean():.4f}")
    print(f"    logit_chosen_prob    : mean={r['logit_chosen_prob'].mean():.4f}  "
          f"min={r['logit_chosen_prob'].min():.4f}")
    print(f"    logit_output_entropy : mean={r['logit_output_entropy'].mean():.4f}")
    print(f"    logit_top_margin     : mean={r['logit_top_margin'].mean():.4f}")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Inspect a features .pt file.")
    parser.add_argument("--features", type=str, required=True,
                        help="Path to the .pt file from step01")
    parser.add_argument("--sample-idx", type=int, default=0,
                        help="Index of the sample example to print in detail")
    args = parser.parse_args()

    print(f"Loading {args.features} ...")
    blob = torch.load(args.features, weights_only=False)
    if isinstance(blob, dict) and "examples" in blob:
        config = blob.get("config")
        results = blob["examples"]
        if config:
            print(f"  Config: model={config.get('model_name')}, "
                  f"teacher_forcing={config.get('teacher_forcing')}, "
                  f"split={config.get('split')}, "
                  f"max_doc_tokens={config.get('max_doc_tokens')}, "
                  f"created_at={config.get('created_at')}")
    else:
        # Legacy list format
        results = blob
    print(f"  Loaded {len(results)} examples.")

    if not check_structure(results):
        return
    check_nans(results)
    check_labels(results)
    check_metadata_breakdown(results)
    check_token_lengths(results)
    check_feature_stats(results)
    check_faithful_vs_hallucinated(results)
    print_sample(results, args.sample_idx)

    print()
    print("═" * 70)
    print("  Done.")
    print("═" * 70)


if __name__ == "__main__":
    main()
