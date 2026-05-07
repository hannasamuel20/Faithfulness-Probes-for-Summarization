"""
Step 7: Layer/head analysis of the ALL static probe.

Trains the `all` logistic-regression probe, extracts |coef| per feature
dimension, maps coefficients back to (layer, head) pairs, and produces:

  1. A heatmap of summed |coef| over aggregation types for each (l, h) pair,
     split by feature family (lookback vs attention entropy).
  2. A layer-marginal bar chart (sum over heads) to test Chuang et al.'s
     late-layer bias claim.
  3. A CSV of the top-K features ranked by |coef|.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step02_static_probes import (
    build_XY,
    fit_lr_probe,
    load_examples,
    split_examples,
)


AGGS = ("mean", "min", "max", "var")


def coef_to_lh_map(coefs, L, H):
    """
    The `all` feature vector is ordered as:
        lookback_full  : 4 aggs × L×H  (4*L*H dims)
        attn_entropy   : 4 aggs × L×H  (4*L*H dims)
        logits_full    : 3 streams × 4 aggs  (12 dims)

    Returns:
        records  — list of dicts with (family, agg, layer, head, coef, abs_coef)
        logit_records — list of dicts for the 12 logit dims
    """
    LH = L * H
    block_size = len(AGGS) * LH

    families = ["lookback", "attn_entropy"]
    records = []
    for fam_idx, family in enumerate(families):
        offset = fam_idx * block_size
        for agg_idx, agg in enumerate(AGGS):
            for l in range(L):
                for h in range(H):
                    dim = offset + agg_idx * LH + l * H + h
                    records.append({
                        "family": family,
                        "agg": agg,
                        "layer": l,
                        "head": h,
                        "coef": float(coefs[dim]),
                        "abs_coef": float(abs(coefs[dim])),
                    })

    logit_offset = 2 * block_size
    logit_streams = ("chosen_prob", "output_entropy", "top_margin")
    logit_records = []
    for s_idx, stream in enumerate(logit_streams):
        for a_idx, agg in enumerate(AGGS):
            dim = logit_offset + s_idx * len(AGGS) + a_idx
            logit_records.append({
                "family": "logit",
                "stream": stream,
                "agg": agg,
                "coef": float(coefs[dim]),
                "abs_coef": float(abs(coefs[dim])),
            })

    return records, logit_records


def build_heatmap(records, family, L, H):
    """Sum |coef| across all four aggregation types for each (l, h)."""
    grid = np.zeros((L, H))
    for r in records:
        if r["family"] == family:
            grid[r["layer"], r["head"]] += r["abs_coef"]
    return grid


def build_layer_marginal(records, L):
    """Sum |coef| over all heads and aggs per layer, both families combined."""
    per_layer = np.zeros(L)
    for r in records:
        per_layer[r["layer"]] += r["abs_coef"]
    return per_layer


def plot_heatmaps(lb_grid, ent_grid, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, grid, title in zip(
        axes,
        [lb_grid, ent_grid],
        ["Lookback ratio", "Attention entropy"],
    ):
        im = ax.imshow(grid, aspect="auto", cmap="YlOrRd")
        ax.set_title(f"|coef| by (layer, head) — {title}", fontsize=11)
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("ALL static probe: feature importance per (layer, head)", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "layer_head_heatmap.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "layer_head_heatmap.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_layer_marginal(per_layer, out_dir):
    L = len(per_layer)
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(L), per_layer, color="#2f6f9f", edgecolor="white", linewidth=0.3)

    half = L // 2
    late_sum = per_layer[half:].sum()
    early_sum = per_layer[:half].sum()
    ratio = late_sum / early_sum if early_sum > 0 else float("inf")

    ax.axvline(half - 0.5, color="red", linestyle="--", linewidth=1, label=f"Layer {half} boundary")
    ax.set_title(
        f"Layer-marginal |coef| (late/early ratio = {ratio:.2f})",
        fontsize=11,
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Sum |coef| over all heads & aggs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "layer_marginal.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "layer_marginal.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_layer_marginal_by_family(records, L, out_dir):
    lb_layer = np.zeros(L)
    ent_layer = np.zeros(L)
    for r in records:
        if r["family"] == "lookback":
            lb_layer[r["layer"]] += r["abs_coef"]
        elif r["family"] == "attn_entropy":
            ent_layer[r["layer"]] += r["abs_coef"]

    x = np.arange(L)
    width = 0.4
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, lb_layer, width, label="Lookback ratio", color="#2f6f9f")
    ax.bar(x + width / 2, ent_layer, width, label="Attention entropy", color="#e07b39")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Sum |coef| over heads & aggs")
    ax.set_title("Layer-marginal |coef| by feature family", fontsize=11)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "layer_marginal_by_family.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "layer_marginal_by_family.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_top_features_csv(records, logit_records, out_dir, top_k=50):
    all_records = []
    for r in records:
        all_records.append({
            "feature": f"{r['family']}/{r['agg']}/L{r['layer']}-H{r['head']}",
            "family": r["family"],
            "agg": r["agg"],
            "layer": r["layer"],
            "head": r["head"],
            "coef": r["coef"],
            "abs_coef": r["abs_coef"],
        })
    for r in logit_records:
        all_records.append({
            "feature": f"logit/{r['stream']}/{r['agg']}",
            "family": "logit",
            "agg": r["agg"],
            "layer": -1,
            "head": -1,
            "coef": r["coef"],
            "abs_coef": r["abs_coef"],
        })

    df = pd.DataFrame(all_records).sort_values("abs_coef", ascending=False)
    df.to_csv(os.path.join(out_dir, "all_features_ranked.csv"), index=False)
    df.head(top_k).to_csv(os.path.join(out_dir, f"top_{top_k}_features.csv"), index=False)


def parse_args():
    p = argparse.ArgumentParser(description="Step 7: layer/head analysis of ALL static probe.")
    p.add_argument("--features", default="/projectnb/cs505am/projects/faithfulness_probe/features_aggrefact_sota.pt")
    p.add_argument("--out-dir", default="/projectnb/cs505am/projects/faithfulness_probe/results/layer_analysis")
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--train-split", default="val")
    p.add_argument("--test-split", default="test")
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    examples = load_examples(args.features)
    train_ex, test_ex = split_examples(examples, args.train_split, args.test_split)

    X_tr, y_tr = build_XY(train_ex, "all")
    clf, scaler = fit_lr_probe(X_tr, y_tr, C=args.C, seed=args.seed)
    coefs = clf.coef_[0]

    L, H, _ = train_ex[0]["lookback_ratio"].shape
    print(f"Model geometry: L={L}, H={H}")
    print(f"Feature vector dim: {len(coefs)} (expected {2*4*L*H + 12})")

    records, logit_records = coef_to_lh_map(coefs, L, H)

    lb_grid = build_heatmap(records, "lookback", L, H)
    ent_grid = build_heatmap(records, "attn_entropy", L, H)
    per_layer = build_layer_marginal(records, L)

    plot_heatmaps(lb_grid, ent_grid, args.out_dir)
    plot_layer_marginal(per_layer, args.out_dir)
    plot_layer_marginal_by_family(records, L, args.out_dir)
    save_top_features_csv(records, logit_records, args.out_dir, top_k=args.top_k)

    half = L // 2
    late_sum = per_layer[half:].sum()
    early_sum = per_layer[:half].sum()
    print(f"\nLate-layer bias check (Chuang et al. 2024):")
    print(f"  Early layers (0-{half-1}) total |coef|: {early_sum:.4f}")
    print(f"  Late layers ({half}-{L-1}) total |coef|:  {late_sum:.4f}")
    print(f"  Late/early ratio: {late_sum/early_sum:.2f}" if early_sum > 0 else "  Early sum is 0")

    print(f"\nTop-10 features by |coef|:")
    ranked = sorted(records + logit_records, key=lambda r: r["abs_coef"], reverse=True)
    for i, r in enumerate(ranked[:10], 1):
        if "layer" in r:
            name = f"{r['family']}/{r['agg']}/L{r['layer']}-H{r['head']}"
        else:
            name = f"logit/{r['stream']}/{r['agg']}"
        print(f"  {i:2d}. {name:40s}  coef={r['coef']:+.4f}")

    print(f"\nSaved results -> {args.out_dir}")


if __name__ == "__main__":
    main()
