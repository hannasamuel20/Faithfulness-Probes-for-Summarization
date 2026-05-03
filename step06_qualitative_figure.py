"""
Step 6: Qualitative lookback-trace figure for the final report.

Selects one confident correct faithful example and one confident correct
hallucinated example from AggreFact test, then plots the per-token lookback
ratio averaged over all layers and heads.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from step03_temporal_probes import (
    CHANNEL_BUILDERS,
    SpanSequenceDataset,
    collate_pad,
    load_examples,
    make_model,
    score_loader,
    split_examples,
)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "state_dict" not in ckpt:
        raise ValueError("Re-run step03 with --save-model to create a metadata checkpoint.")
    model = make_model(
        ckpt["model"],
        ckpt["in_channels"],
        ckpt.get("hidden", 64),
        ckpt.get("dropout", 0.3),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return ckpt, model


def score_examples(model, examples, channels, max_T, batch_size, device):
    loader = torch.utils.data.DataLoader(
        SpanSequenceDataset(examples, CHANNEL_BUILDERS[channels], max_T=max_T),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_pad,
        num_workers=0,
    )
    return score_loader(model, loader, device)


def pick_examples(examples, scores, threshold):
    scores = np.asarray(scores)
    pred_faithful = scores >= threshold
    y = np.array([e["label"] for e in examples], dtype=int)

    faithful_candidates = [
        (sigmoid(scores[i]), i)
        for i in range(len(examples))
        if y[i] == 1 and pred_faithful[i]
    ]
    hallu_candidates = [
        (sigmoid(-scores[i]), i)
        for i in range(len(examples))
        if y[i] == 0 and not pred_faithful[i]
    ]
    if not faithful_candidates or not hallu_candidates:
        raise ValueError("Could not find both faithful and hallucinated confident correct examples.")
    return (
        max(faithful_candidates)[1],
        max(hallu_candidates)[1],
    )


def plot_trace(ax, ex, score, title):
    trace = ex["lookback_ratio"].mean(dim=(0, 1)).detach().cpu().numpy()
    p_hallu = sigmoid(-score)
    ax.plot(np.arange(len(trace)), trace, color="#2f6f9f", linewidth=1.5)
    ax.set_title(f"{title}  |  predicted P(hallucinated)={p_hallu:.2f}", fontsize=10)
    ax.set_ylabel("Lookback ratio")
    ax.set_xlim(0, max(len(trace) - 1, 1))
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.35)
    snippet = str(ex.get("summary_text", "")).replace("\n", " ")
    if len(snippet) > 130:
        snippet = snippet[:127] + "..."
    ax.text(
        0.01, 0.02, snippet,
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 2},
    )


def parse_args():
    p = argparse.ArgumentParser(description="Step 6: qualitative lookback trace figure.")
    p.add_argument("--features", default="results/features_aggrefact_sota.pt")
    p.add_argument("--model-path", default="results/cnn_lookback.pt")
    p.add_argument("--pdf", default="results/lookback_trace_figure.pdf")
    p.add_argument("--png", default="results/lookback_trace_figure.png")
    p.add_argument("--train-split", default="val")
    p.add_argument("--test-split", default="test")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    ckpt, model = load_checkpoint(args.model_path, device)
    examples = load_examples(args.features, require_split=True)
    _, test_ex = split_examples(examples, args.train_split, args.test_split)
    _, scores = score_examples(
        model,
        test_ex,
        ckpt["channels"],
        ckpt.get("max_T", 256),
        args.batch_size,
        device,
    )
    threshold = ckpt.get("threshold", 0.0)
    faithful_i, hallu_i = pick_examples(test_ex, scores, threshold)

    fig, axes = plt.subplots(2, 1, figsize=(6, 4), sharex=False)
    plot_trace(axes[0], test_ex[faithful_i], scores[faithful_i], "Faithful example")
    plot_trace(axes[1], test_ex[hallu_i], scores[hallu_i], "Hallucinated example")
    axes[1].set_xlabel("Summary token position")
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.pdf) or ".", exist_ok=True)
    fig.savefig(args.pdf, bbox_inches="tight")
    fig.savefig(args.png, dpi=200, bbox_inches="tight")
    print(f"Saved figure -> {args.pdf}")
    print(f"Saved figure -> {args.png}")


if __name__ == "__main__":
    main()
