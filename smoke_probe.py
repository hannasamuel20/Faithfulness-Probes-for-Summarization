"""
Smoke test: train a logistic regression probe on the features extracted in
step 1 to confirm they carry faithfulness signal.

Matches the Lookback-Lens paper setup:
  - Mean-pool each (L, H, T) tensor over T → 1024-dim per example
  - Logistic regression, no head selection
  - AUROC on the provided val/test splits

Also fits per-family probes (lookback only, attn_entropy only, logits only)
so you can see which features actually carry signal.

Usage:
    python smoke_probe.py --features /path/to/features_aggrefact_final.pt
"""

import argparse
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def featurize(ex):
    """(L,H,T) → mean over T → flatten → concat with scalar logit feats."""
    lb = ex["lookback_ratio"].mean(dim=-1).flatten().numpy()      # (L*H,)
    ae = ex["attn_entropy"].mean(dim=-1).flatten().numpy()        # (L*H,)
    lg = np.array([
        ex["logit_chosen_prob"].mean().item(),
        ex["logit_output_entropy"].mean().item(),
        ex["logit_top_margin"].mean().item(),
    ])
    return np.concatenate([lb, ae, lg])


def fit_and_score(Xtr, ytr, Xte, yte, name, C=1.0):
    clf = LogisticRegression(max_iter=2000, C=C).fit(Xtr, ytr)
    tr = roc_auc_score(ytr, clf.decision_function(Xtr))
    te = roc_auc_score(yte, clf.decision_function(Xte))
    print(f"  {name:22s}  train AUROC={tr:.4f}   test AUROC={te:.4f}")
    return clf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--C", type=float, default=1.0,
                        help="Inverse L2 strength for logistic regression")
    args = parser.parse_args()

    print(f"Loading {args.features} ...")
    blob = torch.load(args.features, weights_only=False)
    examples = blob["examples"] if isinstance(blob, dict) else blob
    print(f"  Loaded {len(examples)} examples.")

    # Drop examples without labels or split
    examples = [e for e in examples if "label" in e and "split" in e]

    X = np.stack([featurize(e) for e in examples])
    y = np.array([e["label"] for e in examples])
    splits = np.array([e["split"] for e in examples])

    L, H, _ = examples[0]["lookback_ratio"].shape
    n_lh = L * H
    sl_lookback = slice(0, n_lh)
    sl_attn_ent = slice(n_lh, 2 * n_lh)
    sl_logits = slice(2 * n_lh, 2 * n_lh + 3)

    train_mask = splits == "val"      # AggreFact convention: probes train on val
    test_mask = splits == "test"

    Xtr, ytr = X[train_mask], y[train_mask]
    Xte, yte = X[test_mask], y[test_mask]

    print(f"\n  Train: {len(ytr)} examples  ({(ytr==1).sum()} faithful, {(ytr==0).sum()} hallu)")
    print(f"  Test : {len(yte)} examples  ({(yte==1).sum()} faithful, {(yte==0).sum()} hallu)")
    print(f"  Feature dim: {X.shape[1]}  (lookback={n_lh}, attn_ent={n_lh}, logits=3)\n")

    print("═" * 70)
    print("  Per-family probes")
    print("═" * 70)
    fit_and_score(Xtr[:, sl_lookback], ytr, Xte[:, sl_lookback], yte, "lookback_ratio only", args.C)
    fit_and_score(Xtr[:, sl_attn_ent], ytr, Xte[:, sl_attn_ent], yte, "attn_entropy only",   args.C)
    fit_and_score(Xtr[:, sl_logits],   ytr, Xte[:, sl_logits],   yte, "logit feats only",    args.C)

    print("\n" + "═" * 70)
    print("  Combined probes")
    print("═" * 70)
    fit_and_score(
        np.concatenate([Xtr[:, sl_lookback], Xtr[:, sl_attn_ent]], axis=1), ytr,
        np.concatenate([Xte[:, sl_lookback], Xte[:, sl_attn_ent]], axis=1), yte,
        "lookback + attn_ent", args.C,
    )
    clf_all = fit_and_score(Xtr, ytr, Xte, yte, "ALL features", args.C)

    # Top-10 most important heads in the all-features probe (by |coef|)
    print("\n" + "═" * 70)
    print("  Top-10 most informative features (|logistic regression coef|)")
    print("═" * 70)
    coefs = np.abs(clf_all.coef_[0])
    names = (
        [f"lookback-L{l}-H{h}" for l in range(L) for h in range(H)] +
        [f"attnent-L{l}-H{h}"  for l in range(L) for h in range(H)] +
        ["chosen_prob", "output_entropy", "top_margin"]
    )
    top = np.argsort(-coefs)[:10]
    for rank, idx in enumerate(top, 1):
        print(f"  {rank:2d}. {names[idx]:25s}  coef={clf_all.coef_[0][idx]:+.4f}")

    # Sanity benchmark: on AggreFact-SOTA val→test, the paper reports
    # combined probes around AUROC 0.65–0.75 depending on feature set.
    print()
    print("═" * 70)
    print("  Interpretation")
    print("═" * 70)
    print("  - ALL features test AUROC > 0.70 : step-1 extraction is solid.")
    print("  - lookback-only AUROC > 0.55    : per-head signal exists even")
    print("                                     though the grand mean is noisy.")
    print("  - logit-only AUROC is usually the strongest single family")
    print("    (chosen_prob is a very strong signal under teacher forcing).")


if __name__ == "__main__":
    main()
