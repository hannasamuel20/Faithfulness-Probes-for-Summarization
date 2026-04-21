"""
Step 2: Static (non-temporal) Faithfulness Probes
==================================================

Trains logistic regression probes on span-level feature aggregates extracted
in Step 1. Covers the proposal's baselines and the complementary-feature
ablation — but NOT the temporal CNN/LSTM probes (those live in Step 3).

AggreFact convention: probes train on the `cut=val` examples and are
evaluated on the held-out `cut=test` examples. Labels: 1 = faithful,
0 = hallucinated. "Faithful" is the positive class.

Feature sets (see utils/aggregation.FEATURE_SETS):

    Baselines
      random            — uniform-random scores (trivial ~0.50 AUROC floor)
      output_prob_only  — single scalar: mean chosen_prob over span
      lookback_lens     — Chuang et al. (2024): mean lookback ratio, L*H dims

    Single families (full aggregation: mean/min/max/var over T)
      lookback_full, attn_entropy_full, logits_full

    Combinations
      lookback+entropy, lookback+logits, entropy+logits, all

Usage:
    python step02_static_probes.py --features features_aggrefact_sota.pt

    # Tighter L2 + bootstrap CIs for the headline combined model:
    python step02_static_probes.py \\
        --features features_aggrefact_sota.pt \\
        --C 0.5 --bootstrap 1000 \\
        --results-path results/step02_static.json
"""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from utils.aggregation import FEATURE_SETS, build_feature_vector
from utils.evaluation import evaluate_scores, bootstrap_auroc, format_row


# ────────────────────────────────────────────────────────────────────────
#  Data loading
# ────────────────────────────────────────────────────────────────────────

def load_examples(path):
    blob = torch.load(path, weights_only=False)
    examples = blob["examples"] if isinstance(blob, dict) else blob
    # Drop anything we can't split or label.
    examples = [e for e in examples if "label" in e and "split" in e]
    return examples


def split_examples(examples, train_split="val", test_split="test"):
    """AggreFact convention: probes train on `val`, evaluate on `test`."""
    train = [e for e in examples if e["split"] == train_split]
    test = [e for e in examples if e["split"] == test_split]
    return train, test


def build_XY(examples, feature_set):
    X = np.stack([build_feature_vector(e, feature_set) for e in examples])
    y = np.array([e["label"] for e in examples], dtype=int)
    return X, y


# ────────────────────────────────────────────────────────────────────────
#  Probes
# ────────────────────────────────────────────────────────────────────────

def fit_lr_probe(X_tr, y_tr, C=1.0, standardize=True, seed=0):
    """
    Logistic regression probe. StandardScaler helps when feature blocks
    have wildly different scales (lookback ~ [0,1] vs entropy in nats).
    """
    scaler = StandardScaler().fit(X_tr) if standardize else None
    X_tr_s = scaler.transform(X_tr) if standardize else X_tr
    clf = LogisticRegression(max_iter=5000, C=C, random_state=seed).fit(X_tr_s, y_tr)
    return clf, scaler


def score_lr_probe(clf, scaler, X):
    X_s = scaler.transform(X) if scaler is not None else X
    return clf.decision_function(X_s)


def random_baseline(y_true, seed=0):
    """Uniform-random scores — establishes the ~0.50 AUROC trivial floor."""
    rng = np.random.default_rng(seed)
    return rng.random(size=len(y_true))


# ────────────────────────────────────────────────────────────────────────
#  Runner
# ────────────────────────────────────────────────────────────────────────

def run_feature_set(name, train_examples, test_examples, C, bootstrap, seed):
    """Fit LR on train examples, evaluate on test. Returns a results dict."""
    X_tr, y_tr = build_XY(train_examples, name)
    X_te, y_te = build_XY(test_examples, name)

    clf, scaler = fit_lr_probe(X_tr, y_tr, C=C, seed=seed)
    s_tr = score_lr_probe(clf, scaler, X_tr)
    s_te = score_lr_probe(clf, scaler, X_te)

    m_tr = evaluate_scores(y_tr, s_tr)
    m_te = evaluate_scores(y_te, s_te)

    ci = None
    if bootstrap and bootstrap > 0:
        ci = bootstrap_auroc(y_te, s_te, n_boot=bootstrap, seed=seed)

    return {
        "feature_set": name,
        "feat_dim":    int(X_tr.shape[1]),
        "train":       m_tr,
        "test":        m_te,
        "test_auroc_ci95": ci,
        "C":           C,
        "n_train":     int(len(y_tr)),
        "n_test":      int(len(y_te)),
    }


def run_random_baseline(test_examples, seed):
    y_te = np.array([e["label"] for e in test_examples], dtype=int)
    s_te = random_baseline(y_te, seed=seed)
    return {
        "feature_set": "random",
        "feat_dim":    0,
        "train":       None,
        "test":        evaluate_scores(y_te, s_te),
        "test_auroc_ci95": None,
        "C":           None,
        "n_train":     0,
        "n_test":      int(len(y_te)),
    }


def print_header(train_examples, test_examples, C):
    ytr = np.array([e["label"] for e in train_examples])
    yte = np.array([e["label"] for e in test_examples])
    print("═" * 96)
    print("  Step 2: Static Probes on AggreFact")
    print("═" * 96)
    print(f"  Train (cut=val):  n={len(ytr):<5d}  "
          f"faithful={int((ytr==1).sum())}  hallucinated={int((ytr==0).sum())}")
    print(f"  Test  (cut=test): n={len(yte):<5d}  "
          f"faithful={int((yte==1).sum())}  hallucinated={int((yte==0).sum())}")
    print(f"  LogReg C = {C}")
    print("═" * 96)


def print_result(r):
    ci = r.get("test_auroc_ci95")
    print(format_row(r["feature_set"], r["test"], dim=r["feat_dim"], ci=ci))


def print_top_features(results, train_examples, feature_set="all", top_k=15):
    """Inspect the `all` probe's most-informative features (|coef|)."""
    # Grab C from any trained probe (random baseline stores C=None, skip it).
    C = next((r["C"] for r in results if r.get("C") is not None), 1.0)
    X_tr, y_tr = build_XY(train_examples, feature_set)
    clf, scaler = fit_lr_probe(X_tr, y_tr, C=C)
    coefs = clf.coef_[0]
    abs_coefs = np.abs(coefs)

    # Reconstruct feature-block names to find which dim is which.
    # Block order must match utils.aggregation.FEATURE_SETS[feature_set].
    # Each lookback/entropy block contributes 4*L*H dims in agg order (mean,
    # min, max, var), each inner block L*H flattened row-major (L outer, H inner).
    L, H, _ = train_examples[0]["lookback_ratio"].shape
    LH = L * H
    aggs = ("mean", "min", "max", "var")

    names = []
    if feature_set == "all":
        for a in aggs:
            for l in range(L):
                for h in range(H):
                    names.append(f"lookback/{a}/L{l}-H{h}")
        for a in aggs:
            for l in range(L):
                for h in range(H):
                    names.append(f"attn_entropy/{a}/L{l}-H{h}")
        for k in ("chosen_prob", "output_entropy", "top_margin"):
            for a in aggs:
                names.append(f"logit/{k}/{a}")
    else:
        names = [f"dim_{i}" for i in range(len(coefs))]

    print()
    print("═" * 96)
    print(f"  Top-{top_k} most informative features in probe `{feature_set}`")
    print("═" * 96)
    top = np.argsort(-abs_coefs)[:top_k]
    for rank, i in enumerate(top, 1):
        name = names[i] if i < len(names) else f"dim_{i}"
        print(f"  {rank:2d}. {name:38s}  coef={coefs[i]:+.4f}")


def maybe_save_results(results, path):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved results → {path}")


# ────────────────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Step 2: static probes on Step-1 features.")
    p.add_argument("--features", required=True,
                   help="Path to features_*.pt produced by step01.")
    p.add_argument("--C", type=float, default=1.0,
                   help="Inverse L2 strength for logistic regression.")
    p.add_argument("--bootstrap", type=int, default=0,
                   help="Bootstrap rounds for 95%% AUROC CI on test "
                        "(0 disables; 1000 is a good default).")
    p.add_argument("--train-split", default="val",
                   help="AggreFact cut used to train probes (default: val).")
    p.add_argument("--test-split", default="test",
                   help="AggreFact cut used to evaluate probes (default: test).")
    p.add_argument("--feature-sets", nargs="+", default=None,
                   help="Subset of feature-set names to run. "
                        f"Default: all of {list(FEATURE_SETS)}.")
    p.add_argument("--results-path", default=None,
                   help="If set, dump results JSON here.")
    p.add_argument("--show-top-features", action="store_true",
                   help="Print the top-|coef| features in the `all` probe.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading {args.features} ...")
    examples = load_examples(args.features)
    print(f"  Loaded {len(examples)} labeled examples.")

    train_ex, test_ex = split_examples(examples, args.train_split, args.test_split)
    if not train_ex or not test_ex:
        raise SystemExit(
            f"Empty split — train({args.train_split})={len(train_ex)}, "
            f"test({args.test_split})={len(test_ex)}. "
            f"Check the `split` field in your features file."
        )

    print_header(train_ex, test_ex, args.C)

    results = []

    # Random baseline first (no training).
    r = run_random_baseline(test_ex, seed=args.seed)
    results.append(r)
    print_result(r)

    sets_to_run = args.feature_sets or list(FEATURE_SETS.keys())
    for name in sets_to_run:
        if name not in FEATURE_SETS:
            print(f"  [skip] unknown feature set: {name}")
            continue
        r = run_feature_set(name, train_ex, test_ex,
                            C=args.C, bootstrap=args.bootstrap, seed=args.seed)
        results.append(r)
        print_result(r)

    if args.show_top_features and "all" in sets_to_run:
        print_top_features(results, train_ex, feature_set="all")

    maybe_save_results(results, args.results_path)

    # Headline comparison — the two numbers the proposal explicitly asks about.
    lens = next((r for r in results if r["feature_set"] == "lookback_lens"), None)
    allr = next((r for r in results if r["feature_set"] == "all"), None)
    if lens and allr:
        print()
        print("═" * 96)
        print("  Headline: Lookback-Lens baseline vs. combined features")
        print("═" * 96)
        print(f"  lookback_lens AUROC = {lens['test']['auroc']:.4f}")
        print(f"  all-features  AUROC = {allr['test']['auroc']:.4f}")
        delta = allr['test']['auroc'] - lens['test']['auroc']
        print(f"  Δ AUROC             = {delta:+.4f}")


if __name__ == "__main__":
    main()
