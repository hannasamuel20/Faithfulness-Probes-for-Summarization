"""
Evaluation metrics for binary faithfulness probes.

Primary metric:  AUROC  (matches Lookback-Lens paper, threshold-free ranking)
Secondary     :  AUPRC  (class-imbalance sensitive)
Operating pt  :  F1 at the threshold maximising F1 on the scored set.

Label convention in AggreFact: 1 = faithful, 0 = hallucinated.
We score "faithful" as the positive class so AUROC matches sign conventions
in smoke_probe.py and the Chuang et al. baseline.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score,
)


def f1_at_optimal_threshold(y_true, y_score):
    """
    Sweep every threshold implied by y_score, return (best_f1, best_threshold).

    precision_recall_curve returns one extra precision/recall pair at the
    high-threshold end; `thresholds` has one fewer element, so we slice to
    match before computing F1.
    """
    p, r, thr = precision_recall_curve(y_true, y_score)
    p, r = p[:-1], r[:-1]                       # align with thresholds
    denom = (p + r)
    denom[denom == 0] = 1e-12                   # avoid 0/0 at degenerate points
    f1 = 2 * p * r / denom
    if len(f1) == 0:                            # happens only on single-class input
        return 0.0, 0.5
    best = int(np.argmax(f1))
    return float(f1[best]), float(thr[best])


def evaluate_scores(y_true, y_score):
    """
    Returns a dict with auroc, auprc, f1_opt, thr_opt, n, pos_rate.

    y_score should be a continuous confidence the example is faithful
    (e.g. logistic regression decision_function or predict_proba[:, 1]).
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    out = {
        "n":        int(len(y_true)),
        "pos_rate": float((y_true == 1).mean()) if len(y_true) else 0.0,
    }

    # AUROC / AUPRC are undefined on a single-class set — degrade gracefully.
    if len(np.unique(y_true)) < 2:
        out.update(auroc=float("nan"), auprc=float("nan"),
                   f1_opt=float("nan"), thr_opt=float("nan"))
        return out

    out["auroc"] = float(roc_auc_score(y_true, y_score))
    out["auprc"] = float(average_precision_score(y_true, y_score))
    f1_opt, thr_opt = f1_at_optimal_threshold(y_true, y_score)
    out["f1_opt"] = f1_opt
    out["thr_opt"] = thr_opt
    return out


def bootstrap_auroc(y_true, y_score, n_boot=1000, seed=0):
    """
    Percentile bootstrap 95% CI for AUROC. Useful when comparing feature sets
    where the headline AUROC differences are small (e.g. 0.72 vs 0.74).
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    n = len(y_true)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            boots[i] = np.nan
            continue
        boots[i] = roc_auc_score(yt, ys)
    boots = boots[~np.isnan(boots)]
    if len(boots) == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def format_row(name: str, metrics: dict, dim: int | None = None,
               ci: tuple[float, float] | None = None) -> str:
    """Pretty one-line summary row for console output (detailed)."""
    d = f" dim={dim:<5d}" if dim is not None else ""
    ci_str = f"  [{ci[0]:.3f}, {ci[1]:.3f}]" if ci is not None else ""
    return (f"  {name:22s}{d}  "
            f"AUROC={metrics['auroc']:.4f}{ci_str}  "
            f"AUPRC={metrics['auprc']:.4f}  "
            f"F1@opt={metrics['f1_opt']:.4f}  "
            f"(thr={metrics['thr_opt']:+.3f}, n={metrics['n']})")


# ────────────────────────────────────────────────────────────────────────
#  Lookback-Lens-style reporting (Chuang et al. 2024, Table 2)
#
#  AUROC ×100, one decimal place, three columns:
#      Train     — source-task train-split AUROC
#      Test      — source-task held-out AUROC
#      Transfer  — out-of-domain target-task AUROC (optional)
#
#  In our AggreFact setup: Train = cut=val, Test = cut=test, Transfer = XSum.
# ────────────────────────────────────────────────────────────────────────

def _fmt_pct(x):
    """Paper format: AUROC ×100, one decimal, or '--' for missing."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "  --"
    return f"{100.0 * x:5.1f}"


def format_paper_row(name: str,
                     train_auroc: float | None,
                     test_auroc: float | None,
                     transfer_auroc: float | None) -> str:
    """Single row for the Lookback-Lens-style summary table."""
    return (f"  {name:22s}  "
            f"{_fmt_pct(train_auroc)}    "
            f"{_fmt_pct(test_auroc)}    "
            f"{_fmt_pct(transfer_auroc)}")


def format_paper_header(source_label: str = "Source",
                        target_label: str = "Target") -> str:
    """Two-line header matching Lookback Lens Table 2 layout."""
    return (f"  {'Method':22s}  "
            f"{'Train':>5s}   "
            f"{'Test':>5s}   "
            f"{'Transfer':>5s}\n"
            f"  {'':22s}  "
            f"{source_label + ' →':>5s}   "
            f"{source_label + ' →':>5s}   "
            f"{source_label + ' → ' + target_label:>5s}")
