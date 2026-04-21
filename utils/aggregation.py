"""
Span-level aggregation of per-token features.

Step 1 extracts per-token tensors:
    lookback_ratio       : (L, H, T)
    attn_entropy         : (L, H, T)
    logit_chosen_prob    : (T,)
    logit_output_entropy : (T,)
    logit_top_margin     : (T,)

For static (non-temporal) probes we collapse T into a fixed-size vector.
The Lookback-Lens baseline uses mean-only; we also expose min/max/var so
downstream probes can capture mid-span drops/spikes that averaging hides.
"""

import numpy as np
import torch

# Names in the order they appear in logit_feats_vector.
LOGIT_FEAT_KEYS = ("logit_chosen_prob", "logit_output_entropy", "logit_top_margin")

# Aggregations applied along the time axis.
AGGS = ("mean", "min", "max", "var")


def _agg_time(x: torch.Tensor, agg: str) -> torch.Tensor:
    """x has shape (..., T). Returns (...,) after reducing the last axis."""
    if agg == "mean":
        return x.mean(dim=-1)
    if agg == "min":
        return x.amin(dim=-1)
    if agg == "max":
        return x.amax(dim=-1)
    if agg == "var":
        # unbiased=False matches numpy.var default — gives 0 for T=1 instead of NaN.
        return x.var(dim=-1, unbiased=False)
    raise ValueError(f"unknown agg: {agg}")


def aggregate_lh_tensor(x: torch.Tensor, aggs=AGGS) -> np.ndarray:
    """
    x : (L, H, T) → (len(aggs) * L * H,) flat numpy vector.
    Order: for each agg in `aggs`, flatten (L*H) in row-major (L outer, H inner).
    """
    parts = [_agg_time(x, a).flatten().cpu().numpy() for a in aggs]
    return np.concatenate(parts)


def aggregate_scalar_series(x: torch.Tensor, aggs=AGGS) -> np.ndarray:
    """x : (T,) → (len(aggs),) flat numpy vector."""
    return np.array([_agg_time(x, a).item() for a in aggs])


# ────────────────────────────────────────────────────────────────────────
#  Named feature blocks — each returns (name, np.ndarray)
#  so downstream code can assemble combined vectors + track slices.
# ────────────────────────────────────────────────────────────────────────

def block_lookback_mean(ex) -> np.ndarray:
    """Lookback-Lens baseline: mean over T only → (L*H,)."""
    return aggregate_lh_tensor(ex["lookback_ratio"], aggs=("mean",))


def block_lookback_full(ex) -> np.ndarray:
    """Mean/min/max/var of lookback → (4*L*H,)."""
    return aggregate_lh_tensor(ex["lookback_ratio"], aggs=AGGS)


def block_attn_entropy_full(ex) -> np.ndarray:
    return aggregate_lh_tensor(ex["attn_entropy"], aggs=AGGS)


def block_logits_full(ex) -> np.ndarray:
    """Mean/min/max/var of each of the 3 logit streams → (12,)."""
    return np.concatenate([
        aggregate_scalar_series(ex[k], aggs=AGGS) for k in LOGIT_FEAT_KEYS
    ])


def block_output_prob_only(ex) -> np.ndarray:
    """Single-scalar baseline — mean chosen_prob over the span."""
    return np.array([ex["logit_chosen_prob"].mean().item()])


# ────────────────────────────────────────────────────────────────────────
#  Feature-set registry — maps a name to a builder + the blocks it uses.
# ────────────────────────────────────────────────────────────────────────

FEATURE_SETS = {
    # Baselines
    "output_prob_only":   [block_output_prob_only],
    "lookback_lens":      [block_lookback_mean],          # Chuang et al. baseline
    # Single families (with richer aggregation)
    "lookback_full":      [block_lookback_full],
    "attn_entropy_full":  [block_attn_entropy_full],
    "logits_full":        [block_logits_full],
    # Pairwise combinations
    "lookback+entropy":   [block_lookback_full, block_attn_entropy_full],
    "lookback+logits":    [block_lookback_full, block_logits_full],
    "entropy+logits":     [block_attn_entropy_full, block_logits_full],
    # All three
    "all":                [block_lookback_full, block_attn_entropy_full, block_logits_full],
}


def build_feature_vector(ex, feature_set: str) -> np.ndarray:
    """Build the flat feature vector for one example under a named feature set."""
    blocks = FEATURE_SETS[feature_set]
    return np.concatenate([b(ex) for b in blocks])


# ────────────────────────────────────────────────────────────────────────
#  Temporal tensors — used by CNN/LSTM probes, not the static LR probes.
# ────────────────────────────────────────────────────────────────────────

def temporal_lookback_matrix(ex) -> torch.Tensor:
    """(L, H, T) → (L*H, T). Channels-first, time along the last axis."""
    L, H, T = ex["lookback_ratio"].shape
    return ex["lookback_ratio"].reshape(L * H, T).float()


def temporal_all_matrix(ex) -> torch.Tensor:
    """
    Concat all per-token feature streams along the channel dim:
        lookback_ratio (L*H) + attn_entropy (L*H) + 3 logit streams
        → (2*L*H + 3, T).
    """
    lb = temporal_lookback_matrix(ex)                         # (L*H, T)
    ae = ex["attn_entropy"].reshape(lb.shape[0], -1).float()  # (L*H, T)
    lg = torch.stack([
        ex["logit_chosen_prob"].float(),
        ex["logit_output_entropy"].float(),
        ex["logit_top_margin"].float(),
    ], dim=0)                                                  # (3, T)
    return torch.cat([lb, ae, lg], dim=0)
