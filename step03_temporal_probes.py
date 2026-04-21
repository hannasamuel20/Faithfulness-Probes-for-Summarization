"""
Step 3: Temporal Faithfulness Probes (1D-CNN + LSTM)
====================================================

Answers proposal question #1: does a temporal model over the per-token
lookback-ratio sequence beat the mean-pooled Lookback-Lens baseline?

Pipeline:
  - Load per-token features produced by step01_extract_features.py.
  - Build (C, T) sequences per span, where C is either:
        lookback  : L*H channels (lookback ratio only — paper-faithful)
        all       : 2*L*H + 3 channels (+ entropy + 3 logit streams)
  - Pad to the longest span in each batch, track validity mask.
  - Train CNN / LSTM with BCEWithLogits, AdamW, early-stop on val AUROC.
  - Report AUROC / AUPRC / F1@opt on Train + Test, and Transfer if a
    second features file is supplied (Lookback-Lens Table 2 format).

AggreFact convention: train on cut=val, test on cut=test. Labels:
1 = faithful (positive class), 0 = hallucinated.

Usage:
    python step03_temporal_probes.py \\
        --features features_aggrefact_sota.pt \\
        --model cnn --channels lookback \\
        --results-path results/step03_cnn_lookback.json

    # LSTM on full feature stack + XSum transfer:
    python step03_temporal_probes.py \\
        --features features_aggrefact_sota.pt \\
        --transfer-features features_xsum.pt \\
        --model lstm --channels all
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from utils.aggregation import temporal_lookback_matrix, temporal_all_matrix
from utils.evaluation import (
    evaluate_scores, bootstrap_auroc, format_row,
    format_paper_row, format_paper_header,
)
from models.temporal_probes import LookbackCNN, LookbackLSTM, count_params


# ────────────────────────────────────────────────────────────────────────
#  Data
# ────────────────────────────────────────────────────────────────────────

CHANNEL_BUILDERS = {
    "lookback": temporal_lookback_matrix,    # (L*H, T)
    "all":      temporal_all_matrix,         # (2*L*H + 3, T)
}


class SpanSequenceDataset(Dataset):
    """Yields (C, T_i) tensors + labels. Collate handles padding."""

    def __init__(self, examples, channel_builder, max_T=None):
        self.examples = examples
        self.build = channel_builder
        self.max_T = max_T

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ex = self.examples[i]
        x = self.build(ex)                              # (C, T_i)
        if self.max_T is not None and x.shape[1] > self.max_T:
            x = x[:, :self.max_T]                       # truncate long spans
        y = float(ex["label"])
        return x, y


def collate_pad(batch):
    """Pad variable-length (C, T_i) tensors to (B, C, T_max) + mask."""
    xs, ys = zip(*batch)
    C = xs[0].shape[0]
    T_max = max(x.shape[1] for x in xs)
    B = len(xs)

    X = torch.zeros(B, C, T_max, dtype=torch.float32)
    mask = torch.zeros(B, T_max, dtype=torch.bool)
    for i, x in enumerate(xs):
        T_i = x.shape[1]
        X[i, :, :T_i] = x
        mask[i, :T_i] = True

    y = torch.tensor(ys, dtype=torch.float32)
    return X, mask, y


def load_examples(path):
    blob = torch.load(path, weights_only=False)
    examples = blob["examples"] if isinstance(blob, dict) else blob
    return [e for e in examples if "label" in e and "split" in e]


def split_examples(examples, train_split="val", test_split="test"):
    return (
        [e for e in examples if e["split"] == train_split],
        [e for e in examples if e["split"] == test_split],
    )


# ────────────────────────────────────────────────────────────────────────
#  Training
# ────────────────────────────────────────────────────────────────────────

def make_model(name, in_channels, hidden, dropout):
    if name == "cnn":
        return LookbackCNN(in_channels, hidden=hidden, dropout=dropout)
    if name == "lstm":
        return LookbackLSTM(in_channels, hidden=hidden, dropout=dropout)
    raise ValueError(f"unknown model: {name}")


@torch.no_grad()
def score_loader(model, loader, device):
    """Return (y_true, logits) for every example in a loader, in order."""
    model.eval()
    ys, ss = [], []
    for X, mask, y in loader:
        X, mask = X.to(device), mask.to(device)
        logits = model(X, mask)
        ys.append(y.numpy())
        ss.append(logits.detach().cpu().numpy())
    return np.concatenate(ys), np.concatenate(ss)


def train_one_model(model, train_loader, val_loader, device,
                    epochs, lr, weight_decay, patience,
                    pos_weight=None, log_every=1, seed=0):
    """
    Standard BCEWithLogits training with early stopping on val AUROC.
    Returns (best_state_dict, best_val_auroc).
    """
    torch.manual_seed(seed)
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=device) if pos_weight else None
    )

    best_auroc = -1.0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    stale = 0

    for ep in range(1, epochs + 1):
        model.train()
        total_loss, total_n = 0.0, 0
        for X, mask, y in train_loader:
            X, mask, y = X.to(device), mask.to(device), y.to(device)
            optim.zero_grad()
            logits = model(X, mask)
            loss = loss_fn(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total_loss += loss.item() * y.size(0)
            total_n += y.size(0)

        y_val, s_val = score_loader(model, val_loader, device)
        m_val = evaluate_scores(y_val, s_val)

        if ep % log_every == 0 or ep == 1:
            print(f"  epoch {ep:3d}  "
                  f"train_loss={total_loss/total_n:.4f}  "
                  f"val_AUROC={m_val['auroc']:.4f}  "
                  f"val_AUPRC={m_val['auprc']:.4f}")

        if m_val["auroc"] > best_auroc:
            best_auroc = m_val["auroc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"  early stop at epoch {ep}  (best val AUROC={best_auroc:.4f})")
                break

    model.load_state_dict(best_state)
    return model, best_auroc


# ────────────────────────────────────────────────────────────────────────
#  Runner
# ────────────────────────────────────────────────────────────────────────

def run(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")

    # ── seeding ──
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── data ──
    print(f"Loading {args.features} ...")
    examples = load_examples(args.features)
    print(f"  Loaded {len(examples)} labeled examples.")
    train_ex, test_ex = split_examples(examples, args.train_split, args.test_split)

    # Hold out a small val slice from the source-train split for early stop.
    y_train = [e["label"] for e in train_ex]
    tr_ex, val_ex = train_test_split(
        train_ex, test_size=args.val_frac, stratify=y_train, random_state=args.seed
    )

    transfer_ex = None
    if args.transfer_features:
        print(f"Loading transfer features: {args.transfer_features}")
        transfer_ex = load_examples(args.transfer_features)
        print(f"  Loaded {len(transfer_ex)} labeled transfer examples.")

    # ── probe inputs ──
    builder = CHANNEL_BUILDERS[args.channels]
    in_channels = builder(train_ex[0]).shape[0]
    print(f"\n  Model={args.model}  channels={args.channels} (C={in_channels})  "
          f"device={device}")
    print(f"  Train={len(tr_ex)}  Val(for early stop)={len(val_ex)}  "
          f"Test={len(test_ex)}  Transfer={len(transfer_ex) if transfer_ex else 0}")

    # ── datasets ──
    mk = lambda xs: SpanSequenceDataset(xs, builder, max_T=args.max_T)
    kw = dict(batch_size=args.batch_size, collate_fn=collate_pad, num_workers=0)
    tr_loader  = DataLoader(mk(tr_ex),  shuffle=True,  **kw)
    val_loader = DataLoader(mk(val_ex), shuffle=False, **kw)
    full_train_loader = DataLoader(mk(train_ex), shuffle=False, **kw)
    te_loader  = DataLoader(mk(test_ex), shuffle=False, **kw)
    xf_loader  = DataLoader(mk(transfer_ex), shuffle=False, **kw) if transfer_ex else None

    # ── positive-class reweighting for imbalance (63% faithful on AggreFact) ──
    pos_w = None
    if args.balance_loss:
        n_pos = sum(1 for e in tr_ex if e["label"] == 1)
        n_neg = len(tr_ex) - n_pos
        pos_w = n_neg / max(n_pos, 1)
        print(f"  pos_weight = {pos_w:.3f}  (n_pos={n_pos}, n_neg={n_neg})")

    # ── model ──
    model = make_model(args.model, in_channels, args.hidden, args.dropout)
    print(f"  params: {count_params(model):,}")

    # ── train ──
    print("\n  Training ...")
    model, best_val = train_one_model(
        model, tr_loader, val_loader, device,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        patience=args.patience, pos_weight=pos_w, log_every=args.log_every,
        seed=args.seed,
    )

    # ── eval (Train on the FULL train cut to match LR setup in step02) ──
    y_tr, s_tr = score_loader(model, full_train_loader, device)
    y_te, s_te = score_loader(model, te_loader, device)
    m_tr = evaluate_scores(y_tr, s_tr)
    m_te = evaluate_scores(y_te, s_te)

    m_xf = None
    if xf_loader:
        y_xf, s_xf = score_loader(model, xf_loader, device)
        m_xf = evaluate_scores(y_xf, s_xf)

    ci = bootstrap_auroc(y_te, s_te, n_boot=args.bootstrap, seed=args.seed) \
        if args.bootstrap > 0 else None

    # ── report ──
    print()
    print("═" * 96)
    print(f"  Step 3 — Temporal probe results  ({args.model}, channels={args.channels})")
    print("═" * 96)
    print(format_row(f"{args.model}+{args.channels}", m_te,
                     dim=in_channels, ci=ci))
    print(f"  train AUROC = {m_tr['auroc']:.4f}   (best val during training = {best_val:.4f})")
    if m_xf:
        print(f"  transfer    : " + format_row("transfer", m_xf))

    print()
    print("═" * 96)
    print("  Lookback-Lens Table-2 format (AUROC ×100)")
    print("═" * 96)
    print(format_paper_header(source_label="AggF", target_label=args.transfer_label))
    print("  " + "-" * 60)
    print(format_paper_row(
        f"{args.model}+{args.channels}",
        m_tr["auroc"], m_te["auroc"], m_xf["auroc"] if m_xf else None,
    ))

    # ── save ──
    if args.results_path:
        os.makedirs(os.path.dirname(args.results_path) or ".", exist_ok=True)
        payload = {
            "args": vars(args),
            "best_val_auroc": float(best_val),
            "train": m_tr,
            "test": m_te,
            "transfer": m_xf,
            "test_auroc_ci95": ci,
            "in_channels": int(in_channels),
            "n_params": int(count_params(model)),
            "n_train": len(train_ex),
            "n_test": len(test_ex),
            "n_transfer": len(transfer_ex) if transfer_ex else 0,
        }
        with open(args.results_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n  Saved results → {args.results_path}")

    if args.save_model:
        torch.save(model.state_dict(), args.save_model)
        print(f"  Saved model   → {args.save_model}")


# ────────────────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Step 3: temporal CNN/LSTM probes.")
    p.add_argument("--features", required=True)
    p.add_argument("--transfer-features", default=None,
                   help="Optional second features .pt for Transfer-column AUROC.")
    p.add_argument("--transfer-label", default="XSum")
    p.add_argument("--train-split", default="val")
    p.add_argument("--test-split", default="test")

    p.add_argument("--model", choices=["cnn", "lstm"], default="cnn")
    p.add_argument("--channels", choices=list(CHANNEL_BUILDERS), default="lookback",
                   help="lookback: just lookback ratio (L*H). "
                        "all: + attn_entropy + 3 logit streams.")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.3)

    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--patience", type=int, default=10,
                   help="Early-stop patience (epochs without val-AUROC gain).")
    p.add_argument("--val-frac", type=float, default=0.15,
                   help="Fraction of train-split held out for early stopping.")
    p.add_argument("--balance-loss", action="store_true",
                   help="Use BCE pos_weight to offset 63/37 class imbalance.")
    p.add_argument("--max-T", type=int, default=256,
                   help="Truncate spans longer than this many tokens (memory).")

    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--results-path", default=None)
    p.add_argument("--save-model", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
