"""
Step 4: Seed variance for headline temporal probes.

Runs the existing Step-3 training/evaluation path for seeds 0, 1, and 2,
then aggregates AggreFact test and optional transfer AUROC as mean +/- std.
"""

import argparse
import json
import os
from argparse import Namespace

import numpy as np

from step03_temporal_probes import run as run_temporal_probe


PROBES = (
    ("cnn", "lookback"),
    ("lstm", "all"),
)


def make_step3_args(args, model, channels, seed, results_path, save_model):
    return Namespace(
        features=args.features,
        transfer_features=args.transfer_features,
        transfer_label=args.transfer_label,
        train_split=args.train_split,
        test_split=args.test_split,
        model=model,
        channels=channels,
        hidden=args.hidden,
        dropout=args.dropout,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        patience=args.patience,
        val_frac=args.val_frac,
        balance_loss=args.balance_loss,
        max_T=args.max_T,
        bootstrap=args.bootstrap,
        seed=seed,
        device=args.device,
        log_every=args.log_every,
        results_path=results_path,
        save_model=save_model,
    )


def summarize(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"mean": None, "std": None, "values": []}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "values": [float(v) for v in arr],
    }


def parse_args():
    p = argparse.ArgumentParser(description="Step 4: temporal seed variance.")
    p.add_argument("--features", required=True)
    p.add_argument("--transfer-features", default=None)
    p.add_argument("--transfer-label", default="XSum")
    p.add_argument("--train-split", default="val")
    p.add_argument("--test-split", default="test")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--results-path", default="results/step04_seed_variance.json")
    p.add_argument("--checkpoint-dir", default="results/seed_variance")

    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--balance-loss", action="store_true")
    p.add_argument("--max-T", type=int, default=256)
    p.add_argument("--bootstrap", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--log-every", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.results_path) or ".", exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    aggregate = {"args": vars(args), "probes": {}}
    for model, channels in PROBES:
        key = f"{model}+{channels}"
        per_seed = []
        for seed in args.seeds:
            result_path = os.path.join(args.checkpoint_dir, f"{key}_seed{seed}.json")
            checkpoint_path = os.path.join(args.checkpoint_dir, f"{key}_seed{seed}.pt")
            step3_args = make_step3_args(
                args, model, channels, seed, result_path, checkpoint_path
            )
            run_temporal_probe(step3_args)
            with open(result_path) as f:
                per_seed.append(json.load(f))

        test_aurocs = [r["test"]["auroc"] for r in per_seed]
        transfer_aurocs = [
            r["transfer"]["auroc"] if r.get("transfer") else None
            for r in per_seed
        ]
        aggregate["probes"][key] = {
            "per_seed": [
                {
                    "seed": r["args"]["seed"],
                    "test_auroc": r["test"]["auroc"],
                    "transfer_auroc": (
                        r["transfer"]["auroc"] if r.get("transfer") else None
                    ),
                }
                for r in per_seed
            ],
            "test_auroc": summarize(test_aurocs),
            "transfer_auroc": summarize(transfer_aurocs),
        }

    with open(args.results_path, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"Saved seed-variance summary -> {args.results_path}")


if __name__ == "__main__":
    main()
