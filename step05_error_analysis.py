"""
Step 5: Error analysis for the saved CNN temporal probe.

Loads a Step-3 checkpoint, scores AggreFact test examples without retraining
the CNN, and writes dataset/type breakdowns plus hard examples for manual
inspection.
"""

import argparse
import ast
import csv
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from step02_static_probes import (
    build_XY,
    fit_lr_probe,
    score_lr_probe,
    split_train_threshold_examples,
)
from step03_temporal_probes import (
    CHANNEL_BUILDERS,
    SpanSequenceDataset,
    collate_pad,
    load_examples,
    make_model,
    score_loader,
    split_examples,
)
from utils.evaluation import evaluate_scores


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def normalize_text(text):
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "state_dict" not in ckpt:
        raise ValueError(
            "Expected a Step-3 checkpoint saved after this branch's metadata "
            "update. Re-run step03 with --save-model."
        )
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
    builder = CHANNEL_BUILDERS[channels]
    loader = torch.utils.data.DataLoader(
        SpanSequenceDataset(examples, builder, max_T=max_T),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_pad,
        num_workers=0,
    )
    return score_loader(model, loader, device)


def roc_optimal_threshold(y_true, scores):
    if len(np.unique(y_true)) < 2:
        return 0.0
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    return float(thresholds[int(np.argmax(tpr - fpr))])


def safe_metric(fn, y, s):
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(fn(y, s))


def write_dataset_breakdown(path, examples, scores, threshold):
    rows = []
    by_dataset = defaultdict(list)
    for i, ex in enumerate(examples):
        by_dataset[ex.get("source_dataset", "?")].append(i)

    for dataset, idxs in sorted(by_dataset.items()):
        y = np.array([examples[i]["label"] for i in idxs], dtype=int)
        s = np.array([scores[i] for i in idxs], dtype=float)
        pred_hallu = s < threshold
        faithful = y == 1
        hallu = y == 0
        fp_rate = float((pred_hallu & faithful).sum() / faithful.sum()) if faithful.sum() else float("nan")
        fn_rate = float(((~pred_hallu) & hallu).sum() / hallu.sum()) if hallu.sum() else float("nan")
        rows.append({
            "source_dataset": dataset,
            "n": int(len(idxs)),
            "faithful": int(faithful.sum()),
            "hallucinated": int(hallu.sum()),
            "auroc": safe_metric(roc_auc_score, y, s),
            "auprc": safe_metric(average_precision_score, y, s),
            "false_positive_rate": fp_rate,
            "false_negative_rate": fn_rate,
        })

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_model_breakdown(path, examples, scores, threshold):
    rows = []
    by_model = defaultdict(list)
    for i, ex in enumerate(examples):
        by_model[ex.get("source_model", "?")].append(i)

    for model, idxs in sorted(by_model.items()):
        y = np.array([examples[i]["label"] for i in idxs], dtype=int)
        s = np.array([scores[i] for i in idxs], dtype=float)
        pred_hallu = s < threshold
        faithful = y == 1
        hallu = y == 0
        fp_rate = float((pred_hallu & faithful).sum() / faithful.sum()) if faithful.sum() else float("nan")
        fn_rate = float(((~pred_hallu) & hallu).sum() / hallu.sum()) if hallu.sum() else float("nan")
        rows.append({
            "source_model": model,
            "n": int(len(idxs)),
            "faithful": int(faithful.sum()),
            "hallucinated": int(hallu.sum()),
            "auroc": safe_metric(roc_auc_score, y, s),
            "auprc": safe_metric(average_precision_score, y, s),
            "false_positive_rate": fp_rate,
            "false_negative_rate": fn_rate,
        })

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def parse_error_types(value):
    if pd.isna(value):
        return []
    text = str(value)
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
        return [str(parsed)]
    except (ValueError, SyntaxError):
        return [x.strip() for x in text.split(";") if x.strip()]


def load_error_type_maps(mapping_dir):
    exact = defaultdict(set)
    loose = defaultdict(set)
    for name in os.listdir(mapping_dir):
        if not name.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(mapping_dir, name))
        for _, row in df.iterrows():
            dataset = str(row.get("dataset", "")).lower()
            ex_id = str(row.get("id", ""))
            model = str(row.get("model_name", ""))
            summary = normalize_text(row.get("summ", row.get("summary", "")))
            types = parse_error_types(row.get("error_type", ""))
            if not types:
                continue
            exact[(dataset, ex_id, model, summary)].update(types)
            loose[(dataset, ex_id, model)].update(types)
            loose[(dataset, ex_id, "")].update(types)
    return exact, loose


def error_types_for_example(ex, exact, loose):
    dataset = str(ex.get("source_dataset", "")).lower()
    ex_id = str(ex.get("example_id", ""))
    model = str(ex.get("source_model", ""))
    summary = normalize_text(ex.get("summary_text", ""))
    return (
        exact.get((dataset, ex_id, model, summary)) or
        loose.get((dataset, ex_id, model)) or
        loose.get((dataset, ex_id, "")) or
        set()
    )


def write_type_breakdown(path, examples, scores, threshold, mapping_dir):
    exact, loose = load_error_type_maps(mapping_dir)
    counts = defaultdict(lambda: {"n": 0, "caught": 0})
    for ex, score in zip(examples, scores):
        if ex["label"] != 0:
            continue
        caught = score < threshold
        for err_type in error_types_for_example(ex, exact, loose):
            counts[err_type]["n"] += 1
            counts[err_type]["caught"] += int(caught)

    rows = []
    for err_type, vals in sorted(counts.items()):
        rows.append({
            "error_type": err_type,
            "n": vals["n"],
            "caught": vals["caught"],
            "recall": vals["caught"] / vals["n"] if vals["n"] else float("nan"),
        })
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def train_static_lookback_probe(train_examples, seed, threshold_frac):
    fit_examples, threshold_examples = split_train_threshold_examples(
        train_examples, threshold_frac=threshold_frac, seed=seed
    )
    X_fit, y_fit = build_XY(fit_examples, "lookback_lens")
    X_thr, y_thr = build_XY(threshold_examples, "lookback_lens")
    clf, scaler = fit_lr_probe(X_fit, y_fit, seed=seed)
    threshold = evaluate_scores(
        y_thr, score_lr_probe(clf, scaler, X_thr)
    )["thr_opt"]
    return clf, scaler, threshold


def example_payload(ex, score, threshold, kind):
    trace = ex["lookback_ratio"].mean(dim=(0, 1)).detach().cpu().tolist()
    return {
        "kind": kind,
        "example_id": ex.get("example_id"),
        "source_dataset": ex.get("source_dataset"),
        "source_model": ex.get("source_model"),
        "label": int(ex["label"]),
        "cnn_faithful_score": float(score),
        "cnn_hallucination_probability": float(sigmoid(-score)),
        "threshold": float(threshold),
        "document": ex.get("document_text", ""),
        "summary": ex.get("summary_text", ""),
        "lookback_trace": [float(x) for x in trace],
    }


def write_hard_examples(path, train_examples, test_examples, scores, threshold, seed):
    clf, scaler, lr_threshold = train_static_lookback_probe(train_examples, seed, 0.15)
    X_test, _ = build_XY(test_examples, "lookback_lens")
    lr_scores = score_lr_probe(clf, scaler, X_test)

    y = np.array([e["label"] for e in test_examples], dtype=int)
    scores = np.asarray(scores)
    pred_hallu = scores < threshold
    lr_pred_hallu = lr_scores < lr_threshold

    fp = [
        (sigmoid(-scores[i]), i) for i in range(len(test_examples))
        if y[i] == 1 and pred_hallu[i]
    ]
    fn = [
        (sigmoid(scores[i]), i) for i in range(len(test_examples))
        if y[i] == 0 and not pred_hallu[i]
    ]
    saves = [
        (sigmoid(-scores[i]), i) for i in range(len(test_examples))
        if y[i] == 0 and pred_hallu[i] and not lr_pred_hallu[i]
    ]

    hard = []
    for _, i in sorted(fp, reverse=True)[:5]:
        hard.append(example_payload(test_examples[i], scores[i], threshold, "confident_false_positive"))
    for _, i in sorted(fn, reverse=True)[:5]:
        hard.append(example_payload(test_examples[i], scores[i], threshold, "confident_false_negative"))
    for _, i in sorted(saves, reverse=True)[:5]:
        item = example_payload(test_examples[i], scores[i], threshold, "cnn_saves_static_miss")
        item["static_lookback_score"] = float(lr_scores[i])
        item["static_lookback_threshold"] = float(lr_threshold)
        hard.append(item)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(hard, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser(description="Step 5: CNN error analysis.")
    p.add_argument("--features", default="/projectnb/cs505am/projects/faithfulness_probe/features_aggrefact_sota.pt")
    p.add_argument("--model-path", default="results/cnn_lookback.pt")
    p.add_argument("--mapping-dir", default="data/error_type_mapping")
    p.add_argument("--dataset-csv", default="results/error_breakdown_by_dataset.csv")
    p.add_argument("--model-csv", default="results/error_breakdown_by_model.csv")
    p.add_argument("--type-csv", default="results/error_breakdown_by_type.csv")
    p.add_argument("--hard-json", default="results/hard_examples.json")
    p.add_argument("--train-split", default="val")
    p.add_argument("--test-split", default="test")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    ckpt, model = load_checkpoint(args.model_path, device)
    examples = load_examples(args.features, require_split=True)
    train_ex, test_ex = split_examples(examples, args.train_split, args.test_split)
    y_val, s_val = score_examples(
        model,
        train_ex,
        ckpt["channels"],
        ckpt.get("max_T", 256),
        args.batch_size,
        device,
    )
    threshold = roc_optimal_threshold(y_val, s_val)
    y_test, s_test = score_examples(
        model,
        test_ex,
        ckpt["channels"],
        ckpt.get("max_T", 256),
        args.batch_size,
        device,
    )

    write_dataset_breakdown(args.dataset_csv, test_ex, s_test, threshold)
    write_model_breakdown(args.model_csv, test_ex, s_test, threshold)
    write_type_breakdown(args.type_csv, test_ex, s_test, threshold, args.mapping_dir)
    write_hard_examples(args.hard_json, train_ex, test_ex, s_test, threshold, args.seed)
    print(f"Saved dataset breakdown -> {args.dataset_csv}")
    print(f"Saved model breakdown  -> {args.model_csv}")
    print(f"Saved error-type breakdown -> {args.type_csv}")
    print(f"Saved hard examples -> {args.hard_json}")


if __name__ == "__main__":
    main()
