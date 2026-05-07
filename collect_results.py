# transfer/collect_results.py
"""
Collects all probe results into a single summary table matching
Lookback Lens Table 2 format (AUROC x100).

Reads all JSON results files from the results directory and prints
a unified table with Train, Test, and Transfer (NQ) AUROC columns.

Usage:
    python transfer/collect_results.py \
        --results-dir results \
        --output results/summary_table.json
"""

import json, os, argparse

def load_json(path):
    with open(path) as f:
        return json.load(f)

def fmt(val):
    """Format AUROC as x100 with 1 decimal place, or --- if missing."""
    if val is None:
        return "  ---"
    return f"{val*100:5.1f}"

def get_auroc(result, split):
    """Safely get AUROC from a result block."""
    block = result.get(split)
    if block is None:
        return None
    if isinstance(block, dict):
        return block.get("auroc")
    return None

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--output", default="results/summary_table.json")
    return p.parse_args()

def main():
    args = parse_args()

    # Map of result files to probe names
    # Static probes results are a list, temporal are a single dict
    result_files = {
        "static":              os.path.join(args.results_dir, "static_probes.json"),
        "cnn_lookback":        os.path.join(args.results_dir, "temporal_cnn_lookback.json"),
        "cnn_all":             os.path.join(args.results_dir, "temporal_cnn_all.json"),
        "lstm_lookback":       os.path.join(args.results_dir, "temporal_lstm_lookback.json"),
        "lstm_all":            os.path.join(args.results_dir, "temporal_lstm_all.json"),
    }

    all_rows = []

    # ── Static probes (list of dicts) ──
    static_path = result_files["static"]
    if os.path.exists(static_path):
        static_results = load_json(static_path)
        for r in static_results:
            all_rows.append({
                "probe":    r["feature_set"],
                "type":     "static",
                "train":    get_auroc(r, "train"),
                "test":     get_auroc(r, "test"),
                "transfer": get_auroc(r, "transfer"),
                "n_train":  r.get("n_train", 0),
                "n_test":   r.get("n_test", 0),
                "n_transfer": r.get("n_transfer", 0),
                "test_ci95": r.get("test_auroc_ci95"),
            })
    else:
        print(f"  [missing] {static_path}")

    # ── Temporal probes (single dict each) ──
    temporal_files = {
        "cnn+lookback":         result_files["cnn_lookback"],
        "cnn+lookback_entropy": os.path.join(args.results_dir, "temporal_cnn_lookback_entropy.json"),
        "cnn+all":              result_files["cnn_all"],
        "lstm+lookback":        result_files["lstm_lookback"],
        "lstm+lookback_entropy": os.path.join(args.results_dir, "temporal_lstm_lookback_entropy.json"),
        "lstm+all":             result_files["lstm_all"],
    }
    for name, path in temporal_files.items():
        if os.path.exists(path):
            r = load_json(path)
            all_rows.append({
                "probe":    name,
                "type":     "temporal",
                "train":    get_auroc(r, "train"),
                "test":     get_auroc(r, "test"),
                "transfer": get_auroc(r, "transfer"),
                "n_train":  r.get("n_train", 0),
                "n_test":   r.get("n_test", 0),
                "n_transfer": r.get("n_transfer", 0),
                "test_ci95": r.get("test_auroc_ci95"),
            })
        else:
            print(f"  [missing] {path}")

    # ── Print summary table ──
    print()
    print("=" * 80)
    print("  SUMMARY TABLE — Lookback Lens Table 2 Format (AUROC ×100)")
    print(f"  Source=AggreFact (train=val, test=test) | Transfer=NQ")
    print("=" * 80)
    print(f"  {'Probe':<25} {'Type':<10} {'Train':>7} {'Test':>7} {'NQ':>7}")
    print("  " + "-" * 60)

    # Static first
    print("  [Static Probes]")
    for row in all_rows:
        if row["type"] == "static":
            ci = row.get("test_ci95")
            ci_str = f" [{ci[0]*100:.1f},{ci[1]*100:.1f}]" if ci else ""
            print(f"  {row['probe']:<25} {'static':<10} "
                  f"{fmt(row['train']):>7} "
                  f"{fmt(row['test']):>7}{ci_str} "
                  f"{fmt(row['transfer']):>7}")

    print()
    print("  [Temporal Probes]")
    for row in all_rows:
        if row["type"] == "temporal":
            ci = row.get("test_ci95")
            ci_str = f" [{ci[0]*100:.1f},{ci[1]*100:.1f}]" if ci else ""
            print(f"  {row['probe']:<25} {'temporal':<10} "
                  f"{fmt(row['train']):>7} "
                  f"{fmt(row['test']):>7}{ci_str} "
                  f"{fmt(row['transfer']):>7}")

    print("=" * 80)

    # ── Highlight key comparisons ──
    lookback_lens = next((r for r in all_rows if r["probe"] == "lookback_lens"), None)
    cnn_lookback  = next((r for r in all_rows if r["probe"] == "cnn+lookback"), None)
    cnn_all       = next((r for r in all_rows if r["probe"] == "cnn+all"), None)

    if lookback_lens and cnn_lookback:
        print()
        print("  Key comparisons (Test AUROC):")
        print(f"  Lookback Lens baseline : {fmt(lookback_lens['test'])}")
        print(f"  CNN + lookback         : {fmt(cnn_lookback['test'])}")
        if cnn_all:
            print(f"  CNN + all features     : {fmt(cnn_all['test'])}")
        delta = (cnn_lookback['test'] or 0) - (lookback_lens['test'] or 0)
        print(f"  Δ (CNN vs baseline)    : {delta*100:+.1f}")

    if lookback_lens and cnn_lookback:
        print()
        print("  Transfer to NQ (AUROC):")
        print(f"  Lookback Lens baseline : {fmt(lookback_lens['transfer'])}")
        print(f"  CNN + lookback         : {fmt(cnn_lookback['transfer'])}")
        if cnn_all:
            print(f"  CNN + all features     : {fmt(cnn_all['transfer'])}")

    # ── Save ──
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\n  Saved summary → {args.output}")

if __name__ == "__main__":
    main()