"""
Prepare Maynez et al. (2020) XSum faithfulness annotations as JSONL.

Downloads the Google Research annotation scores and joins them to the XSum
documents available through HuggingFace datasets. The output schema matches
the step01 XSum loader: document, summary, label, example_id, system.
"""

import argparse
import csv
import json
import os
import urllib.request

from datasets import load_dataset


EVAL_SCORES_URL = (
    "https://raw.githubusercontent.com/google-research-datasets/"
    "xsum_hallucination_annotations/master/eval_scores_xsum_summaries.csv"
)
FACTUALITY_URL = (
    "https://raw.githubusercontent.com/google-research-datasets/"
    "xsum_hallucination_annotations/master/factuality_annotations_xsum_summaries.csv"
)


def download(url, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        return
    urllib.request.urlretrieve(url, path)


def parse_system_bbcid(value):
    text = str(value)
    if "_" in text:
        system, bbcid = text.rsplit("_", 1)
        return system, bbcid
    return "", text


def canonical_system(value):
    mapping = {
        "bert_withckpt": "BERTS2S",
        "ptgen": "PtGen",
        "tconvs2s": "TConvS2S",
        "trans2s": "TranS2S",
    }
    return mapping.get(str(value), str(value))


def load_xsum_documents(cache_dir=None):
    docs = {}
    for split in ("train", "validation", "test"):
        ds = load_dataset("EdinburghNLP/xsum", split=split, cache_dir=cache_dir)
        for row in ds:
            docs[str(row["id"])] = row["document"]
    return docs


def parse_args():
    p = argparse.ArgumentParser(description="Prepare XSum faithfulness JSONL.")
    p.add_argument("--output", default="data/xsum/xsum_faithfulness.jsonl")
    p.add_argument("--cache-csv", default="data/xsum/eval_scores_xsum_summaries.csv")
    p.add_argument("--cache-factuality-csv",
                   default="data/xsum/factuality_annotations_xsum_summaries.csv")
    p.add_argument("--hf-cache-dir", default="data/xsum/hf_cache")
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--faithful-threshold", type=float, default=1.0,
                   help="Faithful score >= threshold maps to label=1.")
    p.add_argument("--include-partial", action="store_true",
                   help="Keep partially faithful examples below threshold as hallucinated.")
    return p.parse_args()


def main():
    args = parse_args()
    download(EVAL_SCORES_URL, args.cache_csv)
    download(FACTUALITY_URL, args.cache_factuality_csv)
    docs = load_xsum_documents(cache_dir=args.hf_cache_dir)

    summaries = {}
    with open(args.cache_factuality_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (canonical_system(row["system"]), str(row["bbcid"]))
            summaries[key] = row["summary"]

    rows = []
    with open(args.cache_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            system, bbcid = parse_system_bbcid(row.get("system_bbcid", ""))
            system = canonical_system(system)
            faithful = float(row["Faithful"])
            if not args.include_partial and 0.0 < faithful < args.faithful_threshold:
                continue
            if bbcid not in docs:
                continue
            summary = summaries.get((system, bbcid))
            if not summary:
                continue
            rows.append({
                "document": docs[bbcid],
                "summary": summary,
                "label": 1 if faithful >= args.faithful_threshold else 0,
                "example_id": f"{system}_{bbcid}" if system else bbcid,
                "bbcid": bbcid,
                "system": system,
                "dataset": "XSum",
                "split": "transfer",
                "faithful_score": faithful,
            })

    if args.limit and len(rows) > args.limit:
        faithful_rows = [r for r in rows if r["label"] == 1]
        hallucinated_rows = [r for r in rows if r["label"] == 0]
        rows = faithful_rows + hallucinated_rows[:max(args.limit - len(faithful_rows), 0)]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} examples -> {args.output}")


if __name__ == "__main__":
    main()
