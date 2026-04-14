"""
Step 1: Feature Extraction for Faithfulness Probes
===================================================

Runs LLaMA-2-7B-Chat on source documents, extracting three feature
families at every token position:

  1. Lookback Ratio   — (L, H, T) per-layer per-head context vs. new-token ratio
  2. Attention Entropy — (L, H, T) per-layer per-head entropy of attention dist
  3. Output Logit Feats — (T,) x3: chosen_prob, output_entropy, top_margin

Two modes:
  --teacher-forcing : single forward pass on existing summaries (AggreFact)
  (default)         : autoregressive generation + feature extraction

Usage:

  # AggreFact SOTA — teacher forcing on labeled summaries (recommended)
  python step01_extract_features.py \\
      --data-type aggrefact \\
      --data-path data/aggre_fact_sota.csv \\
      --teacher-forcing \\
      --output-path features_aggrefact_sota.pt \\
      --auth-token hf_XXXX

  # Only val split
  python step01_extract_features.py \\
      --data-type aggrefact \\
      --data-path data/aggre_fact_sota.csv \\
      --teacher-forcing --split val \\
      --output-path features_sota_val.pt

  # Quick debug (first 10 examples)
  python step01_extract_features.py \\
      --data-type aggrefact \\
      --data-path data/aggre_fact_sota.csv \\
      --teacher-forcing --limit 10 \\
      --output-path features_debug.pt
"""

import os
import argparse
import datetime
import torch
from tqdm import tqdm

from data_loaders import load_data
from utils.model import load_model
from utils.prompts import build_prompt, truncate_document
from utils.generation import generate_and_extract, teacher_force_and_extract


def build_config(args):
    """Snapshot the feature-extraction run config for reproducibility."""
    return {
        "model_name": args.model_name,
        "data_type": args.data_type,
        "data_path": args.data_path,
        "split": args.split,
        "limit": args.limit,
        "max_doc_tokens": args.max_doc_tokens,
        "teacher_forcing": args.teacher_forcing,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "do_sample": args.do_sample,
        "feature_families": [
            "lookback_ratio", "attn_entropy",
            "logit_chosen_prob", "logit_output_entropy", "logit_top_margin",
        ],
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def save_results(config, examples, output_path):
    """Save as {'config': ..., 'examples': [...]}."""
    torch.save({"config": config, "examples": examples}, output_path)


def load_existing(output_path):
    """Load a previously-saved file. Tolerates both old list format and new dict format."""
    blob = torch.load(output_path, weights_only=False)
    if isinstance(blob, dict) and "examples" in blob:
        return blob.get("config"), blob["examples"]
    # Legacy: plain list
    return None, blob


def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 1: Extract lookback ratio, attention entropy, "
                    "and output logit features from LLaMA during summarization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Model ──
    g = parser.add_argument_group("Model")
    g.add_argument("--model-name", type=str,
                   default="meta-llama/Llama-2-7b-chat-hf")
    g.add_argument("--device", type=str, default="cuda",
                   choices=["cuda", "cpu", "mps"])
    g.add_argument("--num-gpus", type=int, default=1)
    g.add_argument("--max-memory", type=int, default=45,
                   help="Max GPU memory per device in GiB")
    g.add_argument("--auth-token", type=str, default=None,
                   help="HuggingFace auth token for gated models")

    # ── Data ──
    g = parser.add_argument_group("Data")
    g.add_argument("--data-type", type=str, default="aggrefact",
                   choices=["aggrefact", "cnndm", "xsum"])
    g.add_argument("--data-path", type=str, required=True,
                   help="Path to data file (e.g. data/aggre_fact_sota.csv)")
    g.add_argument("--split", type=str, default=None,
                   choices=["val", "test"],
                   help="Filter AggreFact to val or test split only")
    g.add_argument("--limit", type=int, default=None,
                   help="Only process first N examples (for debugging)")
    g.add_argument("--max-doc-tokens", type=int, default=1800,
                   help="Truncate documents longer than this many tokens")

    # ── Generation ──
    g = parser.add_argument_group("Generation")
    g.add_argument("--max-new-tokens", type=int, default=128)
    g.add_argument("--temperature", type=float, default=0.6)
    g.add_argument("--top-p", type=float, default=0.9)
    g.add_argument("--top-k", type=int, default=50)
    g.add_argument("--do-sample", action="store_true",
                   help="Use sampling; default is greedy decoding")
    g.add_argument("--teacher-forcing", action="store_true",
                   help="Single forward pass with known summary (no generation)")

    # ── Output ──
    g = parser.add_argument_group("Output")
    g.add_argument("--output-path", type=str, default="features_step01.pt")
    g.add_argument("--save-every", type=int, default=50,
                   help="Save checkpoint every N examples")

    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load data ──
    print(f"Streaming data: type={args.data_type}, path={args.data_path}")
    data = load_data(args.data_type, args.data_path,
                     limit=args.limit, split=args.split)

    # ── Load model ──
    print(f"Loading model: {args.model_name}")
    model, tokenizer = load_model(
        args.model_name, args.device,
        num_gpus=args.num_gpus,
        max_memory=args.max_memory,
        auth_token=args.auth_token,
    )
    print(f"  Model loaded on {args.device}.")

    # ── Config snapshot ──
    config = build_config(args)

    # ── Resume support ──
    # Dedup on example_id when available (stable across runs / filters),
    # otherwise fall back to data_index.
    results = []
    done_keys = set()
    if os.path.exists(args.output_path):
        print(f"Found existing output at {args.output_path}, resuming...")
        _, results = load_existing(args.output_path)
        for r in results:
            done_keys.add(r.get("example_id") or ("idx", r["data_index"]))
        print(f"  {len(done_keys)} examples already done.")

    # ── Main loop ──
    for sample in tqdm(data, desc="Extracting features"):
        idx = sample["data_index"]
        key = sample.get("example_id") or ("idx", idx)
        if key in done_keys:
            continue

        document = sample.get("document", sample.get("doc", ""))
        document = truncate_document(document, tokenizer, args.max_doc_tokens)
        prompt, _ = build_prompt(document, data_type=args.data_type)

        if args.teacher_forcing:
            summary = sample.get("summary", sample.get("summ", ""))
            if not summary:
                print(f"  Skipping index {idx}: no summary for teacher forcing")
                continue
            feats = teacher_force_and_extract(
                model, tokenizer, prompt, summary, args.device,
            )
        else:
            feats = generate_and_extract(
                model, tokenizer, prompt, args.device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                do_sample=args.do_sample,
            )

        if feats is None:
            print(f"  Skipping index {idx}: empty generation")
            continue

        # ── Attach metadata (no full_prompt — it bloats the .pt) ──
        feats["data_index"] = idx
        feats["summary_text"] = sample.get("summary", "")
        if "label" in sample:
            feats["label"] = sample["label"]
        if "dataset" in sample:
            feats["source_dataset"] = sample["dataset"]
        if "model_name" in sample:
            feats["source_model"] = sample["model_name"]
        if "split" in sample:
            feats["split"] = sample["split"]
        if "example_id" in sample:
            feats["example_id"] = sample["example_id"]

        results.append(feats)
        done_keys.add(key)

        # ── Periodic checkpoint ──
        if len(results) % args.save_every == 0:
            save_results(config, results, args.output_path)
            print(f"  Checkpoint: {len(results)} examples → {args.output_path}")

    # ── Final save ──
    save_results(config, results, args.output_path)
    print(f"\nDone. {len(results)} examples saved to {args.output_path}")

    # ── Summary statistics ──
    if results:
        r = results[0]
        L, H, T = r["lookback_ratio"].shape
        print(f"  Model: layers={L}, heads={H}")
        print(f"  First example: {T} tokens")
        print(f"  Feature shapes per example:")
        print(f"    lookback_ratio       : ({L}, {H}, T)")
        print(f"    attn_entropy         : ({L}, {H}, T)")
        print(f"    logit_chosen_prob    : (T,)")
        print(f"    logit_output_entropy : (T,)")
        print(f"    logit_top_margin     : (T,)")


if __name__ == "__main__":
    main()
