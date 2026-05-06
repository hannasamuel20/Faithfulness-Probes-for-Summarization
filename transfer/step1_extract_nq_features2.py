"""
Step 1: Extract Attention Features from Llama-2-7b-chat on NQ-Open
===================================================================

Extracts features for ALL examples in the NQ-Open jsonl — no annotations
needed at this stage. Labels are joined later in step03 after step02
(Claude annotation) has run.

Features extracted per generated token (single forward pass):
    lookback_ratio  (L, H, T)  — fraction of attention on input context
    attn_entropy    (L, H, T)  — Shannon entropy of each head's attn dist.
    chosen_prob     (T,)       — softmax prob of the generated token
    output_entropy  (T,)       — vocab-level Shannon entropy
    top_margin      (T,)       — prob(rank-1) - prob(rank-2)

Lookback Lens alignment
-----------------------
  - Prompt: #Document# / #Question# / #Answer#: format (paper Appendix B)
    Gold doc in the middle: neg[0], pos[0], neg[1]
  - context_length: attentions[0][0].shape[-1] - extra_prompt_length
    (key-seq length at step 0 minus response-prefix token count)
  - Attention index: layer_attn[0, :, -1, :] (last query = current token)
  - lookback uses mean() not sum() over positions
  - do_sample=False (greedy), max_new_tokens=256 (paper Section A)
  - attn_implementation="eager" required for output_attentions=True

Output .pt schema — dict with "config" and "examples" keys.
Each example:
    {
      "data_index":       int,
      "question":         str,
      "model_completion": str,
      "model_completion_ids": np.ndarray[T],
      "full_input_text":  str,
      "lookback_ratio":   Tensor[L, H, T],
      "attn_entropy":     Tensor[L, H, T],
      "chosen_prob":      Tensor[T],
      "output_entropy":   Tensor[T],
      "top_margin":       Tensor[T],
    }

Labels and split fields are added later by step03 when annotations are joined.

Usage
-----
    python step01_extract_attns.py \\
        --nq-path data/nq-open-10_total_documents_gold_at_4.jsonl \\
        --out-path features_nq.pt \\
        --hf-token YOUR_HF_TOKEN

    # Debug (5 examples):
    python step01_extract_attns.py \\
        --nq-path data/nq-open-10_total_documents_gold_at_4.jsonl \\
        --out-path debug.pt \\
        --debug
"""

import argparse
import datetime
import json
import logging
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
#  Prompt builder — identical to Lookback Lens NQ setup (paper Appendix B)
# ---------------------------------------------------------------------------

DEMO = (
    "Answer the question based on the information in the document. "
    "Explain your reasoning in the document step-by-step before "
    "providing the final answer.\n\n"
)
RESPONSE_PREFIX = "\n#Answer#:"


def build_prompt(question, ctxs):
    """
    Gold doc in the middle: neg[0], pos[0], neg[1].
    Returns (full_prompt_string, response_prefix_string).
    """
    pos_ctxs = [c for c in ctxs if c.get("hasanswer", False)]
    neg_ctxs = [c for c in ctxs if not c.get("hasanswer", False)]

    assert len(pos_ctxs) >= 1, "No positive context found."
    assert len(neg_ctxs) >= 2, "At least two negative contexts required."

    q = question.strip()
    q = q[0].upper() + q[1:]
    if not q.endswith("?"):
        q += "?"

    context = (
        "#Document#: "
        + neg_ctxs[0]["text"] + "\n"
        + pos_ctxs[0]["text"] + "\n"
        + neg_ctxs[1]["text"]
        + f"\n#Question#: {q}"
    )

    return DEMO + context + RESPONSE_PREFIX


# ---------------------------------------------------------------------------
#  Feature helpers
# ---------------------------------------------------------------------------

def compute_lookback_ratio(attn_head, context_len):
    """attn_head: Tensor[H, k_len] — last query position."""
    ctx_attn = attn_head[:, :context_len].mean(dim=-1)
    gen_attn = attn_head[:, context_len:].mean(dim=-1)
    ratio = ctx_attn / (ctx_attn + gen_attn + 1e-9)
    return torch.nan_to_num(ratio, nan=0.0, posinf=1.0, neginf=0.0)


def compute_attn_entropy(attn_head):
    """attn_head: Tensor[H, k_len] — last query position."""
    p = attn_head.clamp(min=1e-12)
    entropy = -(p * p.log()).sum(dim=-1)
    return torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)


def compute_logit_features(logits_step, chosen_id):
    """logits_step: Tensor[vocab] — pre-softmax logits for one step."""
    probs = torch.softmax(logits_step.float(), dim=-1)
    chosen_prob    = probs[chosen_id].item()
    p = probs.clamp(min=1e-12)
    output_entropy = -(p * p.log()).sum().item()
    top2           = torch.topk(probs, k=2).values
    top_margin     = (top2[0] - top2[1]).item()

    def _clean(v):
        return 0.0 if (np.isnan(v) or np.isinf(v)) else v

    return _clean(chosen_prob), _clean(output_entropy), _clean(top_margin)


# ---------------------------------------------------------------------------
#  Save helper
# ---------------------------------------------------------------------------

def save(results, args):
    config = {
        "model_name":       args.model_name,
        "data_type":        "nq_open",
        "nq_path":          args.nq_path,
        "max_new_tokens":   args.max_new_tokens,
        "do_sample":        False,
        "feature_families": [
            "lookback_ratio", "attn_entropy",
            "chosen_prob", "output_entropy", "top_margin",
        ],
        "created_at":  datetime.datetime.now().isoformat(),
        "n_examples":  len(results),
    }
    torch.save({"config": config, "examples": results}, args.out_path)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="meta-llama/Llama-2-7b-chat-hf")
    p.add_argument("--nq-path",    required=True,
                   help="NQ-Open jsonl (with ctxs field).")
    p.add_argument("--out-path",   required=True,
                   help="Output .pt path.")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--hf-token",   default=None)
    p.add_argument("--debug",      action="store_true",
                   help="Run on first 5 examples only.")
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--subsample",  type=int, default=None,
                   help="Keep every Nth example.")
    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        filename="logs/nq-extract.log",
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",   # required for output_attentions=True
    )
    model.eval()
    print("Model loaded.")

    # ── extra_prompt_length — identical to original Lookback Lens ─────────────
    # Token count of "\n#Answer#:" minus BOS. Subtracted from the attention
    # key-sequence length at step 0 to get the true context boundary.
    extra_prompt_length = len(tokenizer("\n#Answer#:")["input_ids"]) - 1
    print(f"extra_prompt_length = {extra_prompt_length}")

    # ── Load NQ data ──────────────────────────────────────────────────────────
    print(f"Loading NQ data from {args.nq_path}...")
    nq_examples = []
    with open(args.nq_path) as f:
        for line in f:
            nq_examples.append(json.loads(line))

    if args.subsample:
        nq_examples = [nq_examples[i] for i in range(len(nq_examples))
                       if i % args.subsample == 0]
    if args.debug:
        nq_examples = nq_examples[:5]
    print(f"  {len(nq_examples)} examples to process.")

    # ── Resume support ────────────────────────────────────────────────────────
    results = []
    done_indices = set()
    if os.path.exists(args.out_path):
        print(f"Resuming from {args.out_path}...")
        blob = torch.load(args.out_path, weights_only=False)
        results = blob["examples"] if isinstance(blob, dict) else blob
        done_indices = {r["data_index"] for r in results}
        print(f"  {len(done_indices)} already done.")

    # ── Extraction loop ───────────────────────────────────────────────────────
    with torch.no_grad():
        for i, ex in enumerate(tqdm(nq_examples)):
            if i in done_indices:
                continue

            question = ex["question"]

            try:
                prompt = build_prompt(question, ex["ctxs"])
            except AssertionError as e:
                logging.warning(f"Skipping idx {i}: {e}")
                continue

            inputs     = tokenizer(prompt, return_tensors="pt").to(model.device)
            prompt_len = inputs["input_ids"].shape[1]

            try:
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    output_attentions=True,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            except Exception as e:
                logging.error(f"Generation failed idx {i}: {e}")
                continue

            generated_ids  = out.sequences[0][prompt_len:]
            generated_text = tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )

            T = len(out.attentions)
            if T == 0:
                logging.warning(f"Empty attentions idx {i}")
                continue

            L = len(out.attentions[0])
            H = out.attentions[0][0].shape[1]

            # context_length — identical to original Lookback Lens
            # attentions[0][0].shape == (1, H, q_len, k_len)
            # k_len at step 0 = full prompt token count.
            context_length = out.attentions[0][0].shape[-1] - extra_prompt_length

            lookback_steps   = []
            entropy_steps    = []
            chosen_probs     = []
            output_entropies = []
            top_margins      = []

            for t in range(T):
                lr_per_layer  = []
                ent_per_layer = []

                for l in range(L):
                    # (1, H, q_len, k_len) → last query → (H, k_len)
                    attn = out.attentions[t][l][0, :, -1, :].float()
                    lr_per_layer.append(compute_lookback_ratio(attn, context_length))
                    ent_per_layer.append(compute_attn_entropy(attn))

                lookback_steps.append(torch.stack(lr_per_layer))   # (L, H)
                entropy_steps.append(torch.stack(ent_per_layer))   # (L, H)

                chosen_id = generated_ids[t].item()
                cp, oe, tm = compute_logit_features(out.scores[t][0], chosen_id)
                chosen_probs.append(cp)
                output_entropies.append(oe)
                top_margins.append(tm)

            # Stack to (L, H, T)
            lookback_ratio = torch.stack(lookback_steps, dim=-1).cpu()
            attn_entropy   = torch.stack(entropy_steps,  dim=-1).cpu()
            chosen_prob    = torch.tensor(chosen_probs,     dtype=torch.float32)
            output_entropy = torch.tensor(output_entropies, dtype=torch.float32)
            top_margin     = torch.tensor(top_margins,      dtype=torch.float32)

            # NaN guard
            lookback_ratio = torch.nan_to_num(lookback_ratio, nan=0.0)
            attn_entropy   = torch.nan_to_num(attn_entropy,   nan=0.0)
            chosen_prob    = torch.nan_to_num(chosen_prob,    nan=0.0)
            output_entropy = torch.nan_to_num(output_entropy, nan=0.0)
            top_margin     = torch.nan_to_num(top_margin,     nan=0.0)

            results.append({
                "data_index":           i,
                "question":             question,
                "model_completion":     generated_text,
                "model_completion_ids": generated_ids.cpu().numpy(),
                "full_input_text":      prompt,
                "lookback_ratio":       lookback_ratio,
                "attn_entropy":         attn_entropy,
                "chosen_prob":          chosen_prob,
                "output_entropy":       output_entropy,
                "top_margin":           top_margin,
            })
            done_indices.add(i)

            logging.info(
                f"[{i+1}/{len(nq_examples)}] "
                f"Q: {question[:60]} | Gen: {generated_text[:60]}"
            )

            if len(results) % args.checkpoint_every == 0:
                save(results, args)
                logging.info(f"Checkpoint at {len(results)} examples")

    save(results, args)
    print(f"\nDone. Saved {len(results)} examples → {args.out_path}")

    if results:
        r      = results[0]
        L, H, T = r["lookback_ratio"].shape
        print(f"  Layers={L}, Heads={H}, First example T={T}")


if __name__ == "__main__":
    main()