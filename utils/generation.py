"""
Two modes for running the model and extracting per-token features:

1. generate_and_extract()
   Token-by-token autoregressive generation with KV-cache.
   Use when you need the model to *write* a new summary.

2. teacher_force_and_extract()
   Single forward pass with a known summary appended to the prompt.
   Use when you already have summaries + labels (e.g. AggreFact).
   Much faster — one forward pass instead of T.
"""

import torch
import torch.nn.functional as F

from utils.features import extract_step_features
from utils.prompts import RESPONSE_PREFIX, STOP_WORDS


# ═══════════════════════════════════════════════════════════════════════
#  Sampling helpers
# ═══════════════════════════════════════════════════════════════════════

def sample_next_token(logits, temperature, top_p, top_k, do_sample):
    """Pick the next token given raw logits and sampling hyper-parameters."""
    if not do_sample or temperature < 1e-6:
        return logits.argmax(dim=-1, keepdim=True)

    scaled = logits / temperature

    # top-k filtering
    if top_k > 0:
        topk_vals, topk_idx = scaled.topk(top_k)
        filtered = torch.full_like(scaled, float("-inf"))
        filtered.scatter_(0, topk_idx, topk_vals)
        scaled = filtered

    # top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_idx = scaled.sort(descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cum_probs = sorted_probs.cumsum(dim=-1)
        # Drop tokens once the cumulative prob crosses top_p.
        # Shift by 1 so the first token that CROSSES the threshold
        # is still kept (always preserves the argmax).
        remove = cum_probs > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        sorted_logits[remove] = float("-inf")
        scaled = torch.full_like(scaled, float("-inf"))
        scaled.scatter_(0, sorted_idx, sorted_logits)

    return torch.multinomial(F.softmax(scaled, dim=-1), 1)


# ═══════════════════════════════════════════════════════════════════════
#  Stop-word detection
# ═══════════════════════════════════════════════════════════════════════

def encode_stop_words(tokenizer):
    """Pre-encode stop words into token-id sequences."""
    result = []
    for sw in STOP_WORDS:
        ids = tokenizer.encode(sw, add_special_tokens=False)
        result.append(ids)
    return result


def check_stop(generated_ids, stop_id_seqs):
    """Return (should_stop, num_ids_to_trim) if a stop sequence appeared."""
    for sw_ids in stop_id_seqs:
        n = len(sw_ids)
        if len(generated_ids) >= n and generated_ids[-n:] == sw_ids:
            return True, n
    return False, 0


# ═══════════════════════════════════════════════════════════════════════
#  Mode 1: Autoregressive generation + feature extraction
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_and_extract(
    model, tokenizer, prompt, device,
    max_new_tokens=128, temperature=1.0, top_p=0.95, top_k=50,
    do_sample=False, context_length=None,
):
    """
    Generate tokens one at a time with KV-cache, extracting features at
    every step.

    Args
    ----
    context_length : number of prompt tokens BEFORE the response prefix.
                     Computed automatically if None.

    Returns
    -------
    dict with feature tensors + generated text, or None if nothing generated.
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    total_prompt_len = input_ids.shape[1]

    if context_length is None:
        resp_prefix_ids = tokenizer.encode(RESPONSE_PREFIX, add_special_tokens=False)
        context_length = total_prompt_len - len(resp_prefix_ids)

    stop_id_seqs = encode_stop_words(tokenizer)

    generated_ids = []
    all_lookback = []
    all_entropy = []
    all_logit_feats = []

    past_key_values = None

    for step in range(max_new_tokens):
        if past_key_values is None:
            outputs = model(input_ids, use_cache=True, output_attentions=True)
        else:
            outputs = model(
                input_ids[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True,
            )

        past_key_values = outputs.past_key_values
        step_logits = outputs.logits[:, -1, :]       # (1, vocab)
        attentions = outputs.attentions

        # Select next token
        next_token = sample_next_token(
            step_logits[0], temperature, top_p, top_k, do_sample
        )
        next_id = next_token.item()
        generated_ids.append(next_id)

        # EOS check
        if next_id == tokenizer.eos_token_id:
            generated_ids.pop()
            break

        # Stop-word check
        stopped, trim = check_stop(generated_ids, stop_id_seqs)
        if stopped:
            generated_ids = generated_ids[:-trim]
            break

        # Extract features
        lr, ae, lf = extract_step_features(
            attentions, step_logits, next_id, context_length
        )
        all_lookback.append(lr)
        all_entropy.append(ae)
        all_logit_feats.append(lf)

        # Append token for the next KV-cache step
        input_ids = torch.cat(
            [input_ids, next_token.unsqueeze(0).to(device)], dim=-1
        )

        del attentions, outputs

    # Pack results
    n_tok = len(all_lookback)
    if n_tok == 0:
        return None

    return {
        "model_completion": tokenizer.decode(generated_ids, skip_special_tokens=True),
        "model_completion_ids": generated_ids,
        "context_length": context_length,
        "lookback_ratio": torch.stack(all_lookback, dim=-1),            # (L, H, T)
        "attn_entropy": torch.stack(all_entropy, dim=-1),               # (L, H, T)
        "logit_chosen_prob": torch.tensor([f["chosen_prob"] for f in all_logit_feats]),
        "logit_output_entropy": torch.tensor([f["output_entropy"] for f in all_logit_feats]),
        "logit_top_margin": torch.tensor([f["top_margin"] for f in all_logit_feats]),
    }


# ═══════════════════════════════════════════════════════════════════════
#  Mode 2: Teacher-forcing + feature extraction
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def teacher_force_and_extract(
    model, tokenizer, prompt, summary, device, context_length=None,
):
    """
    Single forward pass with the known summary appended to the prompt.
    Extracts the same three feature families at each summary-token position.

    Much faster than autoregressive generation (1 pass vs. T passes).
    Use when you already have labeled summaries (e.g. AggreFact).
    """
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    summary_ids = tokenizer.encode(" " + summary.strip(), add_special_tokens=False)

    full_ids = torch.tensor([prompt_ids + summary_ids], device=device)
    prompt_len = len(prompt_ids)
    n_summary_tokens = len(summary_ids)

    if context_length is None:
        resp_prefix_ids = tokenizer.encode(RESPONSE_PREFIX, add_special_tokens=False)
        context_length = prompt_len - len(resp_prefix_ids)

    # Single forward pass
    outputs = model(full_ids, output_attentions=True)
    logits = outputs.logits           # (1, seq_len, vocab)
    attentions = outputs.attentions   # tuple of (1, heads, seq_len, seq_len)

    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]

    all_lookback = torch.zeros(num_layers, num_heads, n_summary_tokens)
    all_entropy = torch.zeros(num_layers, num_heads, n_summary_tokens)
    all_chosen_prob = torch.zeros(n_summary_tokens)
    all_output_entropy = torch.zeros(n_summary_tokens)
    all_top_margin = torch.zeros(n_summary_tokens)

    for t in range(n_summary_tokens):
        pos = prompt_len + t
        token_id = summary_ids[t]

        for l in range(num_layers):
            attn = attentions[l][0, :, pos, :]        # (heads, key_len)

            # Lookback ratio
            attn_ctx = attn[:, :context_length].mean(dim=-1)
            attn_new = attn[:, context_length:pos + 1].mean(dim=-1)
            all_lookback[l, :, t] = attn_ctx / (attn_ctx + attn_new + 1e-10)

            # Attention entropy (only over positions up to current)
            a = attn[:, :pos + 1].clamp(min=1e-10)
            all_entropy[l, :, t] = -(a * a.log()).sum(dim=-1)

        # Logit features — logits at pos-1 predict token at pos
        logit_pos = pos - 1 if pos > 0 else 0
        step_logits = logits[0, logit_pos].float()
        probs = F.softmax(step_logits, dim=-1)

        all_chosen_prob[t] = probs[token_id].item()

        p = probs.clamp(min=1e-10)
        all_output_entropy[t] = -(p * p.log()).sum().item()

        top2, _ = step_logits.topk(2)
        all_top_margin[t] = (top2[0] - top2[1]).item()

    del attentions, outputs

    return {
        "model_completion": summary,
        "model_completion_ids": summary_ids,
        "context_length": context_length,
        "lookback_ratio": all_lookback,
        "attn_entropy": all_entropy,
        "logit_chosen_prob": all_chosen_prob,
        "logit_output_entropy": all_output_entropy,
        "logit_top_margin": all_top_margin,
    }
