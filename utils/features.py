"""
Per-token feature extraction from model internals.

Three feature families are extracted at each generated / forced token:

1. Lookback Ratio  (L layers x H heads)
   mean_attn_on_context / (mean_attn_on_context + mean_attn_on_new_tokens)

2. Attention Entropy  (L layers x H heads)
   Shannon entropy of each head's full attention distribution

3. Output Logit Features  (3 scalars)
   chosen_prob    — softmax probability the model assigned to the actual token
   output_entropy — Shannon entropy of the full vocabulary distribution
   top_margin     — logit gap between the top-1 and top-2 predictions
"""

import torch
import torch.nn.functional as F


def extract_step_features(attentions, logits, next_token_id, context_length):
    """
    Extract all three feature families from a single decoding position.

    Args
    ----
    attentions     : tuple of tensors, one per layer.
                     Shape per tensor: (batch, heads, query_len, key_len).
                     With KV-cache query_len == 1 after the first step.
    logits         : (1, vocab_size) raw logits for this position.
    next_token_id  : int — the token that was selected / forced.
    context_length : int — number of tokens in the prompt BEFORE the
                     response prefix (i.e. instruction + document).

    Returns
    -------
    lookback_ratio : (num_layers, num_heads)
    attn_entropy   : (num_layers, num_heads)
    logit_feats    : dict with chosen_prob, output_entropy, top_margin
    """
    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]

    lookback_ratio = torch.zeros(num_layers, num_heads)
    attn_entropy = torch.zeros(num_layers, num_heads)

    for l in range(num_layers):
        # (heads, key_len) — attention weights FROM the last query position.
        # Cast to fp32: fp16 underflows below ~6e-5 so clamp(1e-10) becomes
        # clamp(0) and log(0) → -inf → NaN in the entropy term.
        attn = attentions[l][0, :, -1, :].float()

        # ── Lookback ratio ──
        # .mean() normalises by segment length → size-independent ratio
        attn_ctx = attn[:, :context_length].mean(dim=-1)    # (heads,)
        attn_new = attn[:, context_length:].mean(dim=-1)     # (heads,)
        lookback_ratio[l] = attn_ctx / (attn_ctx + attn_new + 1e-10)

        # ── Attention entropy ──
        a = attn.clamp(min=1e-10)
        attn_entropy[l] = -(a * a.log()).sum(dim=-1)         # (heads,)

    # ── Logit features ──
    logits_f = logits[0].float()                              # (vocab_size,)
    probs = F.softmax(logits_f, dim=-1)

    chosen_prob = probs[next_token_id].item()

    p = probs.clamp(min=1e-10)
    output_entropy = -(p * p.log()).sum().item()

    top2_vals, _ = logits_f.topk(2)
    top_margin = (top2_vals[0] - top2_vals[1]).item()

    return lookback_ratio, attn_entropy, {
        "chosen_prob": chosen_prob,
        "output_entropy": output_entropy,
        "top_margin": top_margin,
    }
