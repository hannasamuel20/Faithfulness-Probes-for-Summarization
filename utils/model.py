"""
Model loading utility.

Handles:
    - Single GPU / multi-GPU sharding via device_map="auto"
    - FP16 for GPU inference
    - HuggingFace auth tokens for gated models (e.g. LLaMA-2)
    - MPS (Apple Silicon) support
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(model_name, device, num_gpus=1, max_memory=45, auth_token=None):
    """
    Load a causal LM and its tokenizer.

    Args:
        model_name : HuggingFace model identifier
                     (e.g. "meta-llama/Llama-2-7b-chat-hf")
        device     : "cuda", "cpu", or "mps"
        num_gpus   : number of GPUs to shard across (>1 uses device_map="auto")
        max_memory  : max GPU memory per device in GiB (only for multi-GPU)
        auth_token : HuggingFace token for gated repos

    Returns:
        (model, tokenizer)
    """
    # Force eager attention — SDPA / flash-attn silently return None for
    # output_attentions=True, which breaks our feature extraction.
    kwargs = {"attn_implementation": "eager"}
    if device == "cuda":
        kwargs["torch_dtype"] = torch.float16
        if num_gpus > 1:
            kwargs["device_map"] = "auto"
            kwargs["max_memory"] = {
                i: f"{max_memory}GiB" for i in range(num_gpus)
            }

    tok_kwargs = {"token": auth_token} if auth_token else {}
    mdl_kwargs = {**kwargs, **tok_kwargs}

    tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_name, **mdl_kwargs)

    if device == "cuda" and num_gpus == 1:
        model.cuda()
    elif device == "mps":
        model.to("mps")

    model.eval()
    return model, tokenizer
