"""
Model loading utility.

Handles:
    - Single GPU inference
    - Multi-GPU sharding via device_map="auto"
    - FP16 for GPU inference
    - Hugging Face auth tokens for gated models, e.g. LLaMA-2
    - MPS, Apple Silicon, support
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(model_name, device, num_gpus=4, max_memory=14, auth_token=None):
    """
    Load a causal LM and its tokenizer.

    Args:
        model_name : Hugging Face model identifier
                     e.g. "meta-llama/Llama-2-7b-chat-hf"
        device     : "cuda", "cpu", or "mps"
        num_gpus   : number of GPUs to shard across.
                     If num_gpus > 1, uses device_map="auto".
        max_memory : max GPU memory per device in GiB for multi-GPU.
                     For V100 16GB, use 14 to leave safety room.
        auth_token : Hugging Face token for gated repos.

    Returns:
        (model, tokenizer)
    """

    # Force eager attention.
    # SDPA / flash-attn can return None for output_attentions=True,
    # which breaks this project's feature extraction.
    kwargs = {
        "attn_implementation": "eager",
    }

    tok_kwargs = {"token": auth_token} if auth_token else {}

    if device == "cuda":
        kwargs["torch_dtype"] = torch.float16
        kwargs["low_cpu_mem_usage"] = True

        # Multi-GPU sharding.
        # This is important for LLaMA-2-7B on V100 16GB because
        # output_attentions=True needs extra memory during forward pass.
        if num_gpus and num_gpus > 1:
            kwargs["device_map"] = "auto"
            kwargs["max_memory"] = {
                i: f"{max_memory}GiB" for i in range(num_gpus)
            }

    elif device == "cpu":
        kwargs["torch_dtype"] = torch.float32
        kwargs["low_cpu_mem_usage"] = True

    elif device == "mps":
        kwargs["torch_dtype"] = torch.float16
        kwargs["low_cpu_mem_usage"] = True

    else:
        raise ValueError(
            f"Unsupported device: {device}. Expected 'cuda', 'cpu', or 'mps'."
        )

    mdl_kwargs = {**kwargs, **tok_kwargs}

    tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)

    # LLaMA tokenizers often do not define a pad token.
    # Using eos_token as pad_token is common for causal LMs.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, **mdl_kwargs)

    # Only move manually when NOT using device_map.
    # If device_map="auto" is active, Accelerate controls placement.
    if device == "cuda":
        if not (num_gpus and num_gpus > 1):
            model.cuda()
    elif device == "mps":
        model.to("mps")

    model.eval()

    print(f"  Model loaded on {device}.")
    if device == "cuda":
        if num_gpus and num_gpus > 1:
            print(f"  Multi-GPU sharding enabled across {num_gpus} GPUs.")
            print(f"  Max memory per GPU: {max_memory}GiB.")
        else:
            print("  Single-GPU mode enabled.")

    return model, tokenizer