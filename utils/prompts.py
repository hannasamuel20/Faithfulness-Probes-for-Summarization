"""
Prompt construction and document truncation helpers.

The prompt format follows Lookback Lens conventions:
    [instruction]\n\n[#Tag#: document text]\n#Summary#:

The RESPONSE_PREFIX ("\n#Summary#:") separates the context (instruction +
document) from the generation region.  Everything before it counts as
"context" for the lookback ratio computation.
"""

# ── Task instructions ────────────────────────────────────────────────

INSTRUCTIONS = {
    "xsum": "Generate a summary comprising of 1 sentence for the given article.\n\n",
    "cnndm": "Generate a summary based on the information in the document.\n\n",
    "aggrefact": "Generate a summary based on the information in the document.\n\n",
}

CONTEXT_TAGS = {
    "xsum": "#Article#:",
    "cnndm": "#Document#:",
    "aggrefact": "#Document#:",
}

RESPONSE_PREFIX = "\n#Summary#:"

# Stop words that signal the model is looping back into the prompt format
STOP_WORDS = ["#Document#:", "#Article#:", "#Question#:", "#Summary#:", "Q:"]


def build_prompt(document, data_type="aggrefact"):
    """
    Build the full prompt the model will see.

    Returns:
        (prompt_str, response_prefix_str)
    """
    instruction = INSTRUCTIONS.get(data_type, INSTRUCTIONS["aggrefact"])
    tag = CONTEXT_TAGS.get(data_type, "#Document#:")
    context = f"{tag} {document}"
    prompt = instruction + context + RESPONSE_PREFIX
    return prompt, RESPONSE_PREFIX


def truncate_document(document, tokenizer, max_doc_tokens=1800):
    """
    Truncate the document to at most *max_doc_tokens* tokens so the full
    prompt (instruction + document + response prefix + generation) fits
    within the model's context window (4096 for LLaMA-2).
    """
    doc_ids = tokenizer.encode(document, add_special_tokens=False)
    if len(doc_ids) > max_doc_tokens:
        doc_ids = doc_ids[:max_doc_tokens]
        document = tokenizer.decode(doc_ids, skip_special_tokens=True)
    return document
