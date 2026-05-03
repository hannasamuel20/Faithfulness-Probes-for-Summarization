"""
Generic streaming JSONL loaders.

The default loader preserves records and injects data_index. The XSum loader
normalizes common faithfulness-annotation schemas to the fields expected by
step01: document, summary, label, data_index, and optional example_id.
"""

import json


def load_jsonl(file_path, limit=None):
    """
    Stream data from a JSON-lines file one record at a time.

    Args:
        file_path : path to the .jsonl file
        limit     : only yield first N lines (for debugging)

    Yields:
        dict — each record with an injected 'data_index' field
    """
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                return
            item = json.loads(line)
            item["data_index"] = i
            yield item


def _normalize_label(item):
    """Return 1=faithful, 0=hallucinated from common JSONL label names."""
    for key in ("label", "faithful", "is_faithful", "factual", "is_factual"):
        if key not in item:
            continue
        value = item[key]
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"1", "true", "yes", "faithful", "factual", "consistent"}:
                return 1
            if v in {"0", "false", "no", "hallucinated", "unfaithful", "inconsistent"}:
                return 0

    for key in ("hallucinated", "has_hallucination", "is_hallucinated"):
        if key not in item:
            continue
        value = item[key]
        if isinstance(value, bool):
            return 0 if value else 1
        if isinstance(value, (int, float)):
            return 0 if int(value) else 1
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"1", "true", "yes", "hallucinated"}:
                return 0
            if v in {"0", "false", "no", "faithful"}:
                return 1

    raise ValueError(
        "XSum JSONL record is missing a recognized faithfulness label. "
        "Expected one of label/faithful/is_faithful/factual/is_factual or "
        "hallucinated/has_hallucination/is_hallucinated."
    )


def load_xsum_jsonl(file_path, limit=None):
    """
    Stream normalized XSum faithfulness JSONL records.

    Accepted text keys:
        document/doc/article/source, summary/summ/generated_summary
    Accepted id keys:
        example_id/id/bbcid/system_bbcid
    """
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                return
            item = json.loads(line)
            document = (
                item.get("document") or item.get("doc") or
                item.get("article") or item.get("source") or ""
            )
            summary = (
                item.get("summary") or item.get("summ") or
                item.get("generated_summary") or ""
            )
            if not document or not summary:
                raise ValueError(
                    f"XSum JSONL record {i} must include document and summary text."
                )

            out = {
                "document": str(document),
                "summary": str(summary),
                "label": _normalize_label(item),
                "dataset": str(item.get("dataset", "XSum")),
                "model_name": str(item.get("model_name", item.get("system", ""))),
                "split": str(item.get("split", "transfer")),
                "data_index": i,
            }
            example_id = (
                item.get("example_id") or item.get("id") or
                item.get("bbcid") or item.get("system_bbcid")
            )
            if example_id is not None:
                out["example_id"] = str(example_id)
            yield out
