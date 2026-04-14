"""
Generic streaming JSONL loader.

Works for any dataset stored as one JSON object per line
(e.g. CNN/DM, XSum local exports).  Each line must have at least
a 'document' field; 'summary' is optional.
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
