"""
Data loaders for different dataset formats.

Each loader is a generator that yields dicts with at minimum:
    document   : str   — source document text
    summary    : str   — model-generated summary text
    data_index : int   — unique row index within this run
"""

from data_loaders.aggrefact import load_aggrefact_csv
from data_loaders.jsonl import load_jsonl


def load_data(data_type, data_path, limit=None, split=None):
    """Dispatch to the right loader based on data_type."""
    if data_type == "aggrefact":
        return load_aggrefact_csv(data_path, limit=limit, split=split)
    return load_jsonl(data_path, limit=limit)
