"""
Streaming loader for AggreFact CSV files (aggre_fact_sota.csv / aggre_fact_final.csv).

Expected CSV columns:
    doc        : source document text
    summary    : model-generated summary text
    label      : 1 = factually consistent, 0 = hallucinated
    dataset    : source benchmark (e.g. 'FRANK', 'XSumFaith', 'Polytope')
    model_name : summarization model that produced the summary
    cut        : 'val' or 'test'
    id         : unique example identifier
    origin     : origin benchmark name
"""

import pandas as pd


def load_aggrefact_csv(file_path, limit=None, split=None, chunksize=256):
    """
    Stream AggreFact from a local CSV file one row at a time.

    Args:
        file_path : path to the CSV (e.g. data/aggre_fact_sota.csv)
        limit     : only yield first N examples (for debugging)
        split     : if 'val' or 'test', filter to that split only
        chunksize : pandas read_csv chunk size

    Yields:
        dict — with keys: document, summary, label, dataset,
               model_name, split, example_id, data_index
    """
    yielded = 0
    data_index = 0
    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        if split is not None:
            chunk = chunk[chunk["cut"] == split]

        for _, row in chunk.iterrows():
            if limit is not None and yielded >= limit:
                return
            yield {
                "document": str(row["doc"]),
                "summary": str(row["summary"]),
                "label": int(row["label"]),             # 1=faithful, 0=hallucinated
                "dataset": str(row.get("dataset", "")),
                "model_name": str(row.get("model_name", "")),
                "split": str(row.get("cut", "")),
                "example_id": str(row.get("id", "")),
                "data_index": data_index,
            }
            yielded += 1
            data_index += 1
