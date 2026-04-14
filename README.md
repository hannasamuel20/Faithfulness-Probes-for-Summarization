# Faithfulness Probes for Summarization 

## Step 1

Extracts three per-token feature families from LLaMA-2-7B-Chat while it
processes labeled summaries from AggreFact. The output `.pt` file is the
input to Step 2 (probe training).

**Feature families (per token position `t`):**
1. **Lookback ratio** — `(L=32, H=32, T)` — attention on context vs. new tokens.
2. **Attention entropy** — `(L=32, H=32, T)` — Shannon entropy of each head's attention distribution.
3. **Output logit features** — `(T,)` each: `chosen_prob`, `output_entropy`, `top_margin`.

## 1. Install

```bash
pip install -r requirements.txt
```

You need a HuggingFace access token for `meta-llama/Llama-2-7b-chat-hf`
(gated repo). 
<!-- Get one at https://huggingface.co/settings/tokens after -->
<!-- accepting the Meta license. -->

## 2. Data

Place the AggreFact CSV at `data/aggre_fact_sota.csv`. Required columns:
`doc`, `summary`, `label`, `dataset`, `model_name`, `cut`, `id`.

The loader streams the CSV row-by-row (no full materialization).

## 3. Run Step 1: teacher forcing on AggreFact-SOTA

The recommended mode: single forward pass per example with the ground-truth
summary appended to the prompt (fast, no sampling noise, uses existing labels).

```bash
python step01_extract_features.py \
    --data-type aggrefact \
    --data-path data/aggre_fact_sota.csv \
    --teacher-forcing \
    --output-path features_aggrefact_sota.pt \
    --auth-token hf_XXXXXXXXXXXXXXXXXXXXXXXX
```

Useful flags:

| Flag | Purpose |
|------|---------|
| `--split val` / `--split test` | Only process one AggreFact cut. |
| `--limit 10` | Only process the first 10 examples (quick smoke test). |
| `--max-doc-tokens 1800` | Truncate long documents to fit LLaMA-2's 4K context. |
| `--num-gpus 2 --max-memory 45` | Shard across multiple GPUs via `device_map="auto"`. |
| `--save-every 50` | Checkpoint frequency (examples). |

**Resume:** if `--output-path` already exists, the script loads it and
skips any `example_id`s already present. Safe to Ctrl-C mid-run.

### Quick debug run

```bash
python step01_extract_features.py \
    --data-type aggrefact \
    --data-path data/aggre_fact_sota.csv \
    --teacher-forcing --limit 10 \
    --output-path features_debug.pt \
    --auth-token hf_XXXX
```

## 4. Inspect the output

```bash
python inspect_features.py --features features_aggrefact_sota.pt
```

Prints structure, NaN/Inf checks, label distribution, source-dataset/model
breakdown, token-length stats, global feature statistics, and most
importantly, the faithful-vs-hallucinated comparison that verifies the
signal is going in the expected direction (lookback and chosen_prob should
be higher for faithful summaries; entropy should be higher for hallucinated).

## 5. Output format

`features_aggrefact_sota.pt` is a dict:

```python
{
  "config":   { model_name, data_type, teacher_forcing, max_doc_tokens, ... },
  "examples": [
      {
        "lookback_ratio":        Tensor(L, H, T),
        "attn_entropy":          Tensor(L, H, T),
        "logit_chosen_prob":     Tensor(T,),
        "logit_output_entropy":  Tensor(T,),
        "logit_top_margin":      Tensor(T,),
        "label":                 int (1=faithful, 0=hallucinated),
        "split":                 "val" | "test",
        "example_id":            str,        # stable across runs
        "source_dataset":        str,
        "source_model":          str,
        "context_length":        int,        # prompt tokens before "#Summary#:"
        "summary_text":          str,
        "data_index":            int,
      },
      ...
  ]
}
```

`example_id` is the stable key to use in Step 2 for reproducible train/val/test
splits. `config` records the extraction run so the `.pt` is self-describing.
