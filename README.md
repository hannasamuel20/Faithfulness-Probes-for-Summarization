# Faithfulness Probes for Summarization

Detects hallucinations in LLM-generated summaries using white-box probes over LLaMA-2-7B-Chat's internal states. Per-token lookback ratios, attention entropies, and output-logit statistics are extracted and fed into static (logistic regression) and temporal (1D-CNN / BiLSTM) classifiers trained on the AggreFact benchmark.

## Setup

```bash
pip install -r requirements.txt
```

Requires a HuggingFace access token for `meta-llama/Llama-2-7b-chat-hf` (gated repo).

Place the AggreFact CSV at `data/aggre_fact_sota.csv`.

## Step 1 — Feature Extraction

Runs a teacher-forced forward pass through LLaMA-2-7B-Chat on each AggreFact example and saves per-token lookback ratios, attention entropies, and output-logit features to a `.pt` file.

```bash
python step01_extract_features.py \
    --data-type aggrefact \
    --data-path data/aggre_fact_sota.csv \
    --teacher-forcing \
    --output-path features_aggrefact_sota.pt \
    --auth-token hf_XXXXXXXXXXXXXXXXXXXXXXXX
```

## Step 2 — Static Probes

Trains L2-regularized logistic regression probes on time-aggregated (mean/min/max/var) features and reports AUROC, AUPRC, and F1 on the test split.

```bash
python step02_static_probes.py \
    --features features_aggrefact_sota.pt
```

## Step 3 — Temporal Probes

Trains a 1D-CNN or BiLSTM directly on the raw per-token feature sequences, capturing within-span dynamics that static averaging discards.

```bash
# 1D-CNN (default, best single model)
python step03_temporal_probes.py \
    --features features_aggrefact_sota.pt \
    --model cnn \
    --channels lookback

# BiLSTM over all channels
python step03_temporal_probes.py \
    --features features_aggrefact_sota.pt \
    --model lstm \
    --channels all
```
