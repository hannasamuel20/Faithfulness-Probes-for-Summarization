# Phase 2 XSum Cross-Task Transfer Handoff

## 1. Quick Status Summary

The full Phase 2 AggreFact-to-XSum transfer experiments have now been run on BU SCC for `meta-llama/Llama-2-7b-chat-hf`. This branch includes the XSum JSONL preparation path, transfer loading, static and temporal probes, temporal lookback+entropy channels, seed variance, error analysis, qualitative figure generation, and result-file generation for JSON/CSV/PDF/PNG artifacts.

Headline result: temporal CNN over lookback is the best in-domain AggreFact model, reaching 83.1 AUROC. Cross-task transfer to XSum is weaker. The best XSum transfer result is the static `lookback_full` logistic-regression probe at 66.8 AUROC, while CNN+lookback reaches 62.9 AUROC. This suggests temporal lookback patterns are useful in-domain but may be more dataset-specific, while simpler aggregated lookback features transfer slightly better.

| Probe | AggreFact Test AUROC | XSum Transfer AUROC |
|---|---:|---:|
| random | 52.4 | 51.7 |
| output_prob_only LR | 71.7 | 60.5 |
| lookback_lens LR | 75.2 | 61.5 |
| lookback_full LR | 79.4 | 66.8 |
| attn_entropy_full LR | 80.2 | 61.8 |
| logits_full LR | 71.2 | 55.2 |
| lookback+entropy LR | 81.7 | 65.5 |
| lookback+logits LR | 79.5 | 66.5 |
| entropy+logits LR | 80.0 | 62.7 |
| all LR | 81.7 | 65.7 |
| CNN lookback | 83.1 | 62.9 |
| CNN all | 74.9 | 54.5 |
| LSTM lookback | 81.3 | 62.3 |
| CNN lookback+entropy | 68.5 | 54.2 |
| LSTM lookback+entropy | 68.6 | 58.4 |

The XSum transfer set is highly imbalanced, with only 26 faithful examples out of 1000. Treat AUROC as the primary transfer metric; AUPRC and F1 are less stable and should be interpreted carefully.

## 2. SCC Environment and Reproduction Commands

The completed runs used BU SCC with four GPUs:

```bash
qrsh -P cs505am -l gpus=4
module load python3/3.10.12
source .venv/bin/activate
```

LLaMA-2-7B-Chat is gated. Set `HF_TOKEN` in the environment before feature extraction; do not commit tokens or `.env` files.

Useful SCC environment:

```bash
set -a
source .env
set +a

export PROJECT=/projectnb/cs505am/students/<username>/Faithfulness-Probes-for-Summarization
export HF_HOME=$PROJECT/.hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Prepare XSum annotations:

```bash
python prepare_xsum_faithfulness.py \
  --output data/xsum/xsum_faithfulness.jsonl \
  --limit 1000 \
  --include-partial
```

Extract XSum features with the multi-GPU setup used for the real run:

```bash
python step01_extract_features.py \
  --data-type xsum \
  --data-path data/xsum/xsum_faithfulness.jsonl \
  --teacher-forcing \
  --output-path results/features_xsum.pt \
  --auth-token "$HF_TOKEN" \
  --num-gpus 4 \
  --max-memory 14
```

A single V100 16GB GPU is likely to run out of memory when extracting attentions from LLaMA-2-7B. The completed run used `--num-gpus 4 --max-memory 14`.

Inspect XSum features:

```bash
python inspect_features.py \
  --features results/features_xsum.pt
```

Run static probes:

```bash
python step02_static_probes.py \
  --features features_aggrefact_sota.pt \
  --transfer-features results/features_xsum.pt \
  --bootstrap 1000 \
  --show-top-features \
  --results-path results/step02_with_xsum.json
```

Run temporal probes:

```bash
python step03_temporal_probes.py \
  --features features_aggrefact_sota.pt \
  --transfer-features results/features_xsum.pt \
  --model cnn \
  --channels lookback \
  --save-model results/cnn_lookback.pt \
  --results-path results/step03_cnn_lookback_xsum.json

python step03_temporal_probes.py \
  --features features_aggrefact_sota.pt \
  --transfer-features results/features_xsum.pt \
  --model cnn \
  --channels all \
  --results-path results/step03_cnn_all_xsum.json

python step03_temporal_probes.py \
  --features features_aggrefact_sota.pt \
  --transfer-features results/features_xsum.pt \
  --model lstm \
  --channels lookback \
  --results-path results/step03_lstm_lookback_xsum.json

python step03_temporal_probes.py \
  --features features_aggrefact_sota.pt \
  --transfer-features results/features_xsum.pt \
  --model cnn \
  --channels lookback+entropy \
  --results-path results/step03_cnn_lookback_entropy_xsum.json

python step03_temporal_probes.py \
  --features features_aggrefact_sota.pt \
  --transfer-features results/features_xsum.pt \
  --model lstm \
  --channels lookback+entropy \
  --results-path results/step03_lstm_lookback_entropy_xsum.json
```

Run seed variance, error analysis, and qualitative figure generation:

```bash
python step04_seed_variance.py \
  --features features_aggrefact_sota.pt \
  --transfer-features results/features_xsum.pt \
  --seeds 0 1 2 \
  --results-path results/step04_seed_variance.json

python step05_error_analysis.py \
  --features features_aggrefact_sota.pt \
  --model-path results/cnn_lookback.pt

MPLCONFIGDIR=/tmp/mplconfig python step06_qualitative_figure.py \
  --features features_aggrefact_sota.pt \
  --model-path results/cnn_lookback.pt
```

## 3. Repository Map

```text
.
├── prepare_xsum_faithfulness.py      # prepares Maynez et al. XSum annotations
├── data_loaders/jsonl.py             # XSum JSONL loader
├── data_loaders/__init__.py          # routes --data-type xsum
├── step01_extract_features.py        # extracts LLaMA features; stores document_text
├── step02_static_probes.py           # static LR probes and XSum transfer
├── step03_temporal_probes.py         # CNN/LSTM probes, transfer, checkpoint metadata
├── step04_seed_variance.py           # selected temporal probes across seeds
├── step05_error_analysis.py          # dataset/type breakdowns and hard examples
├── step06_qualitative_figure.py      # lookback-trace PDF/PNG figure
├── utils/aggregation.py              # static/temporal feature builders
├── utils/evaluation.py               # metrics and threshold helpers
├── utils/model.py                    # model-loading helpers
├── data/xsum/xsum_faithfulness.jsonl
├── features_aggrefact_sota.pt        # generated feature tensor; do not commit
└── results/                          # experiment outputs
```

`.gitignore` should ignore large `.pt` files and repo-local Hugging Face cache while allowing JSON/CSV/PDF/PNG results:

```text
*.pt
results/*.pt
results/seed_variance/*.pt
data/xsum/hf_cache/
.env
```

## 4. XSum Data and Feature Inspection

XSum annotations come from Maynez et al. (2020), "On Faithfulness and Factuality in Abstractive Summarization." The prepared file is `data/xsum/xsum_faithfulness.jsonl`.

Prepared XSum summary:

| Field | Value |
|---|---:|
| Total examples | 1000 |
| Faithful examples | 26 |
| Hallucinated examples | 974 |
| Split | transfer |

Strict labeling rule:

```text
Faithful == 1.0 -> label 1, faithful
Faithful < 1.0  -> label 0, hallucinated
```

Expected JSONL schema:

```json
{"document": "...", "summary": "...", "label": 1, "example_id": "BERTS2S_34687720", "bbcid": "34687720", "system": "BERTS2S", "dataset": "XSum", "split": "transfer", "faithful_score": 1.0}
```

`label=1` means faithful and `label=0` means hallucinated. The strict rule creates a major imbalance and is the main transfer-set caveat.

Feature file: `results/features_xsum.pt`. Debug feature file: `results/features_xsum_debug.pt`. Do not commit either `.pt` file.

Inspection results for `results/features_xsum.pt`:

| Check | Result |
|---|---|
| Loaded examples | 1000 |
| Labeled examples | 1000/1000 |
| Labels | 974 hallucinated, 26 faithful |
| Source models | PtGen 345, TConvS2S 339, BERTS2S 316 |
| Splits | transfer 1000 |
| Summary token length | min 7, max 57, mean 27.7, median 27 |
| Context length | min 42, max 1822, mean 553.1, median 441 |
| Layer/head dimensions | 32 layers, 32 heads, consistent across all examples |
| NaN/Inf values | none |

Feature keys include `lookback_ratio`, `attn_entropy`, `logit_chosen_prob`, `logit_output_entropy`, `logit_top_margin`, `document_text`, `summary_text`, `model_completion`, `model_completion_ids`, `example_id`, `label`, `split`, `source_dataset`, and `source_model`.

Feature dimensions:

| Feature | Shape |
|---|---|
| `lookback_ratio` | `(32, 32, T)` |
| `attn_entropy` | `(32, 32, T)` |
| `logit_chosen_prob` | `(T,)` |
| `logit_output_entropy` | `(T,)` |
| `logit_top_margin` | `(T,)` |

Global feature statistics:

| Feature | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| `lookback_ratio_mean` | 0.4033 | 0.1214 | 0.1482 | 0.7384 |
| `attn_entropy_mean` | 2.3102 | 0.1988 | 1.5924 | 2.7774 |
| `chosen_prob_mean` | 0.4589 | 0.1054 | 0.1524 | 0.7285 |
| `output_entropy_mean` | 1.0681 | 0.4131 | 0.2858 | 3.1365 |
| `top_margin_mean` | 3.1187 | 0.7914 | 0.9821 | 5.6706 |

Faithful-vs-hallucinated sanity check:

| Feature | Faithful | Hallucinated | Gap | Expected Direction | Status |
|---|---:|---:|---:|---|---|
| `lookback_ratio_mean` | 0.3977 | 0.4035 | -0.0057 | higher for faithful | flipped/weak |
| `chosen_prob_mean` | 0.5004 | 0.4578 | +0.0427 | higher for faithful | expected |
| `top_margin_mean` | 3.5760 | 3.1065 | +0.4695 | higher for faithful | expected |
| `output_entropy_mean` | 0.8732 | 1.0733 | -0.2000 | lower for faithful | expected |
| `attn_entropy_mean` | 2.3438 | 2.3093 | +0.0346 | lower for faithful | flipped/weak |

Interpretation: on XSum, simple mean lookback and attention entropy signals are weak or flipped, while logit confidence features show the expected direction. This helps explain why XSum transfer is weaker than AggreFact in-domain performance.

## 5. Static Step 2 Results

Training/evaluation setup:

| Item | Value |
|---|---|
| Train | AggreFact `cut=val` |
| Test | AggreFact `cut=test` |
| Transfer | XSum |
| Classifier | Logistic Regression, `C=1.0` |
| Output | `results/step02_with_xsum.json` |

| Probe | AggreFact Test AUROC | XSum Transfer AUROC |
|---|---:|---:|
| random | 52.4 | 51.7 |
| output_prob_only LR | 71.7 | 60.5 |
| lookback_lens LR | 75.2 | 61.5 |
| lookback_full LR | 79.4 | 66.8 |
| attn_entropy_full LR | 80.2 | 61.8 |
| logits_full LR | 71.2 | 55.2 |
| lookback+entropy LR | 81.7 | 65.5 |
| lookback+logits LR | 79.5 | 66.5 |
| entropy+logits LR | 80.0 | 62.7 |
| all LR | 81.7 | 65.7 |

Best in-domain static result: `lookback+entropy LR` and `all LR`, both around 81.7 AUROC. Best XSum static transfer result: `lookback_full LR`, 66.8 AUROC.

Static all-features improves over `lookback_lens` on AggreFact from 75.2 to 81.7 AUROC, a +6.5 gain. Transfer remains weaker than in-domain performance but above random.

The top feature printout for the `all` probe included many attention-entropy and lookback variance/min/max features. Strong coefficients included `attn_entropy/var/L9-H23`, `attn_entropy/min/L11-H7`, `lookback/var/L22-H27`, `attn_entropy/min/L25-H25`, and `lookback/var/L21-H20`.

## 6. Temporal Step 3 Results

Output files:

```text
results/step03_cnn_lookback_xsum.json
results/step03_cnn_all_xsum.json
results/step03_lstm_lookback_xsum.json
results/step03_cnn_lookback_entropy_xsum.json
results/step03_lstm_lookback_entropy_xsum.json
```

| Probe | AggreFact Test AUROC | XSum Transfer AUROC |
|---|---:|---:|
| CNN lookback | 83.1 | 62.9 |
| CNN all | 74.9 | 54.5 |
| LSTM lookback | 81.3 | 62.3 |
| CNN lookback+entropy | 68.5 | 54.2 |
| LSTM lookback+entropy | 68.6 | 58.4 |

Best in-domain temporal model: CNN lookback, 83.1 AggreFact Test AUROC. Best temporal XSum transfer: CNN lookback, 62.9 XSum Transfer AUROC.

Temporal CNN over lookback improves in-domain AggreFact performance relative to static `lookback_lens`. However, temporal models do not transfer as well as the best static model: best static transfer is `lookback_full LR` at 66.8 AUROC, while best temporal transfer is CNN lookback at 62.9 AUROC. Adding all channels or lookback+entropy hurt temporal performance in these runs, suggesting the temporal CNN may learn dataset-specific AggreFact temporal patterns rather than fully dataset-general factuality signals.

## 7. Seed Variance

Command output: `results/step04_seed_variance.json`.

Per-seed CNN+lookback:

| Seed | AggreFact Test AUROC | XSum Transfer AUROC |
|---:|---:|---:|
| 0 | 0.8309 | 0.6292 |
| 1 | 0.8168 | 0.6220 |
| 2 | 0.8065 | 0.6826 |

Per-seed LSTM+all:

| Seed | AggreFact Test AUROC | XSum Transfer AUROC |
|---:|---:|---:|
| 0 | 0.7601 | 0.5849 |
| 1 | 0.7238 | 0.4957 |
| 2 | 0.7836 | 0.6674 |

Mean/std summary:

| Probe | AggreFact Test AUROC | XSum Transfer AUROC |
|---|---:|---:|
| CNN+lookback | 0.818 +/- 0.012 | 0.645 +/- 0.033 |
| LSTM+all | 0.756 +/- 0.030 | 0.583 +/- 0.086 |

CNN+lookback is stronger and more stable than LSTM+all in-domain. XSum transfer is more variable across seeds, especially for LSTM+all. The best single XSum transfer from seed variance was CNN+lookback seed 2 at 0.6826, but the average remains around 0.645.

## 8. Error Analysis

Command output files:

```text
results/error_breakdown_by_dataset.csv
results/error_breakdown_by_type.csv
results/hard_examples.json
```

Dataset-level error analysis:

| Source dataset | n | faithful | hallucinated | AUROC | AUPRC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| CLIFF | 147 | 89 | 58 | 0.8479 | 0.8910 | 0.1685 | 0.2586 |
| Cao22 | 235 | 135 | 100 | 0.8679 | 0.8912 | 0.3926 | 0.1200 |
| FRANK | 174 | 143 | 31 | 0.7178 | 0.9014 | 0.0280 | 0.8065 |
| Goyal21 | 50 | 21 | 29 | 0.7077 | 0.5807 | 0.7143 | 0.2414 |
| Polytope | 34 | 34 | 0 | N/A | N/A | 0.0294 | N/A |
| Wang20 | 119 | 59 | 60 | 0.6907 | 0.6615 | 0.5424 | 0.2000 |

Polytope has only faithful examples in this subset, so AUROC/AUPRC are undefined.

Error-type recall:

| Error type | n | caught | recall |
|---|---:|---:|---:|
| extrinsic-NP | 50 | 36 | 0.72 |
| extrinsic-entire_sent | 2 | 2 | 1.00 |
| extrinsic-predicate | 15 | 8 | 0.5333 |
| intrinsic-NP | 30 | 10 | 0.3333 |
| intrinsic-entire_sent | 2 | 2 | 1.00 |
| intrinsic-predicate | 6 | 3 | 0.50 |

The CNN catches extrinsic noun-phrase hallucinations better than intrinsic noun-phrase hallucinations. Performance varies strongly by AggreFact source dataset: strongest on Cao22 and CLIFF, weaker on Wang20, Goyal21, and FRANK.

`results/hard_examples.json` contains 15 examples: 5 confident false positives, 5 confident false negatives, and 5 cases where CNN saves a static miss. Fields include `kind`, `example_id`, `source_dataset`, `source_model`, `label`, `cnn_faithful_score`, `cnn_hallucination_probability`, `threshold`, `document`, `summary`, and `lookback_trace`.

Some `document` fields may be empty because the original AggreFact feature file was created before `document_text` saving was added to Step 1.

## 9. Qualitative Figure

Generated files:

```text
results/lookback_trace_figure.pdf
results/lookback_trace_figure.png
```

The figure shows one confident correct faithful example and one confident correct hallucinated example from AggreFact test. It plots per-token lookback ratio averaged over all LLaMA-2 layers and attention heads. The final cleaned figure places summary captions outside the plot area.

Suggested report caption: Per-token lookback ratio averaged over all LLaMA-2 layers and attention heads for one correctly classified faithful summary and one correctly classified hallucinated summary. The CNN probe assigns low hallucination probability to the faithful example and high hallucination probability to the hallucinated example.

## 10. Results Files

Small result artifacts that can be committed:

```text
results/error_breakdown_by_dataset.csv
results/error_breakdown_by_type.csv
results/hard_examples.json
results/lookback_trace_figure.pdf
results/lookback_trace_figure.png
results/step02_with_xsum.json
results/step03_cnn_all_xsum.json
results/step03_cnn_lookback_entropy_xsum.json
results/step03_cnn_lookback_xsum.json
results/step03_lstm_lookback_entropy_xsum.json
results/step03_lstm_lookback_xsum.json
results/step04_seed_variance.json
results/seed_variance/cnn+lookback_seed0.json
results/seed_variance/cnn+lookback_seed1.json
results/seed_variance/cnn+lookback_seed2.json
results/seed_variance/lstm+all_seed0.json
results/seed_variance/lstm+all_seed1.json
results/seed_variance/lstm+all_seed2.json
```

Large generated artifacts that should not be committed:

```text
results/features_xsum.pt
results/features_xsum_debug.pt
results/cnn_lookback.pt
results/seed_variance/cnn+lookback_seed0.pt
results/seed_variance/cnn+lookback_seed1.pt
results/seed_variance/cnn+lookback_seed2.pt
results/seed_variance/lstm+all_seed0.pt
results/seed_variance/lstm+all_seed1.pt
results/seed_variance/lstm+all_seed2.pt
```

## 11. Caveats

The XSum transfer set is extremely imbalanced under the strict `Faithful == 1.0` rule: 26 faithful and 974 hallucinated examples. AUROC is the most reliable transfer metric here; AUPRC and F1 are much less stable.

The project uses teacher forcing: probes score the provided summary tokens under the source document rather than first generating a new summary and judging that generation. This differs from Chuang et al.'s original free-form setup and should be mentioned when interpreting results.

LLaMA-2-7B-Chat is gated on Hugging Face. If feature extraction fails with a 401/403, confirm model access and ensure `HF_TOKEN` is set. Do not expose the token in docs, commits, logs, or `.env` files.

V100 memory is tight for LLaMA-2-7B attentions. The successful SCC XSum extraction used four GPUs with `--num-gpus 4 --max-memory 14`; a single 16GB V100 is likely to OOM.

Do not commit `.pt` files. Feature tensors and model checkpoints are large generated artifacts and may contain raw examples/model outputs.

Matplotlib may try to write font caches under the home directory. If figure generation hangs or warns about cache permissions, run it with `MPLCONFIGDIR=/tmp/mplconfig`.

Old CNN checkpoints saved as a bare `state_dict` cannot be reloaded by Step 5/6 because architecture/channel metadata is missing. Re-run Step 3 with `--save-model`.

## 12. Future Work

Layer/head analysis could identify whether transfer failure is localized to specific attention heads. This is useful for interpretation now that the full XSum extraction and transfer runs are complete.

Compare validation-selected F1 against test-selected F1 in the report. AUROC/AUPRC are unchanged, but any F1 rows should cite the unbiased validation threshold.

Investigate why mean lookback and attention entropy are weak/flipped on XSum while logit confidence features retain the expected direction.

Classifier-guided decoding from the proposal remains a stretch goal. It is lower priority than interpreting the completed transfer and error-analysis results .

