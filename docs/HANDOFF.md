# Phase 2 XSum Cross-Task Transfer Handoff


## 1. Quick Status Summary

This branch adds the missing XSum transfer plumbing, an acquired 1000-example XSum JSONL subset, temporal `lookback+entropy` channels, seed-variance runs, CNN error analysis, and the final lookback-trace figure script. Step 2 and Step 3 now select F1 thresholds on a held-out validation slice instead of the test set, while still reporting test-selected `f1_opt` for transparency. I verified the new code paths with synthetic feature files in `/private/tmp/fps`; full AggreFact/XSum probe numbers were not run here because this checkout does not contain `features_aggrefact_sota.pt`, and XSum/LLaMA extraction requires HuggingFace access to gated `meta-llama/Llama-2-7b-chat-hf`.

| Probe | AggreFact Test AUROC | XSum Transfer AUROC | Status |
|---|---:|---:|---|
| random | pending | pending | run Step 2 |
| lookback_lens LR | expected ~0.833 | pending | run Step 2 |
| lookback+entropy LR | expected ~0.871 | pending | run Step 2 |
| all LR | pending | pending | run Step 2 |
| CNN lookback | expected ~0.879 | pending | run Step 3 |
| CNN lookback+entropy | pending | pending | run Step 3 |
| CNN all | pending | pending | run Step 3 |
| LSTM lookback | pending | pending | run Step 3 |
| LSTM lookback+entropy | pending | pending | run Step 3 |
| LSTM all | pending | pending | run Step 3 |

## 2. How to Reproduce Every Result

Set the HuggingFace token first. LLaMA-2-7B-Chat is gated, so your HF account must have access.

```bash
export HF_TOKEN=hf_...
```

Prepare XSum annotations:

```bash
python prepare_xsum_faithfulness.py \
  --output data/xsum/xsum_faithfulness.jsonl \
  --limit 1000 \
  --include-partial
```

Expected output: `data/xsum/xsum_faithfulness.jsonl` with 1000 examples. The generated file in this checkout has 26 faithful and 974 hallucinated examples under the strict `Faithful == 1.0` labeling rule. Runtime is mostly HuggingFace XSum download time: about 5-20 minutes depending on cache/network.

Extract AggreFact features if needed:

```bash
python step01_extract_features.py \
  --data-type aggrefact \
  --data-path data/aggre_fact_sota.csv \
  --teacher-forcing \
  --output-path results/features_aggrefact_sota.pt \
  --auth-token "$HF_TOKEN"
```

Extract a tiny XSum debug subset first:

```bash
python step01_extract_features.py \
  --data-type xsum \
  --data-path data/xsum/xsum_faithfulness.jsonl \
  --teacher-forcing \
  --limit 20 \
  --output-path results/features_xsum_debug.pt \
  --auth-token "$HF_TOKEN"
```

Then extract the full XSum subset:

```bash
python step01_extract_features.py \
  --data-type xsum \
  --data-path data/xsum/xsum_faithfulness.jsonl \
  --teacher-forcing \
  --output-path results/features_xsum.pt \
  --auth-token "$HF_TOKEN"
```

Feature extraction is the slow step: expect hours on a single GPU for 500-1000 examples. Outputs are resumable `.pt` files.

Inspect XSum signal direction:

```bash
python inspect_features.py --features results/features_xsum.pt
```

Static probes with transfer:

```bash
python step02_static_probes.py \
  --features results/features_aggrefact_sota.pt \
  --transfer-features results/features_xsum.pt \
  --bootstrap 1000 \
  --show-top-features \
  --results-path results/step02_with_xsum.json
```

Temporal headline probes:

```bash
python step03_temporal_probes.py \
  --features results/features_aggrefact_sota.pt \
  --transfer-features results/features_xsum.pt \
  --model cnn --channels lookback \
  --save-model results/cnn_lookback.pt \
  --results-path results/step03_cnn_lookback_xsum.json

python step03_temporal_probes.py \
  --features results/features_aggrefact_sota.pt \
  --transfer-features results/features_xsum.pt \
  --model lstm --channels all \
  --save-model results/lstm_all.pt \
  --results-path results/step03_lstm_all_xsum.json
```

Fill the missing temporal grid:

```bash
python step03_temporal_probes.py --features results/features_aggrefact_sota.pt --transfer-features results/features_xsum.pt --model cnn --channels all --results-path results/step03_cnn_all_xsum.json
python step03_temporal_probes.py --features results/features_aggrefact_sota.pt --transfer-features results/features_xsum.pt --model lstm --channels lookback --results-path results/step03_lstm_lookback_xsum.json
python step03_temporal_probes.py --features results/features_aggrefact_sota.pt --transfer-features results/features_xsum.pt --model cnn --channels lookback+entropy --results-path results/step03_cnn_lookback_entropy_xsum.json
python step03_temporal_probes.py --features results/features_aggrefact_sota.pt --transfer-features results/features_xsum.pt --model lstm --channels lookback+entropy --results-path results/step03_lstm_lookback_entropy_xsum.json
```

Seed variance:

```bash
python step04_seed_variance.py \
  --features results/features_aggrefact_sota.pt \
  --transfer-features results/features_xsum.pt \
  --seeds 0 1 2 \
  --results-path results/step04_seed_variance.json
```

Error analysis and figure:

```bash
python step05_error_analysis.py \
  --features results/features_aggrefact_sota.pt \
  --model-path results/cnn_lookback.pt

MPLCONFIGDIR=/tmp/mplconfig python step06_qualitative_figure.py \
  --features results/features_aggrefact_sota.pt \
  --model-path results/cnn_lookback.pt
```

Expected outputs: `results/error_breakdown_by_dataset.csv`, `results/error_breakdown_by_type.csv`, `results/hard_examples.json`, `results/lookback_trace_figure.pdf`, and `results/lookback_trace_figure.png`.

## 3. Repository Map

```text
.
├── step01_extract_features.py        # LLaMA feature extraction; now stores document_text
├── step02_static_probes.py           # LR probes + XSum transfer + val-selected threshold
├── step03_temporal_probes.py         # CNN/LSTM probes + transfer + checkpoint metadata
├── step04_seed_variance.py           # new: 3-seed temporal aggregation wrapper
├── step05_error_analysis.py          # new: dataset/type breakdowns and hard examples
├── step06_qualitative_figure.py      # new: paper lookback-trace figure
├── prepare_xsum_faithfulness.py      # new: prepares Maynez XSum JSONL
├── inspect_features.py               # sanity checks; section 7 is the key signal check
├── data_loaders/
│   ├── aggrefact.py                  # AggreFact CSV streaming loader
│   ├── jsonl.py                      # generic JSONL + normalized XSum JSONL loader
│   └── __init__.py                   # routes data-type=xsum to XSum loader
├── utils/
│   ├── aggregation.py                # static and temporal feature builders
│   ├── evaluation.py                 # metrics and threshold helpers
│   └── ...
├── models/temporal_probes.py         # CNN/LSTM modules
├── data/error_type_mapping/          # AggreFact fine-grained error labels
├── data/xsum/                        # XSum JSONL and downloaded annotation CSVs
├── results/                          # experiment outputs, checkpoints, figures
└── docs/HANDOFF.md                   # this document
```

`.gitignore` ignores `*.pt` because feature tensors and model checkpoints are large. JSON/CSV/PDF/PNG results are not ignored.

## 4. Data Provenance

XSum annotations come from Maynez et al. (2020), "On Faithfulness and Factuality in Abstractive Summarization", ACL 2020: https://github.com/google-research-datasets/xsum_hallucination_annotations. The repository license is CC BY 4.0 per its README. The preparation script downloads `eval_scores_xsum_summaries.csv` and `factuality_annotations_xsum_summaries.csv`, joins summaries to XSum documents from HuggingFace `EdinburghNLP/xsum`, and writes up to 1000 examples. The generated subset in this checkout was produced with `--include-partial`, so any `Faithful < 1.0` score maps to hallucinated. The strict rule is highly imbalanced: 26 faithful and 974 hallucinated examples.

Expected JSONL schema:

```json
{"document": "...", "summary": "...", "label": 1, "example_id": "BERTS2S_34687720", "bbcid": "34687720", "system": "BERTS2S", "dataset": "XSum", "split": "transfer", "faithful_score": 1.0}
```

`label=1` means faithful and `label=0` means hallucinated. Without `--include-partial`, partial faithfulness scores between 0 and 1 are skipped and the file is much smaller.

## 5. How the Pipeline Fits Together

```text
raw AggreFact CSV / XSum JSONL
        |
        v
step01_extract_features.py
        |
        v
feature .pt files
        |
        +--> step02_static_probes.py ----+
        |                                |
        +--> step03_temporal_probes.py --+--> cross-task JSON results
                         |
                         +--> saved CNN checkpoint
                                  |
                                  +--> step05_error_analysis.py
                                  |
                                  +--> step06_qualitative_figure.py
```

Step 1 extracts per-token white-box signals under teacher forcing. Step 2 collapses features into span vectors and trains logistic regression probes. Step 3 trains temporal probes over token sequences. Step 4 repeats the headline temporal runs over seeds. Step 5 reuses the saved CNN for breakdowns and hard examples. Step 6 makes the final qualitative figure.

## 6. Key Implementation Decisions

Teacher forcing is used to match the existing AggreFact protocol: the probe scores the provided summary tokens under the source document, instead of first generating a new summary and judging that generation. This differs from Chuang et al.'s original free-form setup and should be described when interpreting AUROC gains.

AggreFact convention is train on `cut=val` and evaluate on `cut=test`; this is odd naming but intentional. Step 3 still carves an internal validation slice from `cut=val` for early stopping and threshold selection.

High-dimensional probes can get train AUROC near 1.000. That is not automatically a bug; the relevant checks are held-out AggreFact AUROC, XSum transfer AUROC, and seed variance.

F1 thresholds are now selected on a validation slice and applied to test/transfer. Results JSONs still include `f1_opt` and `thr_opt`, which are the test-selected values, plus `f1_at_val_threshold` and `val_threshold`.

Temporal `lookback+entropy` is now a first-class channel option implemented in `utils/aggregation.py` and exposed through `CHANNEL_BUILDERS`.

## 7. Known Caveats

The teacher-forcing protocol differs from Chuang et al.'s original free-form-then-judge setup; this affects how the reported +4.6 AUROC gap should be interpreted.

LLaMA-2-7B is gated on HuggingFace. If feature extraction fails with a 401/403, request model access and set `HF_TOKEN`.

Transfer examples originally would have been dropped by Step 2/3 if they lacked `split`; this is fixed by loading transfer files with `require_split=False`.

Matplotlib may try to write font caches under the home directory. If figure generation hangs or warns about cache permissions, run it with `MPLCONFIGDIR=/tmp/mplconfig`.

Old CNN checkpoints saved as a bare `state_dict` cannot be reloaded by Step 5/6 because architecture/channel metadata is missing. Re-run Step 3 with `--save-model`.

## 8. Questions Still Open / Future Work

Run the full XSum extraction and fill the transfer table. Highest severity because the main research question is not complete until those numbers exist.

Compare val-selected F1 against test-selected F1 in the report. Medium severity; AUROC/AUPRC are unchanged, but F1 rows should cite the unbiased threshold.

Layer/head analysis could identify whether transfer failure, if any, is localized to specific attention heads. Medium severity and useful for interpretation.

Classifier-guided decoding from the proposal remains a stretch goal. Low severity for the current report unless all transfer/error analysis finishes early.

