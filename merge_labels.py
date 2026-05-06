"""
merge_labels.py
===============
Merges Claude annotations (anno_nq.jsonl) into the NQ feature file
(nq_features_full_v2.pt), adding 'label' and 'split' fields so the
probe scripts can consume it.

Also normalises key names so both AggreFact and NQ features use the
same field names expected by the probe code:
    logit_chosen_prob    -> chosen_prob
    logit_output_entropy -> output_entropy
    logit_top_margin     -> top_margin

Usage
-----
    python merge_labels.py \
        --features transfer/nq_features_full_v2.pt \
        --anno     anno_nq.jsonl \
        --out      transfer/nq_features_labeled.pt
"""

import argparse
import json
import torch

RENAME = {
    'logit_chosen_prob':    'chosen_prob',
    'logit_output_entropy': 'output_entropy',
    'logit_top_margin':     'top_margin',
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', required=True,
                   help='NQ .pt feature file from step01.')
    p.add_argument('--anno',     required=True,
                   help='Annotation jsonl from step02 (Claude).')
    p.add_argument('--out',      required=True,
                   help='Output .pt path.')
    args = p.parse_args()

    # ── Load features ─────────────────────────────────────────────────────────
    print(f"Loading features from {args.features}...")
    blob = torch.load(args.features, weights_only=False)
    examples = blob['examples'] if isinstance(blob, dict) else blob
    config   = blob.get('config', {}) if isinstance(blob, dict) else {}
    print(f"  {len(examples)} feature examples.")

    # ── Load annotations ──────────────────────────────────────────────────────
    print(f"Loading annotations from {args.anno}...")
    anno_by_index = {}
    with open(args.anno) as f:
        for line in f:
            d = json.loads(line)
            if d.get('decision') is not None:
                anno_by_index[d['index']] = d
    print(f"  {len(anno_by_index)} labeled annotations.")

    # ── Merge ─────────────────────────────────────────────────────────────────
    merged   = []
    skipped  = 0

    for ex in examples:
        idx = ex['data_index']

        # Drop examples with no annotation or undecided verdict
        if idx not in anno_by_index:
            skipped += 1
            continue

        anno  = anno_by_index[idx]
        label = 1 if anno['decision'] is True else 0

        # Copy and add label/split
        new_ex = dict(ex)
        new_ex['label'] = label
        new_ex['split'] = 'test'   # all NQ = transfer/eval only

        # Normalise key names to match what probe code expects
        for old_key, new_key in RENAME.items():
            if old_key in new_ex:
                new_ex[new_key] = new_ex.pop(old_key)

        merged.append(new_ex)

    n_faithful = sum(1 for e in merged if e['label'] == 1)
    n_hallu    = sum(1 for e in merged if e['label'] == 0)
    print(f"\n  Merged  : {len(merged)} examples")
    print(f"  Skipped : {skipped} (no annotation or undecided)")
    print(f"  Faithful: {n_faithful}  Hallucinated: {n_hallu}")

    # ── Save ──────────────────────────────────────────────────────────────────
    config['n_labeled'] = len(merged)
    torch.save({'config': config, 'examples': merged}, args.out)
    print(f"\nSaved → {args.out}")


if __name__ == '__main__':
    main()

