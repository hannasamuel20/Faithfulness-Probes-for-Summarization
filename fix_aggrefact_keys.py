"""
fix_aggrefact_keys.py
=====================
Renames logit_* keys in the AggreFact feature file to match what the
probe code expects (chosen_prob, output_entropy, top_margin).

Creates a new file — does not modify the original.

Usage
-----
    python fix_aggrefact_keys.py \
        --features /path/to/features_aggrefact_final.pt \
        --out      features_aggrefact_fixed.pt
"""

import argparse
import torch

RENAME = {
    'logit_chosen_prob':    'chosen_prob',
    'logit_output_entropy': 'output_entropy',
    'logit_top_margin':     'top_margin',
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', required=True)
    p.add_argument('--out',      required=True)
    args = p.parse_args()

    print(f"Loading {args.features}...")
    blob     = torch.load(args.features, weights_only=False)
    examples = blob['examples'] if isinstance(blob, dict) else blob
    config   = blob.get('config', {}) if isinstance(blob, dict) else {}
    print(f"  {len(examples)} examples.")

    fixed = []
    for ex in examples:
        new_ex = dict(ex)
        for old_key, new_key in RENAME.items():
            if old_key in new_ex:
                new_ex[new_key] = new_ex.pop(old_key)
        fixed.append(new_ex)

    # Verify a sample
    sample_keys = list(fixed[0].keys())
    print(f"  Keys after fix: {sample_keys}")

    torch.save({'config': config, 'examples': fixed}, args.out)
    print(f"Saved → {args.out}")


if __name__ == '__main__':
    main()