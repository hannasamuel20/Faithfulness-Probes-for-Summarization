# step3_eval_probes_nq.py
import torch, json
import numpy as np
from sklearn.metrics import roc_auc_score

# Load your trained probes from AggreFact
# (replace with however you saved them)
cnn_probe = torch.load("trained_cnn_probe.pt")
cnn_probe.eval()

# Load NQ features + labels
features = torch.load("nq_features.pt")
labels_by_q = {}
with open("nq_annotated.jsonl") as f:
    for line in f:
        d = json.loads(line)
        labels_by_q[d["question"]] = d["label"]

all_preds, all_labels = [], []

with torch.no_grad():
    for item in features:
        if item["question"] not in labels_by_q:
            continue
        label = labels_by_q[item["question"]]
        
        # lookback_ratio: (T, L, H) -> reshape to (L*H, T) for your CNN
        lr = item["lookback_ratio"].permute(1, 2, 0)  # (L, H, T)
        L, H, T = lr.shape
        lr = lr.reshape(L*H, T).unsqueeze(0)  # (1, L*H, T)
        
        logit = cnn_probe(lr.to(next(cnn_probe.parameters()).device))
        pred = torch.sigmoid(logit).item()
        all_preds.append(pred)
        all_labels.append(label)

auroc = roc_auc_score(all_labels, all_preds)
print(f"NQ Transfer AUROC: {auroc:.3f}")
print(f"N examples: {len(all_labels)}, "
      f"Faithful: {sum(all_labels)}, "
      f"Hallucinated: {len(all_labels)-sum(all_labels)}")