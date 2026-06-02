"""Evaluation with multilabel metrics for AMP prediction."""

import json, yaml, torch, numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, hamming_loss
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from amp_discovery.data import load_escape, load_genpept, LABELS

def evaluate(config_path: str):
    cfg = yaml.safe_load(open(config_path))
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(cfg["model_name"], num_labels=len(LABELS))
    if cfg.get("adapter_path"):
        model = PeftModel.from_pretrained(model, cfg["adapter_path"])
    ds = load_escape()["test"]
    y_true, y_pred = [], []
    for row in ds:
        inp = tokenizer(row["sequence"], return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        with torch.no_grad(): logits = model(**inp).logits
        preds = (torch.sigmoid(logits) > 0.5).int().squeeze().tolist()
        y_true.append([int(row[l]) for l in LABELS])
        y_pred.append(preds)
    yt, yp = np.array(y_true), np.array(y_pred)
    result = {"accuracy": float(accuracy_score(yt, yp)), "f1_micro": float(f1_score(yt, yp, average="micro")), "f1_macro": float(f1_score(yt, yp, average="macro")), "hamming": float(hamming_loss(yt, yp))}
    for i, label in enumerate(LABELS):
        result[f"f1_{label}"] = float(f1_score(yt[:,i], yp[:,i]))
    out = Path(cfg.get("output_dir","outputs/evals"))
    out.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(out/"summary.json","w"), indent=2)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    import sys; evaluate(sys.argv[1])
