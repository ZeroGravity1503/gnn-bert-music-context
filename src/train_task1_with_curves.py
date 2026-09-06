"""
Task 1 retrain that tracks Macro-F1/Micro-F1 on the VALIDATION set
after every epoch (not just training loss) -- the spec asks for
"Macro-F1/Micro-F1 curves vs. training epochs", which the original
train.py does not track (it only logs training loss per epoch and a
single final test-set F1 via evaluate.py).

Deliberately a SEPARATE script rather than an edit to train.py, so it
can't accidentally break the already-working training path.

    python src/train_task1_with_curves.py --config config_mtat.yaml --epochs 12
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from bert_encoder import BertTagClassifier
from datasets import MusicTagDataset, load_tag_vocab


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def eval_f1(model, dl, device):
    model.eval()
    y_true, y_prob = [], []
    for batch in dl:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        y_prob.append(torch.sigmoid(logits).cpu().numpy())
        y_true.append(batch["tags"].numpy())
    y_true, y_prob = np.concatenate(y_true), np.concatenate(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    model.train()
    return float(macro_f1), float(micro_f1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_mtat.yaml")
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = get_device()
    print("Using device:", device)

    tags = load_tag_vocab()
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["bert_name"])
    splits_dir = cfg["data"].get("splits_dir", "data/splits")

    train_ds = MusicTagDataset("train", tokenizer, cfg["data"]["max_text_len"], splits_dir=splits_dir)
    val_ds = MusicTagDataset("val", tokenizer, cfg["data"]["max_text_len"], splits_dir=splits_dir)
    train_dl = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"])

    model = BertTagClassifier(cfg["model"]["bert_name"], num_tags=len(tags)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    bce = nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "val_macro_f1": [], "val_micro_f1": []}
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in train_dl:
            opt.zero_grad()
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            loss = bce(logits, batch["tags"].to(device))
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_dl)

        macro_f1, micro_f1 = eval_f1(model, val_dl, device)
        history["train_loss"].append(avg_loss)
        history["val_macro_f1"].append(macro_f1)
        history["val_micro_f1"].append(micro_f1)
        print(f"[epoch {epoch+1}/{args.epochs}] train_loss={avg_loss:.4f} "
              f"val_macro_f1={macro_f1:.4f} val_micro_f1={micro_f1:.4f}")

    with open("results/task1_f1_curves.json", "w") as f:
        json.dump(history, f, indent=2)
    print("Saved -> results/task1_f1_curves.json")

    torch.save(model.state_dict(), "results/task1_bert_mtat_with_curves.pt")
    print("Also saved checkpoint -> results/task1_bert_mtat_with_curves.pt")


if __name__ == "__main__":
    main()
