"""
Unified trainer. Usage:

    python src/train.py --task 1 --epochs 3
    python src/train.py --task 2 --epochs 3
    python src/train.py --task 3 --epochs 3
    python src/train.py --task 4 --epochs 3

Implements Algorithms 1-4 from the project spec.
"""
import argparse
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from bert_encoder import BertTagClassifier
from gnn_model import GNNTagClassifier
from fusion_model import GNNBertFusionModel
from contrastive import ContrastiveGNNBert, info_nce_loss
from datasets import (
    MusicTagDataset, MusicGraphDataset, MusicFusionDataset, MusicCapsPairDataset,
    load_tag_vocab, graph_collate_fn, fusion_collate_fn, contrastive_collate_fn,
)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_task1(cfg, epochs, device):
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["bert_name"])
    tags = load_tag_vocab()
    model = BertTagClassifier(cfg["model"]["bert_name"], num_tags=len(tags)).to(device)

    train_ds = MusicTagDataset("train", tokenizer, cfg["data"]["max_text_len"],
                                splits_dir=cfg["data"].get("splits_dir", "data/splits"))
    train_dl = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    bce = nn.BCEWithLogitsLoss()

    model.train()
    history = []
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_dl:
            opt.zero_grad()
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            loss = bce(logits, batch["tags"].to(device))
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg = total_loss / len(train_dl)
        print(f"[Task1][epoch {epoch+1}/{epochs}] BCE loss={avg:.4f}")
        history.append(avg)

    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), "results/task1_bert.pt")
    return model, tokenizer, history


def train_task2(cfg, epochs, device):
    tags = load_tag_vocab()
    train_ds = MusicGraphDataset(
        "train",
        graph_dir=cfg["data"].get("graph_dir", "data/processed/graphs"),
        splits_dir=cfg["data"].get("splits_dir", "data/splits"),
    )
    in_dim = train_ds[0]["x"].shape[1]
    model = GNNTagClassifier(
        in_dim, cfg["model"]["gnn_hidden"], cfg["model"]["gnn_layers"],
        num_tags=len(tags), gnn_type=cfg["model"]["gnn_type"],
    ).to(device)
    train_dl = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"],
                           shuffle=True, collate_fn=graph_collate_fn)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["gnn_lr"])
    bce = nn.BCEWithLogitsLoss()

    model.train()
    history = []
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_dl:
            opt.zero_grad()
            logits = model(batch["x"].to(device), batch["edge_index"].to(device),
                            batch["batch_index"].to(device))
            loss = bce(logits, batch["tags"].to(device))
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg = total_loss / len(train_dl)
        print(f"[Task2][epoch {epoch+1}/{epochs}] BCE loss={avg:.4f}")
        history.append(avg)

    torch.save(model.state_dict(), "results/task2_gnn.pt")
    return model, history


def train_task3(cfg, epochs, device):
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["bert_name"])
    # Graceful fallback: DEAM-only runs have no tag_vocab.json (DEAM has no
    # tag labels) -- fall back to cfg["model"]["num_tags"] and let
    # train.tag_weight=0 zero out the (dummy) tag loss for that run.
    try:
        tags = load_tag_vocab()
        num_tags = len(tags)
    except FileNotFoundError:
        num_tags = cfg["model"]["num_tags"]
        print(f"[Task3] No tag_vocab.json found -- using cfg.model.num_tags="
              f"{num_tags} as a dummy tag dimension (expected for a DEAM-only "
              f"emotion-regression run; make sure train.tag_weight=0).")

    splits_dir = cfg["data"].get("splits_dir", "data/splits")
    graph_dir = cfg["data"].get("graph_dir", "data/processed/graphs")
    train_ds = MusicFusionDataset("train", tokenizer, cfg["data"]["max_text_len"],
                                   graph_dir=graph_dir, splits_dir=splits_dir,
                                   num_tags=num_tags)
    in_dim = train_ds[0]["x"].shape[1]
    model = GNNBertFusionModel(
        bert_name=cfg["model"]["bert_name"], gnn_in_dim=in_dim,
        gnn_hidden=cfg["model"]["gnn_hidden"], gnn_layers=cfg["model"]["gnn_layers"],
        num_tags=num_tags, fusion_type=cfg["model"]["fusion_type"],
    ).to(device)
    train_dl = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"],
                           shuffle=True, collate_fn=fusion_collate_fn)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    alpha, beta = cfg["train"]["emotion_alpha"], cfg["train"]["emotion_beta"]
    tag_weight = cfg["train"].get("tag_weight", 1.0)

    model.train()
    history = []
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_dl:
            opt.zero_grad()
            tag_logits, va_pred = model(
                batch["input_ids"].to(device), batch["attention_mask"].to(device),
                batch["x"].to(device), batch["edge_index"].to(device),
                batch["batch_index"].to(device),
            )
            l_tags = bce(tag_logits, batch["tags"].to(device))
            l_val = mse(va_pred[:, 0], batch["valence"].to(device))
            l_aro = mse(va_pred[:, 1], batch["arousal"].to(device))
            loss = tag_weight * l_tags + alpha * l_val + beta * l_aro
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg = total_loss / len(train_dl)
        print(f"[Task3][epoch {epoch+1}/{epochs}] total loss={avg:.4f}")
        history.append(avg)

    torch.save(model.state_dict(), "results/task3_fusion.pt")
    return model, tokenizer, history


def train_task4(cfg, epochs, device):
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["bert_name"])
    splits_dir = cfg["data"].get("splits_dir", "data/splits")
    graph_dir = cfg["data"].get("graph_dir", "data/processed/graphs")
    train_ds = MusicCapsPairDataset("train", tokenizer, cfg["data"]["max_text_len"],
                                     graph_dir=graph_dir, splits_dir=splits_dir)
    in_dim = train_ds[0]["x"].shape[1]
    model = ContrastiveGNNBert(
        bert_name=cfg["model"]["bert_name"], gnn_in_dim=in_dim,
        gnn_hidden=cfg["model"]["gnn_hidden"], gnn_layers=cfg["model"]["gnn_layers"],
        proj_dim=cfg["model"]["gnn_hidden"],
    ).to(device)
    train_dl = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"],
                           shuffle=True, collate_fn=contrastive_collate_fn)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    tau = cfg["train"]["contrastive_temperature"]

    model.train()
    history = []
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_dl:
            opt.zero_grad()
            g_emb, t_emb = model(
                batch["input_ids"].to(device), batch["attention_mask"].to(device),
                batch["x"].to(device), batch["edge_index"].to(device),
                batch["batch_index"].to(device),
            )
            loss = info_nce_loss(g_emb, t_emb, temperature=tau)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg = total_loss / len(train_dl)
        print(f"[Task4][epoch {epoch+1}/{epochs}] InfoNCE loss={avg:.4f}")
        history.append(avg)

    torch.save(model.state_dict(), "results/task4_contrastive.pt")
    return model, tokenizer, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    print(f"Using device: {device}")
    os.makedirs("results", exist_ok=True)

    if args.task == 1:
        _, _, hist = train_task1(cfg, args.epochs, device)
    elif args.task == 2:
        _, hist = train_task2(cfg, args.epochs, device)
    elif args.task == 3:
        _, _, hist = train_task3(cfg, args.epochs, device)
    elif args.task == 4:
        _, _, hist = train_task4(cfg, args.epochs, device)

    hist_path = f"results/task{args.task}_loss_history.json"
    with open(hist_path, "w") as f:
        json.dump(hist, f, indent=2)
    print(f"Saved loss history -> {hist_path}")


if __name__ == "__main__":
    main()
