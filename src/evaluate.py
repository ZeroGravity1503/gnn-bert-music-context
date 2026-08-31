"""
Evaluation for Tasks 1-4 (Section 6 metrics + Section 8 baseline table).

    python src/evaluate.py --task 1
    python src/evaluate.py --task 2
    python src/evaluate.py --task 3   # also dumps a t-SNE plot of z
    python src/evaluate.py --task 4   # retrieval R@K + qualitative examples
    python src/evaluate.py --task all --baselines   # full comparison table
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score, average_precision_score, r2_score
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from bert_encoder import BertTagClassifier
from gnn_model import GNNTagClassifier, CNNBaseline
from fusion_model import GNNBertFusionModel
from contrastive import ContrastiveGNNBert, retrieval_recall_at_k
from datasets import (
    MusicTagDataset, MusicGraphDataset, MusicFusionDataset, MusicCapsPairDataset,
    load_tag_vocab, graph_collate_fn, fusion_collate_fn, contrastive_collate_fn,
)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tag_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    # AUC-PR per tag, averaged (skip tags with no positive examples)
    aucprs = []
    for k in range(y_true.shape[1]):
        if y_true[:, k].sum() > 0:
            aucprs.append(average_precision_score(y_true[:, k], y_prob[:, k]))
    mean_aucpr = float(np.mean(aucprs)) if aucprs else float("nan")
    return {"macro_f1": float(macro_f1), "micro_f1": float(micro_f1),
            "mean_auc_pr": mean_aucpr}


@torch.no_grad()
def eval_task1(cfg, device):
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["bert_name"])
    tags = load_tag_vocab()
    model = BertTagClassifier(cfg["model"]["bert_name"], num_tags=len(tags)).to(device)
    model.load_state_dict(torch.load("results/task1_bert.pt", map_location=device))
    model.eval()

    ds = MusicTagDataset("test", tokenizer, cfg["data"]["max_text_len"])
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"])
    y_true, y_prob = [], []
    for batch in dl:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        y_prob.append(torch.sigmoid(logits).cpu().numpy())
        y_true.append(batch["tags"].numpy())
    y_true, y_prob = np.concatenate(y_true), np.concatenate(y_prob)
    return tag_metrics(y_true, y_prob)


@torch.no_grad()
def eval_task2(cfg, device, also_cnn_baseline=True):
    tags = load_tag_vocab()
    test_ds = MusicGraphDataset("test")
    in_dim = test_ds[0]["x"].shape[1]
    model = GNNTagClassifier(
        in_dim, cfg["model"]["gnn_hidden"], cfg["model"]["gnn_layers"],
        num_tags=len(tags), gnn_type=cfg["model"]["gnn_type"],
    ).to(device)
    model.load_state_dict(torch.load("results/task2_gnn.pt", map_location=device))
    model.eval()
    dl = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"], collate_fn=graph_collate_fn)

    y_true, y_prob = [], []
    for batch in dl:
        logits = model(batch["x"].to(device), batch["edge_index"].to(device),
                        batch["batch_index"].to(device))
        y_prob.append(torch.sigmoid(logits).cpu().numpy())
        y_true.append(batch["tags"].numpy())
    y_true, y_prob = np.concatenate(y_true), np.concatenate(y_prob)
    metrics = {"gnn": tag_metrics(y_true, y_prob)}

    if also_cnn_baseline and os.path.exists("results/cnn_baseline.pt"):
        cnn = CNNBaseline(cfg["data"]["n_mels"], len(tags)).to(device)
        cnn.load_state_dict(torch.load("results/cnn_baseline.pt", map_location=device))
        cnn.eval()
        # Note: requires mel spectrograms aligned to the same test split;
        # see train_cnn_baseline() below for how B2 is trained.
    return metrics


def train_cnn_baseline(cfg, epochs, device):
    """B2 baseline: CNN on raw mel-spectrogram (no graph, no text)."""
    import json as _json
    tags = load_tag_vocab()
    with open("data/splits/train.json") as f:
        manifest = _json.load(f)

    X, Y = [], []
    for item in manifest:
        d = np.load(item["feature_path"])
        mel = d["mel"]
        # pad/crop to fixed length for simple batching
        T = 130
        if mel.shape[1] < T:
            mel = np.pad(mel, ((0, 0), (0, T - mel.shape[1])))
        else:
            mel = mel[:, :T]
        X.append(mel)
        Y.append(item["tags"])
    X = torch.tensor(np.stack(X), dtype=torch.float32)
    Y = torch.tensor(np.stack(Y), dtype=torch.float32)

    model = CNNBaseline(cfg["data"]["n_mels"], len(tags)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    model.train()
    bs = cfg["train"]["batch_size"]
    for epoch in range(epochs):
        perm = torch.randperm(len(X))
        total = 0.0
        for i in range(0, len(X), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            logits = model(X[idx].to(device))
            loss = bce(logits, Y[idx].to(device))
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"[B2/CNN][epoch {epoch+1}/{epochs}] loss={total / max(1, len(X)//bs):.4f}")
    torch.save(model.state_dict(), "results/cnn_baseline.pt")
    return model


@torch.no_grad()
def eval_task3(cfg, device, make_tsne=True):
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["bert_name"])
    tags = load_tag_vocab()
    test_ds = MusicFusionDataset("test", tokenizer, cfg["data"]["max_text_len"])
    in_dim = test_ds[0]["x"].shape[1]
    model = GNNBertFusionModel(
        bert_name=cfg["model"]["bert_name"], gnn_in_dim=in_dim,
        gnn_hidden=cfg["model"]["gnn_hidden"], gnn_layers=cfg["model"]["gnn_layers"],
        num_tags=len(tags), fusion_type=cfg["model"]["fusion_type"],
    ).to(device)
    model.load_state_dict(torch.load("results/task3_fusion.pt", map_location=device))
    model.eval()
    dl = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"], collate_fn=fusion_collate_fn)

    y_true, y_prob, va_true, va_pred_all, z_all = [], [], [], [], []
    for batch in dl:
        tag_logits, va_pred = model(
            batch["input_ids"].to(device), batch["attention_mask"].to(device),
            batch["x"].to(device), batch["edge_index"].to(device),
            batch["batch_index"].to(device),
        )
        y_prob.append(torch.sigmoid(tag_logits).cpu().numpy())
        y_true.append(batch["tags"].numpy())
        va_true.append(torch.stack([batch["valence"], batch["arousal"]], dim=1).numpy())
        va_pred_all.append(va_pred.cpu().numpy())

    y_true, y_prob = np.concatenate(y_true), np.concatenate(y_prob)
    va_true, va_pred_all = np.concatenate(va_true), np.concatenate(va_pred_all)

    metrics = tag_metrics(y_true, y_prob)
    metrics["valence_mae"] = float(np.mean(np.abs(va_true[:, 0] - va_pred_all[:, 0])))
    metrics["arousal_mae"] = float(np.mean(np.abs(va_true[:, 1] - va_pred_all[:, 1])))
    metrics["valence_r2"] = float(r2_score(va_true[:, 0], va_pred_all[:, 0]))
    metrics["arousal_r2"] = float(r2_score(va_true[:, 1], va_pred_all[:, 1]))

    if make_tsne:
        plot_tsne_fusion(model, test_ds, tags, device, cfg)

    return metrics


@torch.no_grad()
def plot_tsne_fusion(model, test_ds, tags, device, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dl = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"], collate_fn=fusion_collate_fn)
    zs, dominant_tag = [], []
    for batch in dl:
        H_text, t = model.text_encoder(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        num_graphs = batch["input_ids"].shape[0]
        _, g = model.gnn_encoder(batch["x"].to(device), batch["edge_index"].to(device),
                                  batch["batch_index"].to(device), num_graphs=num_graphs)
        z = model.fusion(g, H_text, batch["attention_mask"].to(device)) if model.fusion_type == "cross_attn" \
            else torch.cat([g, t], dim=1)
        zs.append(z.cpu().numpy())
        tag_idx = batch["tags"].numpy().argmax(axis=1)
        dominant_tag.append(tag_idx)
    zs = np.concatenate(zs)
    dominant_tag = np.concatenate(dominant_tag)

    if len(zs) < 5:
        return
    proj = TSNE(n_components=2, perplexity=min(30, max(5, len(zs) // 3)),
                random_state=cfg["train"]["seed"]).fit_transform(zs)

    os.makedirs("results/plots", exist_ok=True)
    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(proj[:, 0], proj[:, 1], c=dominant_tag, cmap="tab20", s=18)
    plt.title("t-SNE of fused representation z (colored by dominant tag)")
    plt.xlabel("dim 1"); plt.ylabel("dim 2")
    plt.tight_layout()
    plt.savefig("results/plots/tsne_fusion_z.png", dpi=150)
    plt.close()
    print("Saved t-SNE plot -> results/plots/tsne_fusion_z.png")


@torch.no_grad()
def eval_task4(cfg, device, n_qualitative=10):
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["bert_name"])
    test_ds = MusicCapsPairDataset("test", tokenizer, cfg["data"]["max_text_len"])
    in_dim = test_ds[0]["x"].shape[1]
    model = ContrastiveGNNBert(
        bert_name=cfg["model"]["bert_name"], gnn_in_dim=in_dim,
        gnn_hidden=cfg["model"]["gnn_hidden"], gnn_layers=cfg["model"]["gnn_layers"],
        proj_dim=cfg["model"]["gnn_hidden"],
    ).to(device)
    model.load_state_dict(torch.load("results/task4_contrastive.pt", map_location=device))
    model.eval()
    dl = DataLoader(test_ds, batch_size=len(test_ds), collate_fn=contrastive_collate_fn)  # full-batch for retrieval
    batch = next(iter(dl))
    g_emb, t_emb = model(
        batch["input_ids"].to(device), batch["attention_mask"].to(device),
        batch["x"].to(device), batch["edge_index"].to(device),
        batch["batch_index"].to(device),
    )
    recall = retrieval_recall_at_k(g_emb, t_emb, k_list=tuple(cfg["eval"]["top_k_retrieval"]))

    # qualitative: for first n_qualitative captions, show top-3 retrieved track_ids
    sim = (t_emb @ g_emb.t()).cpu().numpy()
    examples = []
    for i in range(min(n_qualitative, len(batch["caption"]))):
        top3 = sim[i].argsort()[::-1][:3]
        examples.append({
            "query_caption": batch["caption"][i],
            "true_track_id": batch["track_id"][i],
            "top3_retrieved": [batch["track_id"][j] for j in top3],
        })

    os.makedirs("results/retrieval_examples", exist_ok=True)
    with open("results/retrieval_examples/task4_qualitative.json", "w") as f:
        json.dump(examples, f, indent=2)
    print("Saved qualitative retrieval examples -> results/retrieval_examples/task4_qualitative.json")

    return recall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["1", "2", "3", "4", "all"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--baselines", action="store_true",
                     help="Also train/eval B1 (random) and B2 (CNN) baselines for comparison")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = get_device()
    os.makedirs("results", exist_ok=True)

    all_metrics = {}
    tasks = ["1", "2", "3", "4"] if args.task == "all" else [args.task]
    for t in tasks:
        try:
            if t == "1":
                all_metrics["task1_bert"] = eval_task1(cfg, device)
            elif t == "2":
                all_metrics["task2_gnn"] = eval_task2(cfg, device)["gnn"]
            elif t == "3":
                all_metrics["task3_fusion"] = eval_task3(cfg, device)
            elif t == "4":
                all_metrics["task4_contrastive"] = eval_task4(cfg, device)
        except FileNotFoundError as e:
            print(f"[skip] Task {t}: checkpoint not found ({e}). Run train.py --task {t} first.")

    if args.baselines:
        # B1: random predictor, for reference
        tags = load_tag_vocab()
        with open("data/splits/test.json") as f:
            test_manifest = json.load(f)
        y_true = np.array([m["tags"] for m in test_manifest])
        rng = np.random.default_rng(cfg["train"]["seed"])
        y_prob_random = rng.random(y_true.shape)
        all_metrics["B1_random"] = tag_metrics(y_true, y_prob_random)

        if not os.path.exists("results/cnn_baseline.pt"):
            print("Training B2 CNN baseline...")
            train_cnn_baseline(cfg, epochs=3, device=device)

    print(json.dumps(all_metrics, indent=2))
    with open("results/metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print("Saved -> results/metrics.json")


if __name__ == "__main__":
    main()
