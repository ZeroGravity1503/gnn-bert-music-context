"""
Dataset wrappers shared across Task 1-4.

MusicTagDataset   -> Task 1 (text/caption + tags)
MusicGraphDataset  -> Task 2 (structure graph + tags)
MusicFusionDataset -> Task 3 (text + graph + tags + valence/arousal)
MusicCapsPairDataset -> Task 4 (graph, caption) pairs for contrastive learning
"""
import json

import numpy as np
import torch
from torch.utils.data import Dataset


def load_manifest(split, splits_dir="data/splits"):
    with open(f"{splits_dir}/{split}.json") as f:
        return json.load(f)


def load_tag_vocab(toy_dir="data/processed/toy"):
    with open(f"{toy_dir}/tag_vocab.json") as f:
        return json.load(f)


class MusicTagDataset(Dataset):
    """Task 1: caption text -> multi-label tags."""

    def __init__(self, split, tokenizer, max_len=128, splits_dir="data/splits"):
        self.items = load_manifest(split, splits_dir)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        enc = self.tokenizer(
            item["caption"], truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "tags": torch.tensor(item["tags"], dtype=torch.float32),
        }


class MusicGraphDataset(Dataset):
    """Task 2: segment structure graph -> tags."""

    def __init__(self, split, graph_dir="data/processed/graphs", splits_dir="data/splits"):
        self.items = load_manifest(split, splits_dir)
        self.graph_dir = graph_dir

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        g = np.load(f"{self.graph_dir}/{item['track_id']}.npz")
        return {
            "x": torch.tensor(g["seg_x"], dtype=torch.float32),
            "edge_index": torch.tensor(g["seg_edge_index"], dtype=torch.long),
            "edge_weight": torch.tensor(g["seg_edge_weight"], dtype=torch.float32),
            "tags": torch.tensor(item["tags"], dtype=torch.float32),
        }


class MusicFusionDataset(Dataset):
    """Task 3: text + graph -> tags + valence/arousal."""

    def __init__(self, split, tokenizer, max_len=128,
                 graph_dir="data/processed/graphs", splits_dir="data/splits",
                 num_tags=None):
        self.items = load_manifest(split, splits_dir)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.graph_dir = graph_dir
        # num_tags is only needed as a fallback when a split has tags=null
        # (e.g. a DEAM-only emotion-regression run, see prepare_deam.py) --
        # real tag-bearing splits (MagnaTagATune, toy data) don't need it.
        self.num_tags = num_tags

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        enc = self.tokenizer(
            item["caption"], truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt",
        )
        g = np.load(f"{self.graph_dir}/{item['track_id']}.npz")

        if item.get("tags") is not None:
            tags_t = torch.tensor(item["tags"], dtype=torch.float32)
        else:
            if self.num_tags is None:
                raise ValueError(
                    f"Item {item['track_id']} has tags=null and no num_tags "
                    "fallback was set -- pass num_tags=<int> to MusicFusionDataset "
                    "(e.g. cfg['model']['num_tags']) for tag-less splits like DEAM. "
                    "Set train.tag_weight=0 in config.yaml so this dummy zero "
                    "vector doesn't corrupt the loss."
                )
            tags_t = torch.zeros(self.num_tags, dtype=torch.float32)

        # valence/arousal default to 0.0 when absent (e.g. MagnaTagATune-only
        # runs) -- harmless as long as emotion_alpha/emotion_beta=0 in config,
        # since the MSE term is then weighted to zero regardless of target.
        valence = item.get("valence")
        arousal = item.get("arousal")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "x": torch.tensor(g["seg_x"], dtype=torch.float32),
            "edge_index": torch.tensor(g["seg_edge_index"], dtype=torch.long),
            "edge_weight": torch.tensor(g["seg_edge_weight"], dtype=torch.float32),
            "tags": tags_t,
            "valence": torch.tensor(valence if valence is not None else 0.0, dtype=torch.float32),
            "arousal": torch.tensor(arousal if arousal is not None else 0.0, dtype=torch.float32),
        }


class MusicCapsPairDataset(Dataset):
    """Task 4: (graph, caption) pairs for InfoNCE contrastive training."""

    def __init__(self, split, tokenizer, max_len=128,
                 graph_dir="data/processed/graphs", splits_dir="data/splits"):
        self.items = load_manifest(split, splits_dir)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.graph_dir = graph_dir

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        enc = self.tokenizer(
            item["caption"], truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt",
        )
        g = np.load(f"{self.graph_dir}/{item['track_id']}.npz")
        return {
            "track_id": item["track_id"],
            "caption": item["caption"],
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "x": torch.tensor(g["seg_x"], dtype=torch.float32),
            "edge_index": torch.tensor(g["seg_edge_index"], dtype=torch.long),
            "edge_weight": torch.tensor(g["seg_edge_weight"], dtype=torch.float32),
        }


def collate_graphs(batch, key_prefix=""):
    """
    Manual batching for variable-size graphs (avoids requiring
    torch_geometric's DataLoader/Batch if it's unavailable). Concatenates
    node features and offsets edge_index per-graph, and returns a
    `batch_index` vector mapping each node to its graph, for mean-pool
    readout.
    """
    xs, edge_indices, edge_weights, batch_index = [], [], [], []
    node_offset = 0
    for i, item in enumerate(batch):
        x = item[f"{key_prefix}x"] if f"{key_prefix}x" in item else item["x"]
        ei = item[f"{key_prefix}edge_index"] if f"{key_prefix}edge_index" in item else item["edge_index"]
        ew = item[f"{key_prefix}edge_weight"] if f"{key_prefix}edge_weight" in item else item["edge_weight"]
        n = x.shape[0]
        xs.append(x)
        edge_indices.append(ei + node_offset)
        edge_weights.append(ew)
        batch_index.append(torch.full((n,), i, dtype=torch.long))
        node_offset += n
    return {
        "x": torch.cat(xs, dim=0),
        "edge_index": torch.cat(edge_indices, dim=1),
        "edge_weight": torch.cat(edge_weights, dim=0),
        "batch_index": torch.cat(batch_index, dim=0),
    }


def graph_collate_fn(batch):
    out = collate_graphs(batch)
    out["tags"] = torch.stack([b["tags"] for b in batch])
    return out


def fusion_collate_fn(batch):
    out = collate_graphs(batch)
    out["input_ids"] = torch.stack([b["input_ids"] for b in batch])
    out["attention_mask"] = torch.stack([b["attention_mask"] for b in batch])
    out["tags"] = torch.stack([b["tags"] for b in batch])
    out["valence"] = torch.stack([b["valence"] for b in batch])
    out["arousal"] = torch.stack([b["arousal"] for b in batch])
    return out


def contrastive_collate_fn(batch):
    out = collate_graphs(batch)
    out["input_ids"] = torch.stack([b["input_ids"] for b in batch])
    out["attention_mask"] = torch.stack([b["attention_mask"] for b in batch])
    out["track_id"] = [b["track_id"] for b in batch]
    out["caption"] = [b["caption"] for b in batch]
    return out
