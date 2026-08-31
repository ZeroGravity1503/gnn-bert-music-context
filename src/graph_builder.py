"""
Graph construction for Task 2/3/4.

Two graph types, per the project spec:
  - Chord-transition graph: nodes = unique chords (from chroma argmax),
    edges = observed transitions weighted by count.
  - Segment graph: nodes = time segments (mean chroma/mel per segment),
    edges = temporal adjacency + cosine similarity above a threshold.

We build the segment graph by default since it works directly off the
per-track chroma/mel arrays produced by audio_features.py / the toy
generator, and it's what Task 2/3's GraphSAGE model consumes as `x`
(node features) + `edge_index`.
"""
import json
import os

import numpy as np

CHORD_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def build_segment_graph(mel, chroma, segment_bounds, sim_threshold=0.85):
    """
    mel: (n_mels, n_frames)
    chroma: (12, n_frames)
    segment_bounds: list of frame indices delimiting segments,
                     e.g. [0, 15, 40, 78, ..., n_frames]

    Returns node features (n_segments, feat_dim) and edge_index (2, n_edges).
    """
    n_segments = len(segment_bounds) - 1
    node_feats = []
    for i in range(n_segments):
        s, e = segment_bounds[i], segment_bounds[i + 1]
        e = max(e, s + 1)
        mel_seg = mel[:, s:e].mean(axis=1)
        chroma_seg = chroma[:, s:e].mean(axis=1)
        node_feats.append(np.concatenate([mel_seg, chroma_seg]))
    node_feats = np.stack(node_feats).astype(np.float32)  # (n_segments, 140)

    # normalize for cosine similarity
    norms = np.linalg.norm(node_feats, axis=1, keepdims=True) + 1e-8
    normed = node_feats / norms

    edges = []
    weights = []
    for i in range(n_segments):
        # temporal adjacency (always connect consecutive segments)
        if i + 1 < n_segments:
            edges.append((i, i + 1))
            edges.append((i + 1, i))
            weights += [1.0, 1.0]
        # similarity edges
        for j in range(i + 1, n_segments):
            sim = float(np.dot(normed[i], normed[j]))
            if sim > sim_threshold:
                edges.append((i, j))
                edges.append((j, i))
                weights += [sim, sim]

    if not edges:  # guarantee at least self-loops so GNN doesn't choke
        edges = [(i, i) for i in range(n_segments)]
        weights = [1.0] * n_segments

    edge_index = np.array(edges, dtype=np.int64).T  # (2, n_edges)
    edge_weight = np.array(weights, dtype=np.float32)
    return node_feats, edge_index, edge_weight


def build_chord_transition_graph(chroma):
    """
    Collapses each frame's chroma vector to its dominant pitch class (a
    crude chord/key proxy), then builds a 12-node graph where edges are
    weighted by observed transition counts. Useful as an alternative,
    smaller structural graph (fixed 12-node vocabulary across all tracks).
    """
    dominant = chroma.argmax(axis=0)  # (n_frames,) values in [0, 11]
    trans_counts = np.zeros((12, 12), dtype=np.float32)
    for t in range(len(dominant) - 1):
        trans_counts[dominant[t], dominant[t + 1]] += 1

    edges, weights = [], []
    for i in range(12):
        for j in range(12):
            if trans_counts[i, j] > 0:
                edges.append((i, j))
                weights.append(trans_counts[i, j])
    if not edges:
        edges = [(i, i) for i in range(12)]
        weights = [1.0] * 12

    # node features: one-hot chord identity + overall frequency
    freq = np.bincount(dominant, minlength=12).astype(np.float32)
    freq = freq / (freq.sum() + 1e-8)
    node_feats = np.eye(12, dtype=np.float32)
    node_feats = np.concatenate([node_feats, freq[:, None]], axis=1)

    edge_index = np.array(edges, dtype=np.int64).T
    edge_weight = np.array(weights, dtype=np.float32)
    return node_feats, edge_index, edge_weight


def process_split(split_name, toy_dir="data/processed/toy",
                   out_dir="data/processed/graphs"):
    os.makedirs(out_dir, exist_ok=True)
    with open(f"data/splits/{split_name}.json") as f:
        manifest = json.load(f)

    n_saved = 0
    for item in manifest:
        data = np.load(item["feature_path"])
        mel, chroma, bounds = data["mel"], data["chroma"], data["segment_bounds"]

        seg_x, seg_ei, seg_ew = build_segment_graph(mel, chroma, bounds)
        chord_x, chord_ei, chord_ew = build_chord_transition_graph(chroma)

        out_path = os.path.join(out_dir, f"{item['track_id']}.npz")
        np.savez_compressed(
            out_path,
            seg_x=seg_x, seg_edge_index=seg_ei, seg_edge_weight=seg_ew,
            chord_x=chord_x, chord_edge_index=chord_ei, chord_edge_weight=chord_ew,
        )
        n_saved += 1
    print(f"[{split_name}] built graphs for {n_saved} tracks -> {out_dir}")


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        process_split(split)
