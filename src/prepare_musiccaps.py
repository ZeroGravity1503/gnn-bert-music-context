"""
Build Task 4 splits from downloaded MusicCaps (see download_musiccaps.py).

    python src/prepare_musiccaps.py --manifest data/raw/musiccaps/downloaded_manifest.csv \
        --feat_dir data/processed/real/musiccaps_features \
        --splits_dir data/splits_musiccaps
    python -c "
        import sys; sys.path.insert(0, 'src')
        from graph_builder import process_split
        for s in ['train', 'val', 'test']:
            process_split(s, out_dir='data/processed/musiccaps_graphs')
    "
    # then point MusicCapsPairDataset at splits_dir='data/splits_musiccaps',
    # graph_dir='data/processed/musiccaps_graphs' in train.py's Task 4 call
    # (both are already constructor args -- see datasets.py).

No tags/valence/arousal here -- Task 4 (contrastive retrieval) only needs
(graph, caption) pairs, matching MusicCapsPairDataset in datasets.py.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from audio_features import extract_features, fixed_window_segments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                     help="CSV from download_musiccaps.py: ytid, audio_path, caption")
    ap.add_argument("--feat_dir", default="data/processed/real/musiccaps_features")
    ap.add_argument("--splits_dir", default="data/splits_musiccaps")
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--n_mels", type=int, default=128)
    ap.add_argument("--n_chroma", type=int, default=12)
    ap.add_argument("--window_seconds", type=float, default=2.0)  # clips are only 10s
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    os.makedirs(args.feat_dir, exist_ok=True)

    items = []
    for _, row in df.iterrows():
        if not os.path.exists(row["audio_path"]):
            continue
        try:
            mel, chroma, _ = extract_features(row["audio_path"], sr=args.sr,
                                               n_mels=args.n_mels, n_chroma=args.n_chroma)
        except Exception as e:
            print(f"  [skip] {row['ytid']}: {e}")
            continue
        bounds = fixed_window_segments(mel.shape[1], args.sr, window_seconds=args.window_seconds)
        feat_path = os.path.join(args.feat_dir, f"{row['ytid']}.npz")
        np.savez_compressed(feat_path, mel=mel, chroma=chroma, segment_bounds=np.array(bounds))
        items.append({
            "track_id": row["ytid"],
            "feature_path": feat_path,
            "caption": row["caption"],
            "caption_is_synthetic": False,  # real MusicCaps human caption
            "tags": None,
            "valence": None,
            "arousal": None,
        })

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(items))
    n_test = int(len(items) * args.test_frac)
    n_val = int(len(items) * args.val_frac)
    test_idx, val_idx = set(perm[:n_test]), set(perm[n_test:n_test + n_val])

    splits = {"train": [], "val": [], "test": []}
    for i, item in enumerate(items):
        key = "test" if i in test_idx else ("val" if i in val_idx else "train")
        splits[key].append(item)

    os.makedirs(args.splits_dir, exist_ok=True)
    for name, manifest in splits.items():
        out_path = os.path.join(args.splits_dir, f"{name}.json")
        with open(out_path, "w") as f:
            json.dump(manifest, f)
        print(f"[{name}] {len(manifest)} tracks -> {out_path}")


if __name__ == "__main__":
    main()
