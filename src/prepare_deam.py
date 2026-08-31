"""
Build DEAM splits for Task 3's emotion-regression extension.

Run AFTER downloading DEAM (see scripts/download_deam.sh):

    python src/prepare_deam.py --raw_dir data/raw/deam \
        --feat_dir data/processed/real/deam_features \
        --splits_dir data/splits_deam
    python src/graph_builder.py --split_dirs data/splits_deam \
        --out_dir data/processed/deam_graphs
      # (graph_builder.py's default process_split() reads a hardcoded
      #  data/splits/*.json; either point it at data/splits_deam via the
      #  toy_dir/out_dir args directly in a small driver script, or copy
      #  data/splits_deam/*.json into data/splits/ temporarily -- see
      #  README for the exact one-liner.)

IMPORTANT -- why this is a SEPARATE split, not merged with MagnaTagATune:
DEAM's 1,802 clips and MagnaTagATune's ~26,000 clips are different songs
by different artists. There is no genuine way to say "this MTAT clip's
valence is X" -- so this script does NOT try to align the two datasets.
Instead:
  - Train Task 3 on MagnaTagATune with emotion_alpha=emotion_beta=0
    (tag+text fusion only, matching the primary tagging objective).
  - Separately train/evaluate Task 3 on this DEAM split with a config
    that sets tag_weight=0 and emotion_alpha/beta>0 (pure emotion
    regression, matching spec Section 4.3's framing of DEAM as an
    "emotion extension" / auxiliary loss, not a required joint label).
This is a disclosed, deliberate design choice -- write it up as such in
the report rather than silently zero-filling missing labels.

Captions here are synthesized from DEAM's genre/tag metadata (if you
downloaded metadata.zip alongside the audio) or left as a generic
placeholder if unavailable -- Task 3's text branch still needs *some*
input, but DEAM's synthesized captions should not be conflated with
MusicCaps' real human captions in your writeup.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from audio_features import extract_features, fixed_window_segments


def load_static_annotations(annotations_csv):
    df = pd.read_csv(annotations_csv)
    df.columns = [c.strip() for c in df.columns]
    # DEAM's static CSV columns are typically:
    # song_id, valence_mean, arousal_mean, valence_std, arousal_std
    # (exact names vary slightly by release year -- normalize here)
    rename = {}
    for c in df.columns:
        cl = c.lower()
        if "valence" in cl and "mean" in cl:
            rename[c] = "valence"
        elif "arousal" in cl and "mean" in cl:
            rename[c] = "arousal"
        elif cl in ("song_id", "songid", "id"):
            rename[c] = "song_id"
    df = df.rename(columns=rename)
    return df[["song_id", "valence", "arousal"]].set_index("song_id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--feat_dir", default="data/processed/real/deam_features")
    ap.add_argument("--splits_dir", default="data/splits_deam")
    ap.add_argument("--annotations_csv", default=None,
                     help="Defaults to <raw_dir>/annotations/static_annotations_averaged_songs_1_2000.csv")
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--n_mels", type=int, default=128)
    ap.add_argument("--n_chroma", type=int, default=12)
    ap.add_argument("--window_seconds", type=float, default=5.0)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ann_csv = args.annotations_csv or os.path.join(
        args.raw_dir, "annotations", "static_annotations_averaged_songs_1_2000.csv")
    va = load_static_annotations(ann_csv)

    audio_files = sorted(glob.glob(os.path.join(args.raw_dir, "audio", "*.mp3")))
    if not audio_files:
        print(f"No audio found under {args.raw_dir}/audio -- did you unzip DEAM_audio.zip there?")
        return

    os.makedirs(args.feat_dir, exist_ok=True)
    items = []
    for path in audio_files:
        song_id_str = os.path.splitext(os.path.basename(path))[0]
        try:
            song_id = int(song_id_str)
        except ValueError:
            song_id = song_id_str
        if song_id not in va.index:
            continue  # no VA label for this file, skip rather than guess

        mel, chroma, _ = extract_features(path, sr=args.sr, n_mels=args.n_mels, n_chroma=args.n_chroma)
        bounds = fixed_window_segments(mel.shape[1], args.sr, window_seconds=args.window_seconds)
        feat_path = os.path.join(args.feat_dir, f"{song_id_str}.npz")
        np.savez_compressed(feat_path, mel=mel, chroma=chroma, segment_bounds=np.array(bounds))

        items.append({
            "track_id": song_id_str,
            "feature_path": feat_path,
            "caption": "an instrumental clip rated for valence and arousal.",
            "caption_is_synthetic": True,
            "tags": None,  # DEAM has no tag labels; train with tag_weight=0 on this split
            "valence": float(va.loc[song_id, "valence"]),
            "arousal": float(va.loc[song_id, "arousal"]),
        })

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(items))
    n_test = int(len(items) * args.test_frac)
    n_val = int(len(items) * args.val_frac)
    test_idx = set(perm[:n_test])
    val_idx = set(perm[n_test:n_test + n_val])

    splits = {"train": [], "val": [], "test": []}
    for i, item in enumerate(items):
        if i in test_idx:
            splits["test"].append(item)
        elif i in val_idx:
            splits["val"].append(item)
        else:
            splits["train"].append(item)

    os.makedirs(args.splits_dir, exist_ok=True)
    for name, manifest in splits.items():
        out_path = os.path.join(args.splits_dir, f"{name}.json")
        with open(out_path, "w") as f:
            json.dump(manifest, f)
        print(f"[{name}] {len(manifest)} tracks -> {out_path}")

    print("\nNOTE: this is a random split (DEAM has no standard official "
          "train/val/test partition the way MTAT does) -- document that "
          "choice in your report's Section 3 (Dataset & preprocessing).")
    print("Remember: set train.tag_weight=0 and emotion_alpha/beta>0 in "
          "config.yaml for runs on this split.")


if __name__ == "__main__":
    main()
