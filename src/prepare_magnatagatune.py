"""
Build data/splits/{train,val,test}.json from a downloaded MagnaTagATune.

RESUME-SAFE: skips any clip whose feature file already exists, so an
interrupted Colab session can just be re-run without redoing work.

SUBSETTABLE: --max_per_split lets you cap how many clips per split get
processed, so this finishes in a single Colab session instead of
requiring hours of uninterrupted processing across ~25,863 clips.
Document the subset size in your report -- it's a disclosed, deliberate
scope decision given compute/time constraints, not a hidden shortcut.

    python src/prepare_magnatagatune.py --raw_dir /content/mtat_raw \
        --feat_dir /content/mtat_features --splits_dir data/splits \
        --max_per_split 1500

What this does (entirely self-contained, no external split-file dependency):
  1. Parses `annotations_final.csv` directly (clip_id, 188 tag columns,
     mp3_path column like "8/artist-...-59-88.mp3" -- leading char is
     the hex folder).
  2. Picks the top-K tags by frequency (default 50, matches spec).
  3. Splits by hex folder prefix: 0-9,a,b -> train, c -> val, d,e,f -> test
     (~12:1:3, standard MTAT split).
  4. Extracts mel/chroma features (audio_features.py) to --feat_dir.
  5. Synthesizes a caption per clip from its tags (MTAT has no real
     captions) -- flagged `caption_is_synthetic: true`.
  6. valence/arousal are null (MTAT has none) -- config_mtat.yaml sets
     emotion_alpha/beta=0 accordingly.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from audio_features import extract_features, fixed_window_segments


TRAIN_FOLDERS = set("0123456789ab")
VAL_FOLDERS = set("c")
TEST_FOLDERS = set("def")


def synthesize_caption(active_tags):
    if not active_tags:
        return "an instrumental track with no strong tag associations."
    return "Tags: " + ", ".join(active_tags) + "."


def _sniff_tsv(path):
    with open(path) as f:
        first_line = f.readline()
    return "\t" in first_line and "," not in first_line.split("\t")[0]


def load_annotations(csv_path, top_k=50):
    df = pd.read_csv(csv_path, sep="\t") if _sniff_tsv(csv_path) else pd.read_csv(csv_path)
    non_tag_cols = {"clip_id", "mp3_path"}
    tag_cols = [c for c in df.columns if c not in non_tag_cols]
    tag_freq = df[tag_cols].sum(axis=0).sort_values(ascending=False)
    top_tags = list(tag_freq.head(top_k).index)
    return df, top_tags


def load_existing_manifest(out_path):
    """For resuming: if this split's json already has entries from a
    prior partial run, load them so we don't reprocess those clips."""
    if os.path.exists(out_path):
        with open(out_path) as f:
            return json.load(f)
    return []


def build_manifest(df, top_tags, raw_dir, feat_dir, sr, n_mels, n_chroma,
                    window_seconds, folders, max_clips, existing, split_name):
    os.makedirs(feat_dir, exist_ok=True)
    done_ids = {item["track_id"] for item in existing}
    manifest = list(existing)  # keep what's already done

    subset = df[df["mp3_path"].str[0].isin(folders)]
    if max_clips is not None:
        subset = subset.head(max_clips + len(done_ids))  # overshoot to account for skips/failures

    processed_this_run = 0
    for _, row in subset.iterrows():
        if max_clips is not None and len(manifest) >= max_clips:
            break

        track_id = f"mtat_{row['clip_id']}"
        if track_id in done_ids:
            continue  # already processed in a prior interrupted run

        rel_path = row["mp3_path"]
        abs_path = os.path.join(raw_dir, rel_path)
        if not os.path.exists(abs_path):
            continue

        feat_path = os.path.join(feat_dir, f"{track_id}.npz")
        if os.path.exists(feat_path):
            pass
        else:
            try:
                mel, chroma, _ = extract_features(abs_path, sr=sr, n_mels=n_mels, n_chroma=n_chroma)
            except Exception as e:
                print(f"  [skip] {rel_path}: {e}")
                continue
            bounds = fixed_window_segments(mel.shape[1], sr, window_seconds=window_seconds)
            np.savez_compressed(feat_path, mel=mel, chroma=chroma, segment_bounds=np.array(bounds))

        tags_vec = [int(row[t]) for t in top_tags]
        active = [t for t, v in zip(top_tags, tags_vec) if v]

        manifest.append({
            "track_id": track_id,
            "feature_path": feat_path,
            "caption": synthesize_caption(active),
            "caption_is_synthetic": True,
            "tags": tags_vec,
            "valence": None,
            "arousal": None,
        })
        processed_this_run += 1
        if processed_this_run % 200 == 0:
            print(f"  [{split_name}] {processed_this_run} newly processed this run "
                  f"({len(manifest)} total)...")

    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--annotations_csv", default=None)
    ap.add_argument("--feat_dir", default="/content/mtat_features")
    ap.add_argument("--splits_dir", default="data/splits")
    ap.add_argument("--top_k_tags", type=int, default=50)
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--n_mels", type=int, default=128)
    ap.add_argument("--n_chroma", type=int, default=12)
    ap.add_argument("--window_seconds", type=float, default=5.0)
    ap.add_argument("--max_per_split", type=int, default=None,
                     help="Cap clips per split (e.g. 1500) for a fast, "
                          "one-session run. Omit for the full dataset.")
    args = ap.parse_args()

    ann_csv = args.annotations_csv or os.path.join(args.raw_dir, "annotations_final.csv")
    df, top_tags = load_annotations(ann_csv, top_k=args.top_k_tags)
    print(f"Loaded {len(df)} clips, top {len(top_tags)} tags selected by frequency.")

    os.makedirs(args.splits_dir, exist_ok=True)
    for name, folders in [("train", TRAIN_FOLDERS), ("val", VAL_FOLDERS), ("test", TEST_FOLDERS)]:
        out_path = os.path.join(args.splits_dir, f"{name}.json")
        existing = load_existing_manifest(out_path)
        if existing:
            print(f"[{name}] resuming -- {len(existing)} clips already done in a prior run")

        max_clips = args.max_per_split
        manifest = build_manifest(
            df, top_tags, args.raw_dir, args.feat_dir, args.sr, args.n_mels,
            args.n_chroma, args.window_seconds, folders, max_clips, existing, name,
        )
        with open(out_path, "w") as f:
            json.dump(manifest, f)
        print(f"[{name}] {len(manifest)} tracks total (folders {sorted(folders)}) -> {out_path}")

    os.makedirs("data/processed/toy", exist_ok=True)
    tag_vocab_path = "data/processed/toy/tag_vocab.json"
    with open(tag_vocab_path, "w") as f:
        json.dump(top_tags, f)
    print(f"Tag vocab ({len(top_tags)} tags) -> {tag_vocab_path}")
    print("Remember: config_mtat.yaml already sets emotion_alpha/beta=0 "
          "for this split (MTAT has no valence/arousal labels).")


if __name__ == "__main__":
    main()
