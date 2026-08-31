"""
Build data/splits/{train,val,test}.json from a downloaded MagnaTagATune.

Run AFTER scripts/download_magnatagatune.sh, on a machine with the audio
on disk (not this sandbox).

    python src/prepare_magnatagatune.py --raw_dir data/raw/magnatagatune
    python src/graph_builder.py   # reads data/splits/*.json, writes
                                   # data/processed/graphs/<track_id>.npz
                                   # -- unchanged from the toy-data flow

What this does (entirely self-contained -- see note below on why):
  1. Parses `annotations_final.csv` directly. This file, shipped with
     MagnaTagATune itself, already has everything needed: a `clip_id`,
     188 binary tag columns, and an `mp3_path` column like
     "8/artist-album-track-...-59-88.mp3" -- the leading character is
     the hex folder (0-9, a-f) the mp3 was unzipped into.
  2. Picks the top-50 most frequent tags (standard practice, matches
     the spec's "MagnaTagATune tag subset (top-50 tags)" suggestion in
     Task 1's deliverables).
  3. Splits by hex folder prefix: folders 0-9,a,b -> train (12/16),
     folder c -> val (1/16), folders d,e,f -> test (3/16). This is the
     standard ~12:1:3 MTAT split used across the literature (e.g. Won
     et al. 2020) -- artist/album overlap across the split is minimal
     in practice since clips from one album tend to cluster in the same
     folder, but note in your report that MTAT's folder assignment
     wasn't originally designed as an artist-disjoint split (unlike,
     say, FMA's official splits), so this is a documented limitation,
     not a guarantee.
  4. Extracts mel/chroma features (audio_features.py), written with the
     same keys as make_toy_dataset.py so graph_builder.py works
     unmodified -- graph construction stays a separate step.
  5. Synthesizes a caption per clip from its active tags (MTAT has no
     real captions) -- clearly flagged as synthetic (`caption_is_synthetic:
     true`), not passed off as human-written. For genuine free-form
     captions, use MusicCaps (prepare_musiccaps.py) for Task 4.
  6. Valence/arousal are NOT set (MTAT has no VA labels) -- set
     train.emotion_alpha/emotion_beta to 0 in config.yaml for runs on
     this split (config_mtat.yaml already does this). DEAM is a
     SEPARATE dataset/split for the VA extension (prepare_deam.py) --
     there's no genuine per-track alignment between MTAT and DEAM since
     they're different recordings, so this script does not attempt to
     fake one.

Why self-contained instead of using the widely-used
minzwon/sota-music-tagging-models split/binary/tag files: those are
distributed as row-indexed .npy arrays whose row order matches an
external, undocumented-from-outside file list -- reproducing that
alignment correctly requires their preprocessing script's exact output,
which I couldn't verify against the real files from this sandbox.
annotations_final.csv's own mp3_path column has zero ambiguity, so this
version needs nothing beyond what MagnaTagATune ships.
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


def build_manifest(df, top_tags, raw_dir, feat_dir, sr, n_mels, n_chroma, window_seconds, folders):
    os.makedirs(feat_dir, exist_ok=True)
    manifest = []
    subset = df[df["mp3_path"].str[0].isin(folders)]
    for _, row in subset.iterrows():
        rel_path = row["mp3_path"]
        abs_path = os.path.join(raw_dir, rel_path)
        if not os.path.exists(abs_path):
            continue  # skip missing files rather than crash the whole prep run

        track_id = f"mtat_{row['clip_id']}"
        try:
            mel, chroma, _ = extract_features(abs_path, sr=sr, n_mels=n_mels, n_chroma=n_chroma)
        except Exception as e:
            print(f"  [skip] {rel_path}: {e}")
            continue
        bounds = fixed_window_segments(mel.shape[1], sr, window_seconds=window_seconds)
        feat_path = os.path.join(feat_dir, f"{track_id}.npz")
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
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True,
                     help="Dir containing unzipped hex folders (0-9,a-f) and annotations_final.csv")
    ap.add_argument("--annotations_csv", default=None,
                     help="Defaults to <raw_dir>/annotations_final.csv")
    ap.add_argument("--feat_dir", default="data/processed/real/mtat_features")
    ap.add_argument("--splits_dir", default="data/splits")
    ap.add_argument("--top_k_tags", type=int, default=50)
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--n_mels", type=int, default=128)
    ap.add_argument("--n_chroma", type=int, default=12)
    ap.add_argument("--window_seconds", type=float, default=5.0)
    args = ap.parse_args()

    ann_csv = args.annotations_csv or os.path.join(args.raw_dir, "annotations_final.csv")
    df, top_tags = load_annotations(ann_csv, top_k=args.top_k_tags)
    print(f"Loaded {len(df)} clips, top {len(top_tags)} tags selected by frequency.")

    os.makedirs(args.splits_dir, exist_ok=True)
    for name, folders in [("train", TRAIN_FOLDERS), ("val", VAL_FOLDERS), ("test", TEST_FOLDERS)]:
        manifest = build_manifest(
            df, top_tags, args.raw_dir, args.feat_dir, args.sr, args.n_mels,
            args.n_chroma, args.window_seconds, folders,
        )
        out_path = os.path.join(args.splits_dir, f"{name}.json")
        with open(out_path, "w") as f:
            json.dump(manifest, f)
        print(f"[{name}] {len(manifest)} tracks (folders {sorted(folders)}) -> {out_path}")

    # written to data/processed/toy/ to match load_tag_vocab()'s hardcoded
    # default path used throughout train.py/evaluate.py -- the "toy" in
    # that path is vestigial (it's really just "the tag vocab location"),
    # kept as-is here rather than touching working code.
    os.makedirs("data/processed/toy", exist_ok=True)
    tag_vocab_path = "data/processed/toy/tag_vocab.json"
    with open(tag_vocab_path, "w") as f:
        json.dump(top_tags, f)
    print(f"Tag vocab ({len(top_tags)} tags) -> {tag_vocab_path}")
    print("Remember: config_mtat.yaml already sets emotion_alpha/beta=0 "
          "for this split (MTAT has no valence/arousal labels).")


if __name__ == "__main__":
    main()
