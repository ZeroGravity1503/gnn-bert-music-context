"""
Build DEAM splits for Task 3's emotion-regression extension.

Matches DEAM's REAL on-disk layout (confirmed against an actual download,
2026 release):
  <raw_dir>/MEMD_audio/<song_id>.mp3                          (1,802 files)
  <raw_dir>/annotations/annotations averaged per song/song_level/
      static_annotations_averaged_songs_1_2000.csv
      static_annotations_averaged_songs_2000_2058.csv
  (two files, split by song ID range -- concatenated here to cover all 1,802)

    python src/prepare_deam.py --raw_dir /content/deam_raw \
        --feat_dir /content/deam_features --splits_dir data/splits_deam

IMPORTANT -- why this is a SEPARATE split, not merged with MagnaTagATune:
DEAM's 1,802 clips and MagnaTagATune's clips are different songs by
different artists -- there's no genuine per-track alignment. Train Task 3
on MagnaTagATune with emotion_alpha=beta=0 (tag+text only), and
separately on this DEAM split with tag_weight=0, emotion_alpha/beta>0
(pure emotion regression) -- a disclosed, deliberate design choice, not
a hidden shortcut. See config_deam.yaml.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from audio_features import extract_features, fixed_window_segments


def load_static_annotations(raw_dir):
    """Finds and concatenates BOTH static annotation files (DEAM splits
    them by song ID range), wherever they are nested under raw_dir."""
    pattern = os.path.join(raw_dir, "**", "static_annotations_averaged_songs_*.csv")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(
            f"No static_annotations_averaged_songs_*.csv found under {raw_dir}. "
            f"Check that DEAM_Annotations.zip was extracted there."
        )
    print(f"Found {len(files)} annotation file(s): {[os.path.basename(f) for f in files]}")

    dfs = []
    for path in files:
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]  # DEAM's CSVs have leading-space column names
        rename = {}
        for c in df.columns:
            cl = c.strip().lower()
            if "valence" in cl and "mean" in cl:
                rename[c] = "valence"
            elif "arousal" in cl and "mean" in cl:
                rename[c] = "arousal"
            elif cl in ("song_id", "songid", "id"):
                rename[c] = "song_id"
        df = df.rename(columns=rename)
        dfs.append(df[["song_id", "valence", "arousal"]])

    combined = pd.concat(dfs, ignore_index=True).set_index("song_id")
    print(f"Total songs with valence/arousal labels: {len(combined)}")
    return combined


def find_audio_dir(raw_dir):
    """DEAM's audio folder is named MEMD_audio -- search for it rather
    than hardcode, in case a future release renames it."""
    candidates = glob.glob(os.path.join(raw_dir, "**", "MEMD_audio"), recursive=True)
    if candidates:
        return candidates[0]
    # fallback: any folder directly containing a bunch of numeric .mp3 files
    for root, dirs, files in os.walk(raw_dir):
        mp3s = [f for f in files if f.endswith(".mp3")]
        if len(mp3s) > 100:
            return root
    raise FileNotFoundError(f"Could not find DEAM's audio folder under {raw_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--feat_dir", default="/content/deam_features")
    ap.add_argument("--splits_dir", default="data/splits_deam")
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--n_mels", type=int, default=128)
    ap.add_argument("--n_chroma", type=int, default=12)
    ap.add_argument("--window_seconds", type=float, default=5.0)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_clips", type=int, default=None,
                     help="Optional cap for a faster run, e.g. 500")
    args = ap.parse_args()

    va = load_static_annotations(args.raw_dir)
    audio_dir = find_audio_dir(args.raw_dir)
    print(f"Audio dir: {audio_dir}")

    audio_files = sorted(glob.glob(os.path.join(audio_dir, "*.mp3")))
    if args.max_clips:
        audio_files = audio_files[:args.max_clips]
    if not audio_files:
        print(f"No audio found under {audio_dir}")
        return

    os.makedirs(args.feat_dir, exist_ok=True)
    items = []
    for i, path in enumerate(audio_files):
        song_id_str = os.path.splitext(os.path.basename(path))[0]
        try:
            song_id = int(song_id_str)
        except ValueError:
            song_id = song_id_str
        if song_id not in va.index:
            continue

        feat_path = os.path.join(args.feat_dir, f"{song_id_str}.npz")
        if not os.path.exists(feat_path):  # resume-safe, same pattern as MTAT script
            try:
                mel, chroma, _ = extract_features(path, sr=args.sr, n_mels=args.n_mels, n_chroma=args.n_chroma)
            except Exception as e:
                print(f"  [skip] {path}: {e}")
                continue
            bounds = fixed_window_segments(mel.shape[1], args.sr, window_seconds=args.window_seconds)
            np.savez_compressed(feat_path, mel=mel, chroma=chroma, segment_bounds=np.array(bounds))

        items.append({
            "track_id": f"deam_{song_id_str}",
            "feature_path": feat_path,
            "caption": "an instrumental clip rated for valence and arousal.",
            "caption_is_synthetic": True,
            "tags": None,  # DEAM has no tag labels -- config_deam.yaml sets tag_weight=0
            "valence": float(va.loc[song_id, "valence"]),
            "arousal": float(va.loc[song_id, "arousal"]),
        })
        if (i + 1) % 200 == 0:
            print(f"  processed {i + 1}/{len(audio_files)}")

    print(f"Total usable items (audio + label match): {len(items)}")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(items))
    n_test = int(len(items) * args.test_frac)
    n_val = int(len(items) * args.val_frac)
    test_idx = set(perm[:n_test])
    val_idx = set(perm[n_test:n_test + n_val])

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

    print("\nNOTE: this is a random split (DEAM has no standard official "
          "train/val/test partition) -- document that choice in your report.")
    print("Remember: config_deam.yaml sets tag_weight=0, emotion_alpha/beta>0 for this split.")


if __name__ == "__main__":
    main()
