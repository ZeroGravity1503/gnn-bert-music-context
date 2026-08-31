"""
Audio preprocessing for real data (Section 3, steps 1-2 of the project spec).

  1. Resample to 22,050 Hz; extract log-mel spectrogram (128 bins) and
     chroma (12 bins); normalize per track.
  2. Segment each track into fixed 5-10s windows (beat-synchronous
     segmentation is also provided as an option) using librosa.

Run standalone once you have real audio in `data/raw/<dataset>/`:

    python src/audio_features.py --audio_dir data/raw/fma_medium \
        --out_dir data/processed/real/features --manifest data/raw/fma_medium/tracks.csv

This produces one .npz per track with the same schema as
`make_toy_dataset.py` (mel, chroma, segment_bounds), so graph_builder.py
and datasets.py work unchanged on real data -- just point config.yaml's
`data.mode` at "real" and re-point splits_dir/toy_dir accordingly.
"""
import argparse
import glob
import json
import os

import librosa
import numpy as np


def extract_features(path, sr=22050, n_mels=128, n_chroma=12):
    """Load an audio file and return normalized log-mel + chroma arrays."""
    y, _ = librosa.load(path, sr=sr, mono=True)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    # per-track normalization (zero mean, unit var)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=n_chroma)
    chroma = chroma / (chroma.sum(axis=0, keepdims=True) + 1e-8)  # column-normalize

    return log_mel.astype(np.float32), chroma.astype(np.float32), sr


def fixed_window_segments(n_frames, sr, hop_length=512, window_seconds=5.0):
    """Fixed-length windows, in frame indices. Simple and dataset-agnostic."""
    frames_per_window = int(window_seconds * sr / hop_length)
    bounds = list(range(0, n_frames, max(frames_per_window, 1)))
    if bounds[-1] != n_frames:
        bounds.append(n_frames)
    if len(bounds) < 2:
        bounds = [0, n_frames]
    return bounds


def beat_synchronous_segments(path, sr=22050):
    """Alternative: segment on detected beats rather than fixed windows."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    bounds = [0] + list(beat_frames) 
    return sorted(set(int(b) for b in bounds))


def process_file(path, out_dir, sr=22050, n_mels=128, n_chroma=12,
                  window_seconds=5.0, segmentation="fixed"):
    track_id = os.path.splitext(os.path.basename(path))[0]
    mel, chroma, sr = extract_features(path, sr, n_mels, n_chroma)
    n_frames = mel.shape[1]

    if segmentation == "beat":
        bounds = beat_synchronous_segments(path, sr)
        bounds = [b for b in bounds if b < n_frames] + [n_frames]
    else:
        bounds = fixed_window_segments(n_frames, sr, window_seconds=window_seconds)

    out_path = os.path.join(out_dir, f"{track_id}.npz")
    np.savez_compressed(
        out_path, mel=mel, chroma=chroma, segment_bounds=np.array(bounds)
    )
    return track_id, out_path, n_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", required=True,
                     help="Directory of .wav/.mp3 files (e.g. FMA-medium)")
    ap.add_argument("--out_dir", default="data/processed/real/features")
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--n_mels", type=int, default=128)
    ap.add_argument("--n_chroma", type=int, default=12)
    ap.add_argument("--window_seconds", type=float, default=5.0)
    ap.add_argument("--segmentation", choices=["fixed", "beat"], default="fixed")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(
        glob.glob(os.path.join(args.audio_dir, "**", "*.mp3"), recursive=True)
        + glob.glob(os.path.join(args.audio_dir, "**", "*.wav"), recursive=True)
    )
    if not files:
        print(f"No audio files found under {args.audio_dir}")
        return

    manifest = []
    for i, path in enumerate(files):
        try:
            track_id, out_path, n_frames = process_file(
                path, args.out_dir, args.sr, args.n_mels, args.n_chroma,
                args.window_seconds, args.segmentation,
            )
            manifest.append({"track_id": track_id, "feature_path": out_path,
                              "n_frames": n_frames})
        except Exception as e:
            print(f"  [skip] {path}: {e}")
        if (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{len(files)}")

    manifest_path = os.path.join(args.out_dir, "features_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Done. {len(manifest)}/{len(files)} tracks -> {args.out_dir}")
    print(f"Feature manifest: {manifest_path}")
    print("Next: merge this with your tag/caption/valence-arousal labels "
          "into data/splits/{train,val,test}.json (see make_toy_dataset.py "
          "for the expected schema), then run graph_builder.py.")


if __name__ == "__main__":
    main()
