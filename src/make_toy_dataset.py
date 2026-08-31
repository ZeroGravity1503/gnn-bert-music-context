"""
Generates a small synthetic dataset that mirrors the *schema* of the real
datasets (FMA / MagnaTagATune / MusicCaps / DEAM) so the rest of the
pipeline (audio_features -> graph_builder -> train -> evaluate) can be
built and unit-tested without needing network access to the real hosts.

Swap this out for real downloads before final submission -- see README.md.

Each "track" gets:
  - a synthetic log-mel spectrogram (stand-in for real audio features)
  - a synthetic chroma sequence (for chord/segment graph construction)
  - a natural-language caption (stand-in for MusicCaps)
  - multi-label tags drawn from a fixed vocabulary (stand-in for
    MagnaTagATune / FMA top tags)
  - valence/arousal targets (stand-in for DEAM)
"""
import json
import os
import random

import numpy as np
import yaml

random.seed(42)
np.random.seed(42)

TAG_VOCAB = [
    "jazz", "rock", "electronic", "classical", "hiphop", "acoustic",
    "melancholic", "energetic", "calm", "aggressive", "happy", "sad",
    "instrumental", "vocal", "distorted_guitar", "piano", "synth",
    "fast_tempo", "slow_tempo", "ambient",
]

CAPTION_TEMPLATES = [
    "A {mood} {genre} track with prominent {instrument} and a {tempo} tempo.",
    "This {tempo} {genre} piece feels {mood}, driven by {instrument}.",
    "An {mood} {genre} song featuring {instrument} throughout.",
]

MOODS = ["melancholic", "energetic", "calm", "aggressive", "happy", "uplifting"]
GENRES = ["jazz", "rock", "electronic", "classical", "hip-hop", "acoustic"]
INSTRUMENTS = ["piano", "distorted guitar", "synthesizer", "strings", "drums"]
TEMPOS = ["fast", "slow", "mid-tempo"]


def make_track(idx, n_frames=130, n_segments=8):
    # log-mel spectrogram stand-in: (n_mels, n_frames)
    mel = np.random.randn(128, n_frames).astype(np.float32)
    # chroma stand-in: (12, n_frames), softmax-ish so it looks like real chroma
    chroma_raw = np.random.randn(12, n_frames).astype(np.float32)
    chroma = np.exp(chroma_raw) / np.exp(chroma_raw).sum(axis=0, keepdims=True)

    mood = random.choice(MOODS)
    genre = random.choice(GENRES)
    instrument = random.choice(INSTRUMENTS)
    tempo = random.choice(TEMPOS)
    caption = random.choice(CAPTION_TEMPLATES).format(
        mood=mood, genre=genre, instrument=instrument, tempo=tempo
    )

    # correlate tags loosely with the caption so the task is learnable
    tags = set()
    genre_tag = genre.replace("-", "").replace(" ", "")
    if genre_tag in TAG_VOCAB:
        tags.add(genre_tag)
    if mood in TAG_VOCAB:
        tags.add(mood)
    if tempo == "fast":
        tags.add("fast_tempo")
    elif tempo == "slow":
        tags.add("slow_tempo")
    if "guitar" in instrument:
        tags.add("distorted_guitar")
    if "piano" in instrument:
        tags.add("piano")
    if "synth" in instrument:
        tags.add("synth")
    # add a little noise so it's not trivially deterministic
    if random.random() < 0.15:
        tags.add(random.choice(TAG_VOCAB))

    tag_vector = [1 if t in tags else 0 for t in TAG_VOCAB]

    # valence/arousal in [1, 9], loosely tied to mood (DEAM-style)
    mood_va = {
        "melancholic": (2.5, 3.5), "energetic": (6.5, 8.0), "calm": (6.0, 2.5),
        "aggressive": (3.0, 8.5), "happy": (7.5, 6.0), "uplifting": (7.0, 6.5),
    }
    v_base, a_base = mood_va[mood]
    valence = float(np.clip(np.random.normal(v_base, 0.5), 1, 9))
    arousal = float(np.clip(np.random.normal(a_base, 0.5), 1, 9))

    # segment boundaries for segment-graph construction
    bounds = sorted(random.sample(range(1, n_frames - 1), n_segments - 1))
    segments = [0] + bounds + [n_frames]

    return {
        "track_id": f"toy_{idx:04d}",
        "mel": mel,
        "chroma": chroma,
        "segment_bounds": segments,
        "caption": caption,
        "tags": tag_vector,
        "valence": valence,
        "arousal": arousal,
    }


def main(n_tracks=200, out_dir="data/processed/toy"):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "features"), exist_ok=True)

    manifest = []
    for i in range(n_tracks):
        t = make_track(i)
        feat_path = os.path.join(out_dir, "features", f"{t['track_id']}.npz")
        np.savez_compressed(
            feat_path,
            mel=t["mel"],
            chroma=t["chroma"],
            segment_bounds=np.array(t["segment_bounds"]),
        )
        manifest.append({
            "track_id": t["track_id"],
            "feature_path": feat_path,
            "caption": t["caption"],
            "tags": t["tags"],
            "valence": t["valence"],
            "arousal": t["arousal"],
        })

    random.shuffle(manifest)
    n = len(manifest)
    n_train, n_val = int(n * 0.7), int(n * 0.15)
    splits = {
        "train": manifest[:n_train],
        "val": manifest[n_train:n_train + n_val],
        "test": manifest[n_train + n_val:],
    }

    os.makedirs("data/splits", exist_ok=True)
    for split_name, items in splits.items():
        with open(f"data/splits/{split_name}.json", "w") as f:
            json.dump(items, f, indent=2)

    with open(os.path.join(out_dir, "tag_vocab.json"), "w") as f:
        json.dump(TAG_VOCAB, f, indent=2)

    print(f"Generated {n} toy tracks -> {out_dir}")
    print(f"Splits: train={len(splits['train'])}, val={len(splits['val'])}, "
          f"test={len(splits['test'])}")


if __name__ == "__main__":
    main()
