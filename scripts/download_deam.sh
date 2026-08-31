#!/bin/bash
# DEAM (Database for Emotional Analysis of Music) -- run on your own
# machine, not this sandbox.
#
# Unlike MagnaTagATune, DEAM's exact download URLs shift between mirrors
# (official site vs. Zenodo), so this script won't hardcode a link that
# might silently 404 later. Two reliable sources as of writing:
#
#   Official: https://cvml.unige.ch/databases/DEAM/
#   Mirror:   https://zenodo.org/records/11400122
#
# From either, download:
#   - the audio (DEAM_audio.zip / "MEMD_audio")
#   - annotations/annotations averaged per song, static (whole-song
#     valence/arousal, 1-9 scale):
#       static_annotations_averaged_songs_1_2000.csv (or the two-file
#       1-2000 / 2000-2058 split depending on the release you grab)
#
# Unzip both into: data/raw/deam/audio/*.mp3
#                   data/raw/deam/annotations/static_annotations_averaged_songs_1_2000.csv
#
# Then run: python src/prepare_deam.py --raw_dir data/raw/deam

set -e
OUT_DIR="${1:-data/raw/deam}"
mkdir -p "$OUT_DIR/audio" "$OUT_DIR/annotations"
echo "This dataset requires a manual visit (URLs are not stable enough to hardcode):"
echo "  https://cvml.unige.ch/databases/DEAM/  (official)"
echo "  https://zenodo.org/records/11400122    (mirror)"
echo ""
echo "After downloading, place files as:"
echo "  $OUT_DIR/audio/*.mp3"
echo "  $OUT_DIR/annotations/static_annotations_averaged_songs_1_2000.csv"
echo ""
echo "Then run: python src/prepare_deam.py --raw_dir $OUT_DIR"
