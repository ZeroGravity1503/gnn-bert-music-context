#!/bin/bash
# Downloads MagnaTagATune audio + annotations. Run on YOUR machine or
# Colab -- the host (mi.soi.city.ac.uk) is not reachable from the
# sandbox this repo was built in.
#
# Usage: bash scripts/download_magnatagatune.sh data/raw/magnatagatune

set -e
OUT_DIR="${1:-data/raw/magnatagatune}"
mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

echo "Downloading MagnaTagATune mp3 archive (3 parts, ~3GB total)..."
BASE="http://mi.soi.city.ac.uk/datasets/magnatagatune"
wget -c "$BASE/mp3.zip.001"
wget -c "$BASE/mp3.zip.002"
wget -c "$BASE/mp3.zip.003"
wget -c "$BASE/annotations_final.csv"

echo "Merging zip parts..."
cat mp3.zip.001 mp3.zip.002 mp3.zip.003 > mp3_all.zip
unzip -q mp3_all.zip
# This produces 16 folders '0'-'9','a'-'f' of mp3s.

echo "Done. Audio in $OUT_DIR/<hex folder>/*.mp3, annotations in $OUT_DIR/annotations_final.csv"
echo "Next: python src/prepare_magnatagatune.py --raw_dir $OUT_DIR"
echo "(prepare_magnatagatune.py reads annotations_final.csv directly --"
echo " no separate split download needed, it builds the standard 12:1:3"
echo " hex-folder split itself.)"
