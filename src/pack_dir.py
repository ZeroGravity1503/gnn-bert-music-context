"""
Pack a directory of many small .npz files into ONE archive file.

Why: writing thousands of separate small files directly to Google
Drive's FUSE mount is unstable (crashes with "Transport endpoint is not
connected"), so we keep those in Colab's local, ephemeral /content/
during processing -- but that means a Colab session boundary wipes them.
This script bundles everything into a SINGLE file, which IS safe to
write to Drive (one write operation, not thousands), so it survives
session restarts. Use unpack_dir.py to restore it to local disk at the
start of a new session.

    python src/pack_dir.py --in_dir /content/deam_graphs --out_file data/packed/deam_graphs.npz
"""
import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Directory of many .npz files")
    ap.add_argument("--out_file", required=True, help="Single output .npz file (goes on Drive)")
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.in_dir) if f.endswith(".npz"))
    if not files:
        print(f"No .npz files found in {args.in_dir}")
        return

    packed = {}
    for fname in files:
        track_id = fname[:-4]  # strip .npz
        data = np.load(os.path.join(args.in_dir, fname))
        packed[track_id] = {k: data[k] for k in data.files}

    os.makedirs(os.path.dirname(args.out_file) or ".", exist_ok=True)
    # allow_pickle needed since we're storing a dict-of-dicts-of-arrays
    np.savez_compressed(args.out_file, data=np.array(packed, dtype=object))
    size_mb = os.path.getsize(args.out_file) / (1024 * 1024)
    print(f"Packed {len(files)} files -> {args.out_file} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
