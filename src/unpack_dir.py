"""
Restore a packed archive (from pack_dir.py) back to many small .npz
files on local disk -- fast and reliable since it's local disk I/O,
not Drive's FUSE mount.

    python src/unpack_dir.py --in_file data/packed/deam_graphs.npz --out_dir /content/deam_graphs
"""
import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    packed = np.load(args.in_file, allow_pickle=True)["data"].item()

    for track_id, arrays in packed.items():
        out_path = os.path.join(args.out_dir, f"{track_id}.npz")
        np.savez_compressed(out_path, **arrays)

    print(f"Unpacked {len(packed)} files -> {args.out_dir}")


if __name__ == "__main__":
    main()
