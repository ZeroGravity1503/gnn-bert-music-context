"""
Download MusicCaps: real human captions + the 10s audio clips they
describe. Run on your own machine or Colab (needs yt-dlp + ffmpeg,
huggingface_hub, and YouTube access).

    pip install yt-dlp huggingface_hub
    python src/download_musiccaps.py --out_dir /content/musiccaps_raw \
        --cookies cookies.txt --checkpoint_dir data/musiccaps_checkpoint

CHECKPOINTING: this can take an hour+ due to YouTube rate-limiting, and
Colab sessions can vanish mid-run (confirmed the hard way). Every
--checkpoint_every clips, this bundles everything downloaded SO FAR into
a single tar.gz + manifest on --checkpoint_dir (Drive-safe, one big file
write, not thousands of small ones) and OVERWRITES the previous
checkpoint. If the script dies and you restart it with the same
--checkpoint_dir, it automatically resumes from the last checkpoint
instead of starting over.

IMPORTANT: YouTube's anti-bot check ("Sign in to confirm you're not a
bot") blocks unauthenticated/datacenter-IP downloads. Export real
browser cookies (e.g. via the "Get cookies.txt LOCALLY" extension) and
pass --cookies path/to/cookies.txt.
"""
import argparse
import os
import subprocess
import tarfile

import pandas as pd


def download_csv(out_dir):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id="google/MusicCaps", repo_type="dataset",
        filename="musiccaps-public.csv", local_dir=out_dir,
    )
    return path


def download_clip(ytid, start_s, end_s, out_path, cookies=None):
    if os.path.exists(out_path):
        return True, None
    url = f"https://www.youtube.com/watch?v={ytid}"
    section = f"*{start_s}-{end_s}"
    cmd = [
        "yt-dlp", "-f", "bestaudio", "-x", "--audio-format", "mp3",
        "--download-sections", section, "--force-keyframes-at-cuts",
        "-o", out_path, url,
    ]
    if cookies:
        cmd += ["--cookies", cookies]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    success = result.returncode == 0 and os.path.exists(out_path)
    return success, (result.stderr[-500:] if not success else None)


def save_checkpoint(audio_dir, manifest_rows, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    tar_path = os.path.join(checkpoint_dir, "audio.tar.gz")
    tmp_tar_path = tar_path + ".tmp"
    with tarfile.open(tmp_tar_path, "w:gz") as tar:
        tar.add(audio_dir, arcname="audio")
    os.replace(tmp_tar_path, tar_path)  # atomic-ish: don't corrupt a good checkpoint if interrupted mid-write

    manifest_path = os.path.join(checkpoint_dir, "manifest.csv")
    tmp_manifest_path = manifest_path + ".tmp"
    pd.DataFrame(manifest_rows).to_csv(tmp_manifest_path, index=False)
    os.replace(tmp_manifest_path, manifest_path)
    print(f"  [checkpoint] saved {len(manifest_rows)} clips -> {checkpoint_dir}")


def load_checkpoint(audio_dir, checkpoint_dir):
    tar_path = os.path.join(checkpoint_dir, "audio.tar.gz")
    manifest_path = os.path.join(checkpoint_dir, "manifest.csv")
    if not (os.path.exists(tar_path) and os.path.exists(manifest_path)):
        return []
    print(f"Resuming from checkpoint at {checkpoint_dir}...")
    os.makedirs(audio_dir, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(os.path.dirname(audio_dir) or ".")
    manifest_rows = pd.read_csv(manifest_path).to_dict("records")
    print(f"  Restored {len(manifest_rows)} previously-downloaded clips.")
    return manifest_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/raw/musiccaps")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cookies", default=None)
    ap.add_argument("--checkpoint_dir", default=None,
                     help="Drive-safe dir to periodically save progress to, "
                          "and to resume from if this script is re-run.")
    ap.add_argument("--checkpoint_every", type=int, default=300,
                     help="Save a checkpoint every N successful downloads.")
    args = ap.parse_args()

    audio_dir = os.path.join(args.out_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    manifest_rows = []
    done_ytids = set()
    if args.checkpoint_dir:
        manifest_rows = load_checkpoint(audio_dir, args.checkpoint_dir)
        done_ytids = {row["ytid"] for row in manifest_rows}

    csv_path = download_csv(args.out_dir)
    df = pd.read_csv(csv_path)
    if args.limit:
        df = df.head(args.limit)

    ok, failed = len(manifest_rows), 0
    first_error_shown = False
    since_last_checkpoint = 0
    for i, row in df.iterrows():
        if row["ytid"] in done_ytids:
            continue  # already have this one from a prior checkpoint

        out_path = os.path.join(audio_dir, f"{row['ytid']}.mp3")
        success, error = download_clip(row["ytid"], int(row["start_s"]), int(row["end_s"]),
                                        out_path, cookies=args.cookies)
        if success:
            ok += 1
            since_last_checkpoint += 1
            manifest_rows.append({
                "ytid": row["ytid"], "audio_path": out_path,
                "caption": row["caption"],
            })
        else:
            failed += 1
            if not first_error_shown and error:
                print(f"  [first failure detail] {error}")
                first_error_shown = True

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(df)} processed ({ok} ok, {failed} failed)")

        if args.checkpoint_dir and since_last_checkpoint >= args.checkpoint_every:
            save_checkpoint(audio_dir, manifest_rows, args.checkpoint_dir)
            since_last_checkpoint = 0

    if args.checkpoint_dir and manifest_rows:
        save_checkpoint(audio_dir, manifest_rows, args.checkpoint_dir)  # final checkpoint

    manifest_path = os.path.join(args.out_dir, "downloaded_manifest.csv")
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"\nDone: {ok} downloaded, {failed} failed (out of {len(df)}).")
    print(f"Manifest -> {manifest_path}")
    print(f"Next: python src/prepare_musiccaps.py --manifest {manifest_path}")


if __name__ == "__main__":
    main()
