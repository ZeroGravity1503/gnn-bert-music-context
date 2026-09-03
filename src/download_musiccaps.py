"""
Download MusicCaps: real human captions + the 10s audio clips they
describe. Run on your own machine or Colab (needs yt-dlp + ffmpeg,
huggingface_hub, and YouTube access).

    pip install yt-dlp huggingface_hub
    python src/download_musiccaps.py --out_dir data/raw/musiccaps --cookies cookies.txt

MusicCaps ships as a CSV of (YouTube ID, start_s, end_s, caption) --
Google doesn't redistribute the audio directly, so each clip has to be
pulled from YouTube and trimmed. Expect some clips to fail (video taken
down, region-locked, etc.) -- normal for MusicCaps.

IMPORTANT: YouTube's anti-bot check ("Sign in to confirm you're not a
bot") blocks unauthenticated/datacenter-IP downloads (this affects
Colab specifically). Export real browser cookies (e.g. via the
"Get cookies.txt LOCALLY" extension) and pass --cookies path/to/cookies.txt
to get past it.
"""
import argparse
import os
import subprocess

import pandas as pd


def download_csv(out_dir):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id="google/MusicCaps", repo_type="dataset",
        filename="musiccaps-public.csv", local_dir=out_dir,
    )
    return path


def download_clip(ytid, start_s, end_s, out_path, cookies=None):
    """Pull just the needed [start_s, end_s) window via yt-dlp + ffmpeg
    download-sections, to avoid downloading full videos."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/raw/musiccaps")
    ap.add_argument("--limit", type=int, default=None,
                     help="Cap number of clips (useful for a quick first pass)")
    ap.add_argument("--cookies", default=None,
                     help="Path to a cookies.txt file (Netscape format) exported "
                          "from a logged-in browser -- needed to bypass YouTube's "
                          "anti-bot check, which otherwise blocks ALL downloads "
                          "from datacenter IPs like Colab's.")
    args = ap.parse_args()

    audio_dir = os.path.join(args.out_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    csv_path = download_csv(args.out_dir)
    df = pd.read_csv(csv_path)
    if args.limit:
        df = df.head(args.limit)

    ok, failed = 0, 0
    first_error_shown = False
    manifest_rows = []
    for i, row in df.iterrows():
        out_path = os.path.join(audio_dir, f"{row['ytid']}.mp3")
        success, error = download_clip(row["ytid"], int(row["start_s"]), int(row["end_s"]),
                                        out_path, cookies=args.cookies)
        if success:
            ok += 1
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

    manifest_path = os.path.join(args.out_dir, "downloaded_manifest.csv")
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"\nDone: {ok} downloaded, {failed} failed (out of {len(df)}).")
    print(f"Manifest -> {manifest_path}")
    print(f"Next: python src/prepare_musiccaps.py --manifest {manifest_path}")


if __name__ == "__main__":
    main()
