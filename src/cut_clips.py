"""Cut the long recordings into one clip per demo, using a list of timestamps.

Put the long videos in data/raw/ and write data/clips.csv like this
(times as seconds, m:ss or m:ss.s; one row per demo, in order):

    video,start,end
    long1.mp4,0:03.5,0:17.0
    long1.mp4,0:21,0:35.5
    long2.mp4,12.0,26.4

Output: data/clips/ep_001.mp4, ep_002.mp4, ... at the original resolution,
constant 30 fps, no audio, plus data/clips/episodes.csv recording where each
clip came from.

Usage:
    python src/cut_clips.py
    python src/cut_clips.py --csv data/clips.csv --raw data/raw --out data/clips
"""

import argparse
import csv
import subprocess
from pathlib import Path


def to_seconds(t):
    seconds = 0.0
    for part in t.strip().split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/clips.csv")
    p.add_argument("--raw", default="data/raw")
    p.add_argument("--out", default="data/clips")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(args.csv, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("video", "").strip()]

    index = []
    for n, row in enumerate(rows, 1):
        src = Path(args.raw) / row["video"].strip()
        start, end = to_seconds(row["start"]), to_seconds(row["end"])
        if end <= start:
            raise ValueError(f"row {n}: end ({row['end']}) is not after start ({row['start']})")
        dst = out / f"ep_{n:03d}.mp4"
        # -ss before -i plus re-encoding gives frame-accurate cuts; -r 30 removes the
        # phone's variable frame rate so frame k is always at k/30 s.
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", str(src),
             "-t", f"{end - start:.3f}", "-an", "-r", "30",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
             str(dst)],
            check=True,
        )
        index.append({"episode": dst.name, "source": src.name, "start_s": start, "end_s": end})
        print(f"{dst.name}: {end - start:5.1f}s from {src.name}")

    with open(out / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["episode", "source", "start_s", "end_s"])
        w.writeheader()
        w.writerows(index)
    print(f"\n{len(index)} clips written to {out}/")


if __name__ == "__main__":
    main()
