"""Select sharp, evenly spaced keyframes from the iPhone HLG captures.

Decodes with VideoToolbox (HLG/BT.2020 -> SDR BT.709 tone mapping), samples
at --sample-fps, and keeps the sharpest frame (variance of Laplacian) in each
window so that every ring yields ~--per-video frames with motion blur filtered out.

Usage:
    python recon/extract_frames.py ~/Desktop/DL4CV_Data/IMG_58{16,18,20}.MOV \
        --out data/alma_v2 --per-video 300
"""

import argparse
import csv
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_frames:format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    info = json.loads(out.stdout)
    s = info["streams"][0]
    return int(s["width"]), int(s["height"]), float(info["format"]["duration"])


def decode(path, width, height, fps):
    """Yields (timestamp, BGR frame) at `fps`, tone mapped to SDR."""
    vf = (
        f"fps={fps},"
        f"scale_vt=w={width}:h={height}:color_matrix=bt709:"
        "color_primaries=bt709:color_transfer=bt709,"
        "hwdownload,format=p010le,format=bgr24"
    )
    cmd = ["ffmpeg", "-v", "error", "-hwaccel", "videotoolbox",
           "-hwaccel_output_format", "videotoolbox_vld",
           "-i", str(path), "-vf", vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)
    frame_bytes = width * height * 3
    i = 0
    while True:
        buf = proc.stdout.read(frame_bytes)
        if len(buf) < frame_bytes:
            break
        yield i / fps, np.frombuffer(buf, np.uint8).reshape(height, width, 3)
        i += 1
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path}")


def sharpness(img):
    gray = cv2.cvtColor(cv2.resize(img, None, fx=0.5, fy=0.5), cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/alma_v2"))
    ap.add_argument("--per-video", type=int, default=300)
    ap.add_argument("--long-side", type=int, default=1920)
    ap.add_argument("--sample-fps", type=float, default=6.0)
    ap.add_argument("--quality", type=int, default=95)
    args = ap.parse_args()

    img_dir = args.out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for ring, video in enumerate(args.videos):
        w, h, duration = probe(video)
        scale = args.long_side / max(w, h)
        ow, oh = round(w * scale / 2) * 2, round(h * scale / 2) * 2
        window = duration / args.per_video

        best = {}  # window index -> (score, t, frame)
        total = int(duration * args.sample_fps)
        for t, frame in tqdm(decode(video, ow, oh, args.sample_fps), total=total, desc=video.name):
            k = int(t // window)
            s = sharpness(frame)
            if k not in best or s > best[k][0]:
                best[k] = (s, t, frame.copy())

        scores = np.array([v[0] for v in best.values()])
        for k in sorted(best):
            s, t, frame = best[k]
            name = f"r{ring}_{video.stem}_{int(round(t * 1000)):07d}.jpg"
            cv2.imwrite(str(img_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            rows.append({"image": name, "ring": ring, "video": video.name,
                         "time_s": round(t, 3), "sharpness": round(s, 1)})
        print(f"{video.name}: kept {len(best)} frames, sharpness median "
              f"{np.median(scores):.0f} (min {scores.min():.0f})")

    with open(args.out / "frames.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} frames to {img_dir}")


if __name__ == "__main__":
    main()
