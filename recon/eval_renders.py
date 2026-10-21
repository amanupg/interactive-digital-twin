"""Score held-out renders against ground-truth frames (PSNR / SSIM).

Renders are matched to ground truth by file stem; ground truth is resized
(INTER_AREA) to the render resolution. With --masks, metrics are computed only
over pixels the mask keeps (255), so transient objects in the ground truth
(e.g. pedestrians the model rightly omits) don't count against the render. Also writes a side-by-side strip of the
worst and best views for quick inspection.

Usage:
    python recon/eval_renders.py data/alma_v2/brush_preview/eval_5000 \
        data/alma_v2/undistorted/images --out data/alma_v2/brush_preview/metrics_5000.json
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity


def psnr(a, b, keep=None):
    err = (a.astype(np.float64) - b.astype(np.float64)) ** 2
    mse = err.mean() if keep is None else err[keep].mean()
    return 10 * np.log10(255.0**2 / mse)


def ssim(a, b, keep=None):
    score, smap = structural_similarity(a, b, channel_axis=2, data_range=255, full=True)
    return score if keep is None else smap.mean(axis=2)[keep].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("renders", type=Path)
    ap.add_argument("gt", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--masks", type=Path, help="Brush-style masks dir (255 = keep)")
    ap.add_argument("--width", type=int, help="score at this width (renders downsampled) "
                    "so runs trained at different resolutions compare like for like")
    args = ap.parse_args()

    rows = []
    for r in sorted(args.renders.glob("*.png")):
        gt_path = next(args.gt.glob(r.stem + ".*"))
        pred = cv2.imread(str(r))
        if args.width and pred.shape[1] != args.width:
            size = (args.width, round(pred.shape[0] * args.width / pred.shape[1]))
            pred = cv2.resize(pred, size, interpolation=cv2.INTER_AREA)
        gt = cv2.resize(cv2.imread(str(gt_path)), pred.shape[1::-1], interpolation=cv2.INTER_AREA)
        keep = None
        if args.masks:
            m = cv2.imread(str(args.masks / f"{r.stem}.png"), 0)
            keep = cv2.resize(m, pred.shape[1::-1], interpolation=cv2.INTER_NEAREST) > 127
        rows.append({
            "image": r.stem,
            "ring": int(r.stem.split("_")[0][1:]),
            "psnr": psnr(pred, gt, keep),
            "ssim": ssim(pred, gt, keep),
            "masked_frac": 0.0 if keep is None else float(1 - keep.mean()),
        })

    summary = {"n": len(rows), "masked": bool(args.masks), "width": args.width, "psnr": np.mean([x["psnr"] for x in rows]),
               "ssim": np.mean([x["ssim"] for x in rows]), "per_ring": {}}
    for ring in sorted({x["ring"] for x in rows}):
        sub = [x for x in rows if x["ring"] == ring]
        summary["per_ring"][ring] = {"n": len(sub), "psnr": np.mean([x["psnr"] for x in sub]),
                                     "ssim": np.mean([x["ssim"] for x in sub])}
    print(json.dumps(summary, indent=2, default=float))

    if args.out:
        args.out.write_text(json.dumps({"summary": summary, "views": rows}, indent=2, default=float))
        ranked = sorted(rows, key=lambda x: x["psnr"])
        strip = []
        for x in ranked[:2] + ranked[-2:]:
            pred = cv2.imread(str(args.renders / f"{x['image']}.png"))
            if args.width and pred.shape[1] != args.width:
                size = (args.width, round(pred.shape[0] * args.width / pred.shape[1]))
                pred = cv2.resize(pred, size, interpolation=cv2.INTER_AREA)
            gt = cv2.resize(cv2.imread(str(next(args.gt.glob(x["image"] + ".*")))),
                            pred.shape[1::-1], interpolation=cv2.INTER_AREA)
            row = np.hstack([gt, pred])
            cv2.putText(row, f"{x['image']}  PSNR {x['psnr']:.1f}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
            strip.append(cv2.resize(row, None, fx=0.5, fy=0.5))
        cv2.imwrite(str(args.out.with_suffix(".jpg")), np.vstack(strip), [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    main()
