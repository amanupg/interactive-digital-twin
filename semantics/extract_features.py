"""Per-view 2D semantic features: SAM 2.1 masks, each encoded with OpenCLIP.

Two SAM passes per view: the whole frame, plus a zoomed crop around the statue
(its 3D extent from the splat, projected into the view). The statue is small in
most frames, so the zoom pass is what isolates parts like the crown, book and
scepter; its CLIP crops also come from the full-resolution frame.

For every selected training view this writes <out>/<stem>.npz with:
  part   (H, W) uint16  index of the smallest mask covering each pixel (0 = none)
  object (H, W) uint16  index of the largest mask covering each pixel, ignoring
                        masks above --max-object-area (sky / ground blobs)
  feats  (N+1, D) f16   L2-normalised CLIP embedding per mask; row 0 is zeros
Maps are at --width; lifting rescales pixel coordinates to match.

Usage:
    python semantics/extract_features.py data/alma_v2/undistorted data/alma_v2/sem2d --every 3
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import open_clip
import torch
from tqdm import tqdm
from ultralytics import SAM

from scene import Splats, in_statue_cylinder, load_views

CLIP_MODEL = ("ViT-B-16", "laion2b_s34b_b88k")


def mask_crops(img_rgb, masks, size=224, pad=0.1):
    """Square crops around each mask, background zeroed (LangSplat-style)."""
    h, w = masks.shape[1:]
    crops = []
    for m in masks:
        ys, xs = np.nonzero(m)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        side = int(max(y1 - y0, x1 - x0) * (1 + 2 * pad)) + 1
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        canvas = np.zeros((side, side, 3), np.uint8)
        sy0, sx0 = cy - side // 2, cx - side // 2
        iy0, ix0, iy1, ix1 = max(sy0, 0), max(sx0, 0), min(sy0 + side, h), min(sx0 + side, w)
        patch = img_rgb[iy0:iy1, ix0:ix1] * m[iy0:iy1, ix0:ix1, None]
        canvas[iy0 - sy0:iy1 - sy0, ix0 - sx0:ix1 - sx0] = patch
        crops.append(cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA))
    return np.stack(crops)


def segment(sam, img, min_area):
    """SAM 'everything' with a looser filter than ultralytics' defaults."""
    with torch.no_grad():
        if sam.predictor is None:
            sam(img, device="mps", verbose=False)
        p = sam.predictor
        im = p.preprocess([img])
        res = p.postprocess(p.generate(im, points_stride=32, points_batch_size=16,
                                       conf_thres=0.7, stability_score_thresh=0.85), im, [img])[0]
    h, w = img.shape[:2]
    masks = res.masks.data.cpu().numpy() if res.masks is not None else np.zeros((0, h, w), bool)
    return masks[masks.reshape(len(masks), -1).mean(1) >= min_area]


def encode(clip, norm, img_bgr, masks):
    if not len(masks):
        return np.zeros((0, clip.visual.output_dim), np.float16)
    crops = mask_crops(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), masks)
    x = torch.from_numpy(crops).to("mps").permute(0, 3, 1, 2).float() / 255
    mean, std = norm
    with torch.no_grad():
        f = torch.cat([clip.encode_image((x[i:i + 64] - mean) / std) for i in range(0, len(x), 64)])
    return torch.nn.functional.normalize(f, dim=-1).cpu().numpy().astype(np.float16)


def statue_box(view, statue_pts, pad=0.12):
    """Pixel box (x0, y0, x1, y1) around the projected statue, or None."""
    uv, z = view.project(statue_pts)
    uv = uv[z > 0]
    if len(uv) < 50:
        return None
    (x0, y0), (x1, y1) = np.percentile(uv, 1, axis=0), np.percentile(uv, 99, axis=0)
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    x0, y0 = max(int(x0 - px), 0), max(int(y0 - py), 0)
    x1, y1 = min(int(x1 + px), view.width), min(int(y1 + py), view.height)
    return (x0, y0, x1, y1) if x1 - x0 > 32 and y1 - y0 > 32 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--every", type=int, default=3, help="use every Nth image")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--sam", default="sam2.1_b.pt")
    ap.add_argument("--min-area", type=float, default=0.0003, help="fraction of image")
    ap.add_argument("--max-object-area", type=float, default=0.25)
    ap.add_argument("--splat", type=Path, default=Path("data/alma_v2/brush_30k_masked/splat_30000.ply"))
    ap.add_argument("--zoom-max-frac", type=float, default=0.5,
                    help="skip the zoom pass when the statue box already covers this much of the frame")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sam = SAM(args.sam)
    clip, _, preprocess = open_clip.create_model_and_transforms(*CLIP_MODEL, device="mps")
    clip.eval()
    norm = (torch.tensor(preprocess.transforms[-1].mean, device="mps").view(1, 3, 1, 1),
            torch.tensor(preprocess.transforms[-1].std, device="mps").view(1, 3, 1, 1))

    views = load_views(args.scene / "sparse")
    splats = Splats.load(args.splat)
    # Bronze figure + throne only: the pedestal is already well segmented full-frame.
    statue_pts = splats.means[(splats.opacity > 0.3) & in_statue_cylinder(splats.means, views, radius=1.1, h_min=-0.2)]

    paths = sorted((args.scene / "images").glob("*.jpg"))[:: args.every]
    for path in tqdm(paths):
        out_path = args.out / f"{path.stem}.npz"
        if out_path.exists():
            continue
        full = cv2.imread(str(path))
        scale = args.width / full.shape[1]
        img = cv2.resize(full, (args.width, round(full.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

        masks = segment(sam, img, args.min_area)
        feats = encode(clip, norm, img, masks)

        box = statue_box(views[path.name], statue_pts)
        n_zoom = 0
        if box and (box[2] - box[0]) * (box[3] - box[1]) < args.zoom_max_frac * full.shape[0] * full.shape[1]:
            x0, y0, x1, y1 = box
            crop = full[y0:y1, x0:x1]
            zs = 1024 / max(crop.shape[:2])
            crop = cv2.resize(crop, None, fx=zs, fy=zs, interpolation=cv2.INTER_CUBIC if zs > 1 else cv2.INTER_AREA)
            zmasks = segment(sam, crop, args.min_area)
            zfeats = encode(clip, norm, crop, zmasks)
            # Paste zoom masks back into the working frame.
            bx0, by0 = round(x0 * scale), round(y0 * scale)
            bw, bh = max(round(x1 * scale) - bx0, 1), max(round(y1 * scale) - by0, 1)
            pasted = np.zeros((len(zmasks), h, w), bool)
            for i, m in enumerate(zmasks):
                small = cv2.resize(m.astype(np.uint8), (bw, bh), interpolation=cv2.INTER_NEAREST) > 0
                pasted[i, by0:by0 + bh, bx0:bx0 + bw] = small[: h - by0, : w - bx0]
            keep = pasted.reshape(len(pasted), -1).any(1)
            masks = np.concatenate([masks, pasted[keep]])
            feats = np.concatenate([feats, zfeats[keep]])
            n_zoom = int(keep.sum())
        areas = masks.reshape(len(masks), -1).mean(1)

        # Paint largest first so smaller masks overwrite -> per-pixel smallest mask.
        part = np.zeros((h, w), np.uint16)
        for i in np.argsort(-areas):
            part[masks[i]] = i + 1
        obj = np.zeros((h, w), np.uint16)
        for i in np.argsort(areas):
            if areas[i] <= args.max_object_area:
                obj[masks[i]] = i + 1
        feats = np.concatenate([np.zeros((1, feats.shape[1]), np.float16), feats])

        np.savez_compressed(out_path, part=part, object=obj, feats=feats, n_zoom=n_zoom)


if __name__ == "__main__":
    main()
