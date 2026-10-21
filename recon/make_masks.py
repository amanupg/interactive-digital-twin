"""Transient-object masks (pedestrians, bikes, bags...) for splat training.

YOLO instance segmentation proposes people/vehicles/etc. The catch at this
site: Alma Mater is herself a seated human figure, so a person detector fires
on the statue. Two SfM cues separate the statue from passers-by:

  * Statue region. The cameras orbit the statue, so it sits on the orbit
    axis. SfM points inside a vertical cylinder around that axis are projected
    into each view; their (dilated) convex hull is the statue region.
  * Track support. Static geometry carries triangulated keypoints; people
    moving through the scene almost never produce consistent multi-view tracks.

A detection is kept as static only if >= --statue-overlap of it lies in the
statue region, it covers >= --statue-min-area of the frame (the statue fills
>= ~2% of every view, while passers-by seen behind it are small), AND its
triangulated-keypoint density is >= --static-ratio times the image average. Everything else is masked: over-masking a static patch
costs a little supervision, masking the statue costs the reconstruction.

Output follows Brush's convention: <scene>/masks/<stem>.png, single channel,
255 = train on this pixel, 0 = ignore.

Usage:
    python recon/make_masks.py data/alma_v2/undistorted
"""

import argparse
import csv
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# COCO ids: person, bicycle, car, motorcycle, bus, truck, dog, backpack, umbrella, handbag, suitcase
TRANSIENT_CLASSES = [0, 1, 2, 3, 5, 7, 16, 24, 25, 26, 28]


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ])


def load_model(sparse_dir):
    """Reads a PINHOLE COLMAP model.

    Returns (K, views, points) where views maps image name -> dict(R, t, kp) with
    kp the (N, 2) keypoints that have a 3D point, and points is (M, 3).
    """
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["colmap", "model_converter", "--input_path", str(sparse_dir),
                        "--output_path", tmp, "--output_type", "TXT"],
                       check=True, capture_output=True)
        read = lambda f: [l for l in open(Path(tmp) / f) if not l.startswith("#")]
        cam, images, points = read("cameras.txt"), read("images.txt"), read("points3D.txt")
    fx, fy, cx, cy = map(float, cam[0].split()[4:8])
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    views = {}
    for header, pts in zip(images[::2], images[1::2]):
        p = header.split()
        v = np.array(pts.split(), float).reshape(-1, 3) if pts.strip() else np.zeros((0, 3))
        views[p[9]] = {"R": qvec2rotmat(np.array(p[1:5], float)),
                       "t": np.array(p[5:8], float), "kp": v[v[:, 2] >= 0, :2]}
    return K, views, np.array([l.split()[1:4] for l in points], float)


def statue_points(views, points, radius, h_min, h_max):
    """SfM points in a vertical cylinder around the orbit axis.

    Up is the mean camera up vector (phone held upright); the axis passes through
    the mean camera centre; heights are relative to mean camera height.
    """
    centres = np.array([-v["R"].T @ v["t"] for v in views.values()])
    up = np.mean([v["R"].T @ np.array([0, -1.0, 0]) for v in views.values()], axis=0)
    up /= np.linalg.norm(up)
    d = points - centres.mean(0)
    h = d @ up
    r = np.linalg.norm(d - np.outer(h, up), axis=1)
    return points[(r < radius) & (h > h_min) & (h < h_max)]


def statue_region(view, K, pts3d, shape, pad):
    h, w = shape
    cam = pts3d @ view["R"].T + view["t"]
    cam = cam[cam[:, 2] > 0]
    uv = (cam @ K.T)[:, :2] / cam[:, 2:3]
    uv = uv[(uv[:, 0] > -w) & (uv[:, 0] < 2 * w) & (uv[:, 1] > -h) & (uv[:, 1] < 2 * h)]
    region = np.zeros((h, w), np.uint8)
    if len(uv) >= 3:
        cv2.fillConvexPoly(region, cv2.convexHull(uv.astype(np.int32)), 1)
        region = cv2.dilate(region, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1,) * 2))
    return region > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", type=Path, help="COLMAP scene with images/ and sparse/")
    ap.add_argument("--model", default="yolo11l-seg.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--static-ratio", type=float, default=0.05)
    ap.add_argument("--statue-overlap", type=float, default=0.5)
    ap.add_argument("--statue-min-area", type=float, default=0.01, help="fraction of frame")
    ap.add_argument("--statue-radius", type=float, default=1.3, help="SfM units")
    ap.add_argument("--statue-hmin", type=float, default=-1.0, help="rel. to camera height")
    ap.add_argument("--statue-hmax", type=float, default=1.8, help="rel. to camera height")
    ap.add_argument("--statue-pad", type=int, default=20, help="px")
    ap.add_argument("--dilate", type=int, default=15, help="dilation radius in px")
    ap.add_argument("--preview", type=int, default=24, help="number of overlay previews to save")
    args = ap.parse_args()

    img_dir, mask_dir = args.scene / "images", args.scene / "masks"
    preview_dir = args.scene / "masks_preview"
    mask_dir.mkdir(exist_ok=True)
    preview_dir.mkdir(exist_ok=True)

    K, views, points = load_model(args.scene / "sparse")
    statue = statue_points(views, points, args.statue_radius, args.statue_hmin, args.statue_hmax)
    print(f"{len(statue)} SfM points define the statue region")
    model = YOLO(args.model)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * args.dilate + 1,) * 2)

    paths = sorted(img_dir.glob("*.jpg"))
    preview_every = max(1, len(paths) // args.preview)
    rows = []
    for i, path in enumerate(tqdm(paths)):
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        view = views.get(path.name)
        kp = view["kp"] if view else np.zeros((0, 2))
        in_statue = statue_region(view, K, statue, (h, w), args.statue_pad) if view else np.zeros((h, w), bool)
        kp_mask = np.zeros((h, w), bool)
        xi = np.clip(kp[:, 0].astype(int), 0, w - 1)
        yi = np.clip(kp[:, 1].astype(int), 0, h - 1)
        kp_mask[yi, xi] = True
        img_density = len(kp) / (h * w)

        res = model.predict(img, imgsz=args.imgsz, conf=args.conf, classes=TRANSIENT_CLASSES,
                            retina_masks=True, verbose=False, device="mps")[0]
        ignore = np.zeros((h, w), bool)
        if res.masks is not None:
            inst = res.masks.data.cpu().numpy() > 0.5
            for m, cls, conf in zip(inst, res.boxes.cls.tolist(), res.boxes.conf.tolist()):
                area = m.sum()
                ratio = (kp_mask & m).sum() / max(area, 1) / max(img_density, 1e-12)
                overlap = (in_statue & m).sum() / max(area, 1)
                static = (overlap >= args.statue_overlap and ratio >= args.static_ratio
                          and area / (h * w) >= args.statue_min_area)
                rows.append({"image": path.name, "cls": int(cls), "conf": round(conf, 3),
                             "area_frac": round(area / (h * w), 5), "kp_ratio": round(float(ratio), 3),
                             "statue_overlap": round(float(overlap), 3), "static": int(static)})
                if not static:
                    ignore |= m

        ignore = cv2.dilate(ignore.astype(np.uint8), kernel) > 0
        cv2.imwrite(str(mask_dir / f"{path.stem}.png"), np.where(ignore, 0, 255).astype(np.uint8))

        if i % preview_every == 0:
            vis = img.copy()
            contours, _ = cv2.findContours(in_statue.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (255, 200, 0), 3)
            vis[ignore] = (0.4 * vis[ignore] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
            if res.masks is not None:
                for box, r in zip(res.boxes.xyxy.cpu().numpy().astype(int), rows[-len(res.boxes):]):
                    color = (0, 255, 0) if r["static"] else (0, 0, 255)
                    cv2.rectangle(vis, tuple(box[:2]), tuple(box[2:]), color, 3)
                    cv2.putText(vis, f"kp {r['kp_ratio']:.2f} ov {r['statue_overlap']:.2f}", (box[0], box[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.imwrite(str(preview_dir / f"{path.stem}.jpg"), cv2.resize(vis, None, fx=0.5, fy=0.5))

    with open(args.scene / "masks_instances.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "cls", "conf", "area_frac", "kp_ratio", "statue_overlap", "static"])
        writer.writeheader()
        writer.writerows(rows)

    n_masked = sum(1 for p in mask_dir.glob("*.png") if cv2.imread(str(p), 0).min() == 0)
    n_static = sum(r["static"] for r in rows)
    print(f"{len(rows)} detections: {len(rows) - n_static} masked as transient, {n_static} kept as static")
    print(f"{n_masked}/{len(paths)} images have masked pixels")


if __name__ == "__main__":
    main()
