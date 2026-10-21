"""Open-vocabulary 3D queries over the lifted language field.

    field = LanguageField.load("data/alma_v2/language_field.npz")
    hit = field.locate("the laurel crown")
    hit.centre, hit.extent, hit.indices   # 3D answer the agent can point a camera at

Relevancy is scored on the k-means codebook (a few thousand rows), then looked
up per Gaussian, so a query over ~1M Gaussians takes milliseconds after the
text encoder. Each Gaussian holds several codes (see lift.py); its score is its
best code with at least `min_count` observations. Both levels (part/object) are
scored and the stronger one wins.

CLI (renders an overlay of the hit into a few views):
    python semantics/query.py "laurel crown" "open book" --views r2_IMG_5820_0001000
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from scene import Splats, load_views
from text import relevancy


@dataclass
class Hit:
    query: str
    level: str
    score: float           # peak relevancy
    indices: np.ndarray    # Gaussian indices (into the PLY) of the selected cluster
    centre: np.ndarray     # (3,) opacity-weighted centroid
    extent: np.ndarray     # (2, 3) robust min / max corner


class LanguageField:
    def __init__(self, data, splats):
        self.index = data["index"]
        self.codes = {lvl: data[lvl] for lvl in ("part", "object")}
        self.counts = {lvl: data[f"{lvl}_count"] for lvl in ("part", "object")}
        self.codebook = data["codebook"]
        self.views_seen = data["views"]
        self.means = splats.means[self.index]
        self.opacity = splats.opacity[self.index]

    @classmethod
    def load(cls, path, splat="data/alma_v2/brush_30k_masked/splat_30000.ply"):
        return cls(np.load(path), Splats.load(splat))

    def scores(self, query, level, min_count=3):
        rel = relevancy(self.codebook, query)
        rel[0] = 0.0  # empty slot
        per_code = rel[self.codes[level]]
        per_code[self.counts[level] < min_count] = 0.0
        return per_code.max(axis=1)

    def locate(self, query, levels=("part", "object"), top_frac=0.002, min_views=3, cluster_radius=0.25,
               peak_k=20):
        """Highest-relevancy Gaussians, reduced to their densest spatial cluster."""
        best = None
        for level in levels:
            s = self.scores(query, level)
            s[self.views_seen < min_views] = 0
            if best is None or s.max() > best[1].max():
                best = (level, s)
        level, s = best
        k = max(int(top_frac * len(s)), 50)
        top = np.argpartition(-s, k)[:k]
        top = top[s[top] > 0.5]
        if len(top) == 0:
            return Hit(query, level, float(s.max()), np.zeros(0, int), np.full(3, np.nan), np.full((2, 3), np.nan))
        # Link hits closer than cluster_radius; keep the component with the highest
        # peak relevancy (mean of its top `peak_k` scores). Ranking by total mass
        # instead favours big, mediocre clusters (a lamp post) over a small part
        # that matches strongly (the scepter).
        pts = self.means[top]
        pairs = cKDTree(pts).query_pairs(cluster_radius, output_type="ndarray")
        graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(top), len(top)))
        _, labels = connected_components(graph, directed=False)
        # Components smaller than peak_k are single noisy Gaussians, not parts.
        sizes = np.bincount(labels)
        peak = [np.sort(s[top[labels == c]])[-peak_k:].mean() if sizes[c] >= peak_k else -1
                for c in range(len(sizes))]
        members = top[labels == int(np.argmax(peak) if max(peak) > 0 else np.argmax(sizes))]
        w = self.opacity[members] * s[members]
        p = self.means[members]
        return Hit(query, level, float(s.max()), self.index[members],
                   (p * w[:, None]).sum(0) / w.sum(), np.percentile(p, [5, 95], axis=0))


def overlay(field, hit, view, image_path, scores=None):
    """Draws the hit (and optional per-Gaussian scores) into a training view."""
    img = cv2.imread(str(image_path))
    sx = img.shape[1] / view.width
    if scores is not None:
        sel = scores > 0.55
        uv, z = view.project(field.means[sel])
        ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < view.width) & (uv[:, 1] >= 0) & (uv[:, 1] < view.height)
        heat = np.zeros(img.shape[:2], np.float32)
        u = (uv[ok] * sx).astype(int)
        np.maximum.at(heat, (u[:, 1], u[:, 0]), scores[sel][ok])
        heat = cv2.dilate(heat, np.ones((7, 7), np.uint8))
        colour = cv2.applyColorMap((np.clip((heat - 0.5) / 0.2, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        img = np.where(heat[..., None] > 0, cv2.addWeighted(img, 0.4, colour, 0.6, 0), img)
    if len(hit.indices):
        c, z = view.project(hit.centre[None])
        if z[0] > 0:
            cv2.drawMarker(img, tuple((c[0] * sx).astype(int)), (0, 255, 0), cv2.MARKER_CROSS, 60, 6)
    cv2.putText(img, f"{hit.query} [{hit.level}] {hit.score:.2f}", (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 5)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("queries", nargs="+")
    ap.add_argument("--field", type=Path, default=Path("data/alma_v2/language_field.npz"))
    ap.add_argument("--scene", type=Path, default=Path("data/alma_v2/undistorted"))
    ap.add_argument("--views", nargs="+", default=["r2_IMG_5820_0001000", "r1_IMG_5818_0156167"])
    ap.add_argument("--out", type=Path, default=Path("data/alma_v2/queries.jpg"))
    args = ap.parse_args()

    field = LanguageField.load(args.field)
    views = load_views(args.scene / "sparse")
    rows = []
    for q in args.queries:
        hit = field.locate(q)
        print(f"{q!r}: level={hit.level} score={hit.score:.3f} n={len(hit.indices)} centre={np.round(hit.centre, 3)}")
        scores = field.scores(q, hit.level)
        tiles = [overlay(field, hit, views[v + ".jpg"], args.scene / "images" / f"{v}.jpg", scores)
                 for v in args.views]
        rows.append(np.hstack([cv2.resize(t, (640, 360)) for t in tiles]))
    cv2.imwrite(str(args.out), np.vstack(rows))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
