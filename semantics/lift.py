"""Lift per-view CLIP mask features onto the Gaussians (training-free).

Every 2D mask embedding from semantics/extract_features.py becomes a codebook
row (or, with --codebook K, is quantised into K k-means centres; that loses weakly
matched parts such as the book, so exact is the default while it fits in uint16). For each view, candidate Gaussian centres are projected
in; a coarse z-buffer over the opaque Gaussians decides visibility, and each
visible Gaussian casts a vote for the code of the mask under it, at both levels
(part / object), weighted by mask specificity (1 / sqrt(region area)).

Each Gaussian keeps its top --top codes rather than one averaged embedding. A
small part such as the crown is isolated in only a handful of close views and
falls inside a generic "head" or "statue" mask in the many others; averaging
washes it out, while a multi-label field keeps it as one of the Gaussian's
codes. A query scores a Gaussian by its best code with enough support (query.py).

Output (<out>.npz):
  index         (M,) int32        Gaussian indices into the splat PLY
  part, object  (M, T) uint16     top codes per Gaussian (0 = empty slot)
  part_count, object_count (M, T) uint16  observations behind each code
  codebook      (K+1, D) float16  row 0 zeros
  views         (M,) uint16       number of views each Gaussian was visible in

Usage:
    python semantics/lift.py data/alma_v2/sem2d data/alma_v2/language_field.npz
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from scene import Splats, load_views, orbit_frame

LEVELS = ("part", "object")


def zbuffer_visible(uv, z, opaque, shape, cell, rel_tol, abs_tol):
    """Boolean visibility per point against a min-depth buffer of opaque points."""
    h, w = shape
    gh, gw = (h + cell - 1) // cell, (w + cell - 1) // cell
    inside = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    cx = np.clip((uv[:, 0] // cell).astype(np.int64), 0, gw - 1)
    cy = np.clip((uv[:, 1] // cell).astype(np.int64), 0, gh - 1)
    flat = cy * gw + cx
    zbuf = np.full(gh * gw, np.inf, np.float32)
    occ = inside & opaque
    np.minimum.at(zbuf, flat[occ], z[occ])
    # Spread each cell's depth to its 3x3 neighbourhood minimum so thin gaps between
    # splat centres don't let background Gaussians leak through.
    zb = zbuf.reshape(gh, gw)
    pad = np.pad(zb, 1, constant_values=np.inf)
    zb = np.min([pad[dy:dy + gh, dx:dx + gw] for dy in range(3) for dx in range(3)], axis=0).ravel()
    return inside & (z <= zb[flat] * (1 + rel_tol) + abs_tol)


class VoteTable:
    """Sparse (gaussian, code) -> (weight, count), merged in batches to bound memory."""

    def __init__(self, n_codes):
        self.n_codes = n_codes
        self.keys = np.zeros(0, np.int64)
        self.weight = np.zeros(0, np.float32)
        self.count = np.zeros(0, np.int32)
        self._pending = []

    def add(self, gauss, codes, weight):
        self._pending.append((gauss.astype(np.int64) * self.n_codes + codes, weight.astype(np.float32)))

    def merge(self):
        if not self._pending:
            return
        keys = np.concatenate([self.keys] + [k for k, _ in self._pending])
        weight = np.concatenate([self.weight] + [w for _, w in self._pending])
        count = np.concatenate([self.count] + [np.ones(len(k), np.int32) for k, _ in self._pending])
        self._pending = []
        self.keys, inv = np.unique(keys, return_inverse=True)
        self.weight = np.bincount(inv, weights=weight).astype(np.float32)
        self.count = np.bincount(inv, weights=count).astype(np.int32)

    def top(self, n_gauss, k):
        """(n_gauss, k) codes (+1, 0 = empty) and counts, ordered by vote weight."""
        self.merge()
        g, c = self.keys // self.n_codes, self.keys % self.n_codes
        order = np.lexsort((-self.weight, g))
        g, c, cnt = g[order], c[order], self.count[order]
        first = np.searchsorted(g, g, side="left")
        rank = np.arange(len(g)) - first
        sel = rank < k
        codes = np.zeros((n_gauss, k), np.uint16)
        counts = np.zeros((n_gauss, k), np.uint16)
        codes[g[sel], rank[sel]] = c[sel] + 1
        counts[g[sel], rank[sel]] = np.minimum(cnt[sel], 65535)
        return codes, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sem2d", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--scene", type=Path, default=Path("data/alma_v2/undistorted"))
    ap.add_argument("--splat", type=Path, default=Path("data/alma_v2/brush_30k_masked/splat_30000.ply"))
    ap.add_argument("--min-opacity", type=float, default=0.1)
    ap.add_argument("--max-dist", type=float, default=25.0, help="from orbit centre, SfM units")
    ap.add_argument("--occluder-opacity", type=float, default=0.5)
    ap.add_argument("--cell", type=int, default=4, help="z-buffer cell size in feature-map px")
    ap.add_argument("--rel-tol", type=float, default=0.02)
    ap.add_argument("--abs-tol", type=float, default=0.03)
    ap.add_argument("--codebook", type=int, default=12000, help="k-means size; 0 = exact (one row per mask)")
    ap.add_argument("--top", type=int, default=6, help="codes kept per Gaussian per level")
    ap.add_argument("--merge-every", type=int, default=20, help="views between vote merges")
    args = ap.parse_args()

    views = load_views(args.scene / "sparse")
    splats = Splats.load(args.splat)
    centre, _ = orbit_frame(views)
    keep = (splats.opacity > args.min_opacity) & (np.linalg.norm(splats.means - centre, axis=1) < args.max_dist)
    index = np.nonzero(keep)[0].astype(np.int32)
    means, opaque = splats.means[index], splats.opacity[index] > args.occluder_opacity
    print(f"lifting onto {len(index):,} Gaussians")

    # Shared codebook over every 2D mask embedding (both levels index the same masks).
    files = sorted(args.sem2d.glob("*.npz"))
    all_feats = [np.load(f)["feats"][1:].astype(np.float32) for f in files]
    stacked = np.concatenate(all_feats)
    if args.codebook:
        print(f"fitting {args.codebook}-entry codebook on {len(stacked):,} mask embeddings")
        km = MiniBatchKMeans(args.codebook, batch_size=8192, n_init=3, random_state=0).fit(stacked)
        centres = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
        labels = km.labels_
        print(f"  mean cosine(mask, its code) = {np.mean(np.sum(stacked * centres[labels], axis=1)):.3f}")
    else:
        assert len(stacked) < 65535, "too many masks for uint16 codes; pass --codebook"
        centres, labels = stacked, np.arange(len(stacked))
        print(f"exact codebook: {len(stacked):,} mask embeddings")
    mask_codes = np.split(labels, np.cumsum([len(f) for f in all_feats])[:-1])

    votes = {lvl: VoteTable(len(centres)) for lvl in LEVELS}
    seen = np.zeros(len(index), np.uint16)
    for i, (f, codes_of_mask) in enumerate(zip(tqdm(files), mask_codes)):
        d = np.load(f)
        view = views[f.stem + ".jpg"]
        h, w = d["part"].shape
        uv, z = view.project(means)
        uv = uv * (w / view.width)
        vis = zbuffer_visible(uv, z, opaque, (h, w), args.cell, args.rel_tol, args.abs_tol)
        idx = np.nonzero(vis)[0]
        px = np.clip(uv[idx].astype(np.int64), 0, [w - 1, h - 1])
        seen[idx] += 1
        for lvl in LEVELS:
            ids = d[lvl][px[:, 1], px[:, 0]]
            sel = ids > 0
            area = np.bincount(d[lvl].ravel(), minlength=len(codes_of_mask) + 1) / d[lvl].size
            votes[lvl].add(idx[sel], codes_of_mask[ids[sel] - 1], 1 / np.sqrt(np.maximum(area[ids[sel]], 1e-5)))
        if (i + 1) % args.merge_every == 0:
            for lvl in LEVELS:
                votes[lvl].merge()

    out = {"index": index, "views": seen,
           "codebook": np.concatenate([np.zeros((1, centres.shape[1])), centres]).astype(np.float16)}
    for lvl in LEVELS:
        codes, counts = votes[lvl].top(len(index), args.top)
        out[lvl], out[f"{lvl}_count"] = codes, counts
        print(f"{lvl}: {(codes[:, 0] > 0).sum():,}/{len(index):,} Gaussians observed; "
              f"mean filled slots {(codes > 0).sum(1).mean():.1f}/{args.top}")

    np.savez_compressed(args.out, **out)
    print(f"saved {args.out} ({args.out.stat().st_size / 1e6:.1f} MB); "
          f"median views per Gaussian {np.median(seen):.0f}")


if __name__ == "__main__":
    main()
