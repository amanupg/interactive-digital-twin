"""Plot camera centres from a COLMAP TXT model, coloured by capture ring.

The orbit plane is fitted with PCA over camera centres; its normal is taken as
"up", which is enough to check that the three rings sit at distinct heights.

Usage:
    python recon/plot_cameras.py data/alma_v2/sparse_txt data/alma_v2/cameras.png
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ])


def load_centres(model_dir):
    names, centres = [], []
    lines = [l for l in open(Path(model_dir) / "images.txt") if not l.startswith("#")]
    for line in lines[::2]:
        parts = line.split()
        q = np.array(parts[1:5], float)
        t = np.array(parts[5:8], float)
        centres.append(-qvec2rotmat(q).T @ t)
        names.append(parts[9])
    return names, np.array(centres)


def main():
    model_dir, out = sys.argv[1], sys.argv[2]
    names, C = load_centres(model_dir)
    rings = np.array([int(n.split("_")[0][1:]) for n in names])

    mu = C.mean(0)
    _, _, vt = np.linalg.svd(C - mu)
    axes = vt  # rows: two in-plane directions, then the plane normal
    P = (C - mu) @ axes.T
    # Orient "up" so the highest ring has the largest height; arbitrary sign otherwise.
    radius = np.linalg.norm(P[:, :2], axis=1)

    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5.5))
    for r in sorted(set(rings)):
        m = rings == r
        a.scatter(P[m, 0], P[m, 1], s=4, label=f"ring {r}")
        b.scatter(radius[m], P[m, 2], s=4, label=f"ring {r}")
        print(f"ring {r}: n={m.sum():3d}  height {P[m, 2].mean():+.3f} ± {P[m, 2].std():.3f}  "
              f"radius {radius[m].mean():.3f} ± {radius[m].std():.3f}  (SfM units)")
    a.set_title("Top view (orbit plane)")
    a.set_aspect("equal")
    b.set_title("Side view: radius vs height along plane normal")
    b.set_xlabel("distance from orbit centre")
    b.set_ylabel("height (sign arbitrary)")
    for ax in (a, b):
        ax.legend(markerscale=4)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
