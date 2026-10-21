"""Shared loaders for the semantic layer: Brush splat PLYs and COLMAP cameras."""

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SH_C0 = 0.28209479177387814


def read_ply(path):
    """Reads a binary little-endian float PLY into a dict of name -> array.

    Brush writes properties in alphabetical order, so everything is by name.
    """
    with open(path, "rb") as f:
        names, n = [], 0
        while (line := f.readline().decode().strip()) != "end_header":
            parts = line.split()
            if parts[:2] == ["element", "vertex"]:
                n = int(parts[2])
            elif parts[0] == "property":
                assert parts[1] == "float", line
                names.append(parts[2])
        data = np.fromfile(f, dtype=np.float32, count=n * len(names)).reshape(n, len(names))
    return {name: data[:, i] for i, name in enumerate(names)}


@dataclass
class Splats:
    means: np.ndarray      # (N, 3)
    opacity: np.ndarray    # (N,) in [0, 1]
    scales: np.ndarray     # (N, 3) linear
    rgb: np.ndarray        # (N, 3) base colour in [0, 1] from the DC term

    @classmethod
    def load(cls, path):
        p = read_ply(path)
        dc = np.stack([p["f_dc_0"], p["f_dc_1"], p["f_dc_2"]], 1)
        return cls(
            means=np.stack([p["x"], p["y"], p["z"]], 1),
            opacity=1 / (1 + np.exp(-p["opacity"])),
            scales=np.exp(np.stack([p["scale_0"], p["scale_1"], p["scale_2"]], 1)),
            rgb=np.clip(dc * SH_C0 + 0.5, 0, 1),
        )


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ])


@dataclass
class View:
    name: str
    R: np.ndarray  # world -> camera
    t: np.ndarray
    K: np.ndarray
    width: int
    height: int

    @property
    def centre(self):
        return -self.R.T @ self.t

    def project(self, pts):
        """Returns (uv (N, 2), depth (N,)) for world points."""
        cam = pts @ self.R.T + self.t
        z = cam[:, 2]
        uv = (cam[:, :2] / np.maximum(z[:, None], 1e-6)) * [self.K[0, 0], self.K[1, 1]] + self.K[:2, 2]
        return uv, z


def orbit_frame(views):
    """(orbit centre, unit up) from camera centres and the phones' up vectors."""
    centres = np.array([v.centre for v in views.values()])
    up = np.mean([v.R.T @ np.array([0, -1.0, 0]) for v in views.values()], axis=0)
    return centres.mean(0), up / np.linalg.norm(up)


def in_statue_cylinder(pts, views, radius=1.3, h_min=-1.0, h_max=1.8):
    """Points inside the vertical cylinder around the orbit axis (SfM units,
    heights relative to mean camera height). Same region recon/make_masks.py uses."""
    c, up = orbit_frame(views)
    d = pts - c
    h = d @ up
    r = np.linalg.norm(d - np.outer(h, up), axis=1)
    return (r < radius) & (h > h_min) & (h < h_max)


def load_views(sparse_dir):
    """PINHOLE COLMAP model -> {image name: View}."""
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["colmap", "model_converter", "--input_path", str(sparse_dir),
                        "--output_path", tmp, "--output_type", "TXT"],
                       check=True, capture_output=True)
        read = lambda f: [l for l in open(Path(tmp) / f) if not l.startswith("#")]
        cam, images = read("cameras.txt")[0].split(), read("images.txt")
    assert cam[1] == "PINHOLE", cam[1]
    w, h = int(cam[2]), int(cam[3])
    fx, fy, cx, cy = map(float, cam[4:8])
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    views = {}
    for header in images[::2]:
        p = header.split()
        views[p[9]] = View(p[9], qvec2rotmat(np.array(p[1:5], float)), np.array(p[5:8], float), K, w, h)
    return views
