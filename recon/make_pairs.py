"""Build a COLMAP match-pair list for multi-ring orbit captures.

Each video is one full orbit around the statue, all starting near the front
and walking the same direction. So a frame's normalized time (phase in [0, 1))
is a good proxy for azimuth:
  - within a ring: match the +-`--seq` neighbouring frames, wrapping around so
    the end of the orbit closes the loop with the start;
  - across rings: match frames whose phase differs by less than `--cross`.

Usage:
    python recon/make_pairs.py data/alma_v2/frames.csv data/alma_v2/pairs.txt
"""

import argparse
import csv
from collections import defaultdict
from itertools import combinations


def circ_dist(a, b):
    d = abs(a - b) % 1.0
    return min(d, 1.0 - d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames_csv")
    ap.add_argument("out")
    ap.add_argument("--seq", type=int, default=8)
    ap.add_argument("--cross", type=float, default=0.06)
    args = ap.parse_args()

    rings = defaultdict(list)
    with open(args.frames_csv) as f:
        for row in csv.DictReader(f):
            rings[int(row["ring"])].append((float(row["time_s"]), row["image"]))

    phased = {}
    pairs = set()
    for r, items in rings.items():
        items.sort()
        n = len(items)
        phased[r] = [(i / n, name) for i, (_, name) in enumerate(items)]
        for i in range(n):
            for k in range(1, args.seq + 1):
                a, b = items[i][1], items[(i + k) % n][1]
                pairs.add(tuple(sorted((a, b))))

    n_within = len(pairs)
    for ra, rb in combinations(sorted(phased), 2):
        for pa, a in phased[ra]:
            for pb, b in phased[rb]:
                if circ_dist(pa, pb) < args.cross:
                    pairs.add(tuple(sorted((a, b))))

    with open(args.out, "w") as f:
        for a, b in sorted(pairs):
            f.write(f"{a} {b}\n")
    print(f"{len(pairs)} pairs ({n_within} within-ring, {len(pairs) - n_within} cross-ring)")


if __name__ == "__main__":
    main()
