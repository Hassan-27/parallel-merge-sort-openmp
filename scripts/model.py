"""
model.py - Analytical performance model for task-parallel 2-way / 4-way merge sort.

The model PREDICTS the run time of a full sort using only micro-benchmark
calibration data (results/calib.csv, produced by `pdcsort calibrate`).  It never
looks at the measured sort times, so comparing its predictions with the
benchmark is an honest test of accuracy.

Structure of the recursion (k-way, k in {2,4}, task cutoff C):

    level 0            1 merge of size n                       <- runs on 1 thread
    level 1            k merges of size n/k                    <- up to k threads
    ...
    level d-1          k^(d-1) merges of size n/k^(d-1)
    leaves (level d)   k^d sequential sub-sorts of size n/k^d <= C

d = smallest integer with n / k^d <= C.

Time of a phase with M equal, independent jobs of size s on p threads:

    q        = min(p, M)                    (threads that can be busy)
    rate     = r(k, q, s)                   (calibrated elements/s PER THREAD with
                                             q threads running concurrently -> captures
                                             cache size effects, DRAM bandwidth sharing
                                             and Hyper-Threading contention)
    T_phase  = ceil(M / q) * s / rate       (level-synchronous "rounds" model)

    T(n,p)   = T_leaves + sum_levels T_level + N_tasks * t_task(p) / p

The sequential versions are modelled the same way with p = 1 and a "virtual"
cutoff equal to the largest calibrated leaf size (no task overhead).

Work / span view (used in the report):
    2-way span  = n + n/2 + n/4 + ... ~= 2n   merge steps   (top merge is serial)
    4-way span  = n + n/4 + n/16 + ... ~= 4n/3 merge steps
so the 4-way tree has ~33% shorter critical path and half as many passes over
memory, at the price of a more expensive (2 comparisons / element) merge.
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class Calibration:
    def __init__(self, path: Path = ROOT / "results" / "calib.csv"):
        self.merge = defaultdict(list)   # (way, q) -> [(size, rate)]
        self.leaf = defaultdict(list)
        self.task = {}
        self.bw = {}
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                kind, way, q = row["kind"], int(row["way"]), int(row["q"])
                size, val = int(row["size"]), float(row["value"])
                if kind == "merge":
                    self.merge[(way, q)].append((size, val))
                elif kind == "leaf":
                    self.leaf[(way, q)].append((size, val))
                elif kind == "task":
                    self.task[q] = val
                elif kind == "bw":
                    self.bw[q] = val
        for d in (self.merge, self.leaf):
            for k in d:
                d[k].sort()
        self.max_q = max(self.task) if self.task else 1

    @staticmethod
    def _interp(points, s):
        """Log-linear interpolation of rate vs log2(size); clamped at the ends."""
        if s <= points[0][0]:
            return points[0][1]
        if s >= points[-1][0]:
            return points[-1][1]
        for (s0, r0), (s1, r1) in zip(points, points[1:]):
            if s0 <= s <= s1:
                t = (math.log2(s) - math.log2(s0)) / (math.log2(s1) - math.log2(s0))
                return r0 + t * (r1 - r0)
        return points[-1][1]

    def merge_rate(self, way, q, s):
        return self._interp(self.merge[(way, min(q, self.max_q))], s)

    def leaf_rate(self, way, q, s):
        return self._interp(self.leaf[(way, min(q, self.max_q))], s)

    def leaf_max(self, way):
        return self.leaf[(way, 1)][-1][0]


def predict(cal: Calibration, way: int, n: int, p: int, cutoff: int,
            sequential: bool = False, breakdown: bool = False):
    """Predicted wall time (s) of a k-way merge sort of n ints on p threads."""
    if sequential:
        p, cutoff = 1, cal.leaf_max(way)
    d = 0
    while n / way ** d > cutoff:
        d += 1
    leaves = way ** d
    leaf_size = n / leaves

    parts = {}
    q = min(p, leaves)
    parts["leaves"] = math.ceil(leaves / q) * leaf_size / cal.leaf_rate(way, q, leaf_size)

    t_merge = 0.0
    levels = []
    for k in range(d - 1, -1, -1):
        m = way ** k                     # independent merges at this level
        s = n / m                        # output size of each merge
        q = min(p, m)
        t = math.ceil(m / q) * s / cal.merge_rate(way, q, s)
        levels.append((k, m, s, q, t))
        t_merge += t
    parts["merges"] = t_merge

    if sequential:
        parts["tasks"] = 0.0
    else:
        n_tasks = (way ** (d + 1) - 1) // (way - 1)
        parts["tasks"] = n_tasks * cal.task.get(min(p, cal.max_q), 0.0) / p

    total = sum(parts.values())
    if breakdown:
        return total, parts, levels
    return total


def span_bound(cal: Calibration, way: int, n: int, cutoff: int):
    """T_infinity: time with unlimited threads = critical path (one chain of merges
    from a leaf to the root, every merge executed by a single thread)."""
    d = 0
    while n / way ** d > cutoff:
        d += 1
    leaf_size = n / way ** d
    t = leaf_size / cal.leaf_rate(way, 1, leaf_size)
    for k in range(d):
        s = n / way ** k
        t += s / cal.merge_rate(way, 1, s)
    return t


if __name__ == "__main__":
    cal = Calibration()
    for way in (2, 4):
        for n in (10**7, 10**8, 2.5 * 10**8):
            n = int(n)
            ts = predict(cal, way, n, 1, 0, sequential=True)
            row = [f"{way}-way n={n:>11,}  seq {ts:7.3f}s"]
            for p in (1, 2, 4, 8):
                row.append(f"p{p} {predict(cal, way, n, p, 16384):7.3f}s")
            row.append(f"T_inf {span_bound(cal, way, n, 16384):6.3f}s")
            print("  ".join(row))
