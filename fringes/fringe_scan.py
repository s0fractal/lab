"""Fringe scan of the committed phs-soliton manifests (post hoc, exploratory): for every task-A run that learned
(Acc(0) >= 0.9), locate the first accuracy minimum sigma_1 and the first revival peak sigma_2 after it, and the
ratio sigma_2 / sigma_1. A phase that grows as sigma^2 puts the minimum at phase pi and the revival at 2 pi, so the
ratio would be sqrt(2), whatever the model's own precession rate."""
import glob
import json
import math
import sys

CHANCE = 1 / 7


def curve(m):
    c = m["cells"]
    s = sorted(c, key=float)
    return [float(x) for x in s], [c[x]["acc"] for x in s]


def fringes(sig, acc):
    """Successive extrema below/above chance after accuracy first drops under chance."""
    i0 = next((i for i, a in enumerate(acc) if a < CHANCE), None)
    if i0 is None:
        return None
    ext, want_min, i = [], True, i0
    while i < len(acc):
        j = i
        if want_min:
            while j + 1 < len(acc) and acc[j + 1] <= acc[j]:
                j += 1
        else:
            while j + 1 < len(acc) and acc[j + 1] >= acc[j]:
                j += 1
        if j + 1 >= len(acc) and not want_min:
            break   # a peak at the grid edge is censored, not a revival
        ext.append((sig[j], acc[j]))
        want_min, i = not want_min, j + 1
    return ext


root = sys.argv[1]
rows = []
for kind in ("soliton", "pure", "gru", "gru_pm"):
    for f in sorted(glob.glob(f"{root}/results/manifests/A-{kind}-T*-s*.json")):
        m = json.load(open(f))
        sig, acc = curve(m)
        if acc[0] < 0.9:
            continue
        ext = fringes(sig, acc) or []
        mins = [e for k, e in enumerate(ext) if k % 2 == 0]
        peaks = [e for k, e in enumerate(ext) if k % 2 == 1 and e[1] > CHANCE + 0.03]
        below = min(acc)
        rows.append((kind, m["T"], m["seed"], below, mins, peaks))

for kind, T, seed, below, mins, peaks in rows:
    s = f"{kind:8s} T={T:3d} s{seed}  min Acc {below:.3f}"
    if mins:
        s += "  extrema: " + "  ".join(f"{'v' if k % 2 == 0 else '^'}{x:.2f}({a:.2f})"
                                     for k, (x, a) in enumerate(sorted(mins + peaks)))
    if mins and peaks:
        s1, s2 = mins[0][0], peaks[0][0]
        s += f"   s2/s1 = {s2 / s1:.3f}"
        if len(mins) > 1:
            s += f"   s3/s1 = {mins[1][0] / s1:.3f}"
        if len(peaks) > 1:
            s += f"   s4/s1 = {peaks[1][0] / s1:.3f}"
    print(s)

r = [p[0][0] / mi[0][0] for k, T, s, b, mi, p in rows if k == "soliton" and mi and p]
if r:
    mu = sum(r) / len(r)
    sd = math.sqrt(sum((x - mu) ** 2 for x in r) / (len(r) - 1)) if len(r) > 1 else 0
    print(f"\nsoliton s2/s1 over {len(r)} runs: mean {mu:.3f} sd {sd:.3f}  (sqrt 2 = {math.sqrt(2):.3f}, "
          f"linear-phase would give 2.000)")
