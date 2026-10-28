"""Data collapse of the committed Soliton-PHS task-A curves: x = (sigma / sigma_1)^2 with sigma_1 the first
accuracy minimum (parabolic refinement on the 0.05 grid). A phase that grows as sigma^2 puts every run's
extrema at x = 1 (pi), 2 (2 pi), 3 (3 pi), ..."""
import json
import sys

d = json.load(open(sys.argv[1]))
sig = d["sigma"]


def sigma1(acc):
    i0 = next(i for i, a in enumerate(acc) if a < 1 / 7)
    j = i0
    while j + 1 < len(acc) and acc[j + 1] <= acc[j]:
        j += 1
    # first index of the flat bottom .. last index: take the centre of the plateau within 0.01 of the minimum
    lo = j
    while lo - 1 >= 0 and acc[lo - 1] <= acc[j] + 0.01:
        lo -= 1
    hi = j
    while hi + 1 < len(acc) and acc[hi + 1] <= acc[j] + 0.01:
        hi += 1
    return 0.5 * (sig[lo] + sig[hi])


out = {}
for T in (16, 32, 64, 128):
    for run in d[f"soliton-{T}"]:
        s1 = sigma1(run["acc"])
        x = [(s / s1) ** 2 for s in sig]
        out[f"{T}-s{run['seed']}"] = {"s1": s1, "x": x, "acc": run["acc"]}
        # where are the revival peaks in x?
        peaks = [x[i] for i in range(1, len(x) - 1)
                 if run["acc"][i] >= run["acc"][i - 1] and run["acc"][i] > run["acc"][i + 1] and run["acc"][i] > 0.2
                 and x[i] > 1]
        print(f"T={T:3d} s{run['seed']}  sigma_1 {s1:.3f}  revival at x = {', '.join(f'{p:.2f}' for p in peaks) or '-'}")
json.dump(out, open(sys.argv[2], "w"))
