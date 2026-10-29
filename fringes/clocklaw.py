"""The clock law on one frozen model, without any readout (post hoc play; the prediction is in README before the run).
For hold lengths T = 24, 32, 48 and sigma 0, 0.05, ..., 1.2 (1024 sequences per cell, namespace "clocklaw"), the class
signal c(sigma, T) of the full state z_T (class means minus their mean) and its overlap with the sigma = 0 signal at
the same T: r(sigma, T) = <c(sigma, T), c(0, T)> / (|c(sigma, T)| |c(0, T)|). Its first zero in sigma, sigma0(T)
(linear between grid points), is the temperature part of the memory phase, free of the part the hold time adds at
sigma = 0. Also |c(sigma, T)| / |c(0, T)| and, across T, the overlap of the sigma = 0 signals with the one at T = 32.
Usage: clocklaw.py LAB CKPT OUT [threads] [T list]   (T list, e.g. 48, replays a subset; artifacts/verify.py does)"""
import json
import sys

import numpy as np
import torch

LAB, ckpt, out = sys.argv[1], sys.argv[2], sys.argv[3]
torch.set_num_threads(int(sys.argv[4]) if len(sys.argv) > 4 else 1)
TS = [int(v) for v in sys.argv[5].split(",")] if len(sys.argv) > 5 else [24, 32, 48]
sys.path.insert(0, f"{LAB}/phs-soliton")
from phs import tasks  # noqa: E402
from phs.models import build  # noqa: E402
from phs.train import to_torch  # noqa: E402

c = torch.load(ckpt)
spec = tasks.TASKS["A"]
model = build(c["kind"], spec["vocab"], spec["classes"], c["cfg"], torch.Generator().manual_seed(0))
model.load_state_dict(c["state"])
model.eval()
K = spec["classes"]
grab = {}
model.readout.register_forward_hook(lambda mod, inp, o: grab.__setitem__("z", inp[0].detach()))
GRID = [round(0.05 * i, 2) for i in range(25)]


@torch.no_grad()
def signal(sigma, T, n=1024):
    b = tasks.test_set("A", T, sigma, n=n, namespace="clocklaw")
    x, tg, sg = to_torch(b)
    model(x, sg, generator=torch.Generator().manual_seed(tasks.derive_seed("clocklaw-noise", T, f"{sigma:.2f}")))
    z, y = grab["z"].double().numpy(), tg[:, -1].numpy()
    m = np.stack([z[y == k].mean(0) for k in range(K)])
    return (m - m.mean(0, keepdims=True)).ravel()


def first_zero(sig, r):
    for i in range(1, len(r)):
        if r[i] < 0 <= r[i - 1]:
            return sig[i - 1] + (sig[i] - sig[i - 1]) * r[i - 1] / (r[i - 1] - r[i])
    return None


res = {"ckpt": ckpt, "kind": c["kind"], "seed": c["seed"], "n": 1024, "sigma": GRID}
c0 = {}
for T in TS:
    cs = [signal(s, T) for s in GRID]
    c0[T] = cs[0]
    r = [float(v @ cs[0] / (np.linalg.norm(v) * np.linalg.norm(cs[0]))) for v in cs]
    rel = [float(np.linalg.norm(v) / np.linalg.norm(cs[0])) for v in cs]
    res[f"T{T}"] = {"r": r, "norm": rel, "sigma0": first_zero(GRID, r)}
    print(T, "sigma0", res[f"T{T}"]["sigma0"], " ".join(f"{s:.2f}:{a:+.2f}" for s, a in zip(GRID[::2], r[::2])), flush=True)
if 32 in TS:
    res["cross_T"] = {f"T{T}": float(c0[T] @ c0[32] / (np.linalg.norm(c0[T]) * np.linalg.norm(c0[32])))
                      for T in TS if T != 32}
    print("sigma = 0 signal overlap with T = 32:", res["cross_T"], flush=True)
json.dump(res, open(out, "w"), indent=1)
