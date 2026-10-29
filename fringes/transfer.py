"""Do the honest decoders transfer? (post hoc play; the prediction is in README before the runs)
The decoders are readout.py's own, fit once on its calibration set (T = 32, sigma 0, 0.05, ..., 1.2, 2048 per sigma) and
frozen in artifacts/decoder-2026-10-29 by the same readout.py run that wrote artifacts/readout-2026-10-28. Nothing is
refit and no calibration state is drawn here, so a CPU whose last bits differ scores the same decoder (Codex review of
#8, 3ad06e9: refitting on an Apple M4 Pro moved the sensor-bin edges by 3.3e-5 and stopped the run). They are scored
on data none of them saw, from a new namespace ("transfer", 1024 sequences per cell):

  fresh      the same sigma grid points 0.5-0.9 at T = 32: a new corpus and new noise
  between    sigma halfway between grid points (0.525-0.925): the sigma-aware decoder takes the nearest grid cell
  beyond     decoders fit only on sigma <= 0.6, scored at 0.7, 0.8, 0.9: outside their calibration range
  T24, T48   the whole grid at hold length T = 24 and T = 48 with the T = 32 decoders: the clock has run for a
             different time; the model's own readout shows where its anti-memory dip moves
Decoders: fixed (the model's readout), pooled (one LDA), sensor (16 bins of D = <cos 2 theta>), sigma (LDA of the
nearest calibration cell; knows sigma, not T), oracle (LDA on one half of the test cell, a bound); all scored on the
other half. The decoder file must be the one in the manifest of artifacts/decoder-2026-10-29 (SHA-256), and its
sensor-bin edges and mean calibration sensors must equal those of artifacts/readout-2026-10-28 (both written by one
run, so exactly); the run stops otherwise.
Usage: transfer.py LAB CKPT OUT [threads] [--decoder DEC.npz] [--cells T:sigma,... | --readout-cells sigma,...]
--decoder names another copy of the decoder file (it must still be the frozen one).
--cells scores only the model's readout and the oracle on the listed test cells, which is what artifacts/verify.py
replays. --readout-cells scores the frozen decoders on readout.py's own test cells (namespace "play", T = 32, 2048
per sigma): they must give readout.py's numbers, bit for bit on the machine that made them."""
import hashlib
import json
import sys

import numpy as np
import torch

dec_path = None
if "--decoder" in sys.argv:
    i = sys.argv.index("--decoder")
    dec_path = sys.argv[i + 1]
    del sys.argv[i:i + 2]
LAB, ckpt, out = sys.argv[1], sys.argv[2], sys.argv[3]
torch.set_num_threads(int(sys.argv[4]) if len(sys.argv) > 4 else 1)
sys.path.insert(0, f"{LAB}/phs-soliton")
from phs import tasks  # noqa: E402
from phs.models import build  # noqa: E402
from phs.train import to_torch  # noqa: E402

c = torch.load(ckpt)
spec = tasks.TASKS["A"]
model = build(c["kind"], spec["vocab"], spec["classes"], c["cfg"], torch.Generator().manual_seed(0))
model.load_state_dict(c["state"])
model.eval()
T0, N, d, mp, K = c["T"], model.N, model.d, model.core.m_pin, spec["classes"]
grab = {}
model.readout.register_forward_hook(lambda mod, inp, o: grab.__setitem__("z", inp[0].detach()))
GRID = [round(0.05 * i, 2) for i in range(25)]


@torch.no_grad()
def states(sigma, namespace, T, n):
    b = tasks.test_set("A", T, sigma, n=n, namespace=namespace)
    x, tg, sg = to_torch(b)
    # readout.py's test cells ("play") are seeded with two decimals ("0.70"), as there; the new test cells need three
    # for the halfway sigmas. An earlier version drew the calibration here with three decimals, which gave other
    # calibration states and refit the decoder (Codex review of #8, 1dec5d4); the calibration is no longer drawn here
    seed = tasks.derive_seed(f"{namespace}-noise", T, f"{sigma:.2f}" if namespace == "play" else f"{sigma:.3f}")
    logits = model(x, sg, generator=torch.Generator().manual_seed(seed))
    z = grab["z"].double().numpy()
    zz = z.reshape(len(z), N, 3 * d)                                   # cos | sin | P per node (phs/models.py)
    D = np.cos(mp * np.arctan2(zz[..., d:2 * d], zz[..., :d])).mean((1, 2))
    return z, logits[:, -1].double().numpy().argmax(1), tg[:, -1].numpy(), D


def lda_fit(z, y, lam=1e-3):
    mu = np.stack([z[y == k].mean(0) for k in range(K)])
    r = z - mu[y]
    S = r.T @ r / len(z)
    S = S + lam * S.diagonal().mean() * np.eye(S.shape[0])
    W = np.linalg.solve(S, mu.T)
    return W, -0.5 * (mu * W.T).sum(1)


def lda_predict(m, z):
    # a class with no state among the cells an LDA was fit on has a NaN mean, so NaN scores; argmax would return the
    # first NaN and give every state that class. It is never predicted instead. Of the frozen decoders only bin 15 of
    # Soliton seed 901's sigma <= 0.6 sensor decoder lacks a class (class 0), and no transfer test state falls in it
    s = z @ m[0] + m[1]
    return np.where(np.isnan(s), -np.inf, s).argmax(1)


def frozen_decoders():
    """(full, low, check) from the frozen decoder file of this model; stops unless it is the frozen one."""
    rel = f"decoders/{c['kind']}-s{c['seed']}.npz"
    path = dec_path or f"{LAB}/fringes/artifacts/decoder-2026-10-29/{rel}"
    man = json.load(open(f"{LAB}/fringes/artifacts/decoder-2026-10-29/MANIFEST.json"))
    want = next((f["sha256"] for f in man["files"] if f["path"] == rel), None)
    got = hashlib.sha256(open(path, "rb").read()).hexdigest()
    a = np.load(path, allow_pickle=False)
    ref = json.load(open(f"{LAB}/fringes/artifacts/readout-2026-10-28/results/readout/{c['kind']}-s{c['seed']}.json"))
    grid_ok = a["sigmas"].tolist() == GRID == [r["sigma"] for r in ref["rows"]] and a["edges"].shape == (15,)
    d_edges = float(np.max(np.abs(a["edges"] - np.asarray(ref["edges"])))) if grid_ok else float("inf")
    d_dcal = max(abs(float(x) - r["D_cal"]) for x, r in zip(a["D_cal"], ref["rows"])) if grid_ok else float("inf")
    print(f"decoder {path}: SHA-256 {'is' if got == want else 'is NOT'} the frozen one, sensor-bin edges off by "
          f"{d_edges:.1e}, mean calibration sensor off by {d_dcal:.1e} (from artifacts/readout-2026-10-28)", flush=True)
    if got != want or d_edges != 0 or d_dcal != 0:
        sys.exit("not the frozen readout.py decoder: nothing is refit, so there is nothing to score")

    def m(name):
        return a[f"{name}_W"], a[f"{name}_b"]
    cells = {s: m(f"cell{i:02d}") for i, s in enumerate(GRID)}
    full = {"pooled": m("pooled"), "edges": a["edges"], "bins": [m(f"bin{k:02d}") for k in range(16)], "cells": cells}
    low = {"pooled": m("low_pooled"), "edges": a["low_edges"], "bins": [m(f"low_bin{k:02d}") for k in range(16)],
           "cells": {s: cells[s] for s in a["low_sigmas"].tolist()}}
    return full, low, {"edges": d_edges, "D_cal": d_dcal}


def score(dec, cell, sigma):
    z, fixed_pred, y, D = cell
    tb = np.digitize(D, dec["edges"])
    sensor_pred = np.empty(len(z), dtype=int)
    for k in range(16):
        if (tb == k).any():
            sensor_pred[tb == k] = lda_predict(dec["bins"][k], z[tb == k])
    near = min(dec["cells"], key=lambda s: abs(s - sigma))
    half = len(y) // 2
    ev = slice(half, None)
    acc = lambda p: float((p[ev] == y[ev]).mean())  # noqa: E731
    return {"fixed": acc(fixed_pred), "pooled": acc(lda_predict(dec["pooled"], z)), "sensor": acc(sensor_pred),
            "sigma_lda": acc(lda_predict(dec["cells"][near], z)), "sigma_cell": near,
            "oracle": float((lda_predict(lda_fit(z[:half], y[:half]), z[ev]) == y[ev]).mean()), "D": float(D.mean())}


if "--cells" in sys.argv:
    cells = []
    for spec_ in sys.argv[sys.argv.index("--cells") + 1].split(","):
        T, s = spec_.split(":")
        z, fixed_pred, y, D = states(float(s), "transfer", int(T), 1024)
        half = len(y) // 2
        cells.append({"T": int(T), "sigma": float(s), "fixed": float((fixed_pred[half:] == y[half:]).mean()),
                      "oracle": float((lda_predict(lda_fit(z[:half], y[:half]), z[half:]) == y[half:]).mean())})
        print(json.dumps(cells[-1]), flush=True)
    json.dump({"ckpt": ckpt, "cells": cells}, open(out, "w"), indent=1)
    sys.exit(0)
full, low, check = frozen_decoders()
if "--readout-cells" in sys.argv:
    rows = [{"sigma": s, **score(full, states(s, "play", T0, 2048), s)}
            for s in map(float, sys.argv[sys.argv.index("--readout-cells") + 1].split(","))]
    for r in rows:
        print(json.dumps(r), flush=True)
    json.dump({"ckpt": ckpt, "rows": rows}, open(out, "w"), indent=1)
    sys.exit(0)
res = {"ckpt": ckpt, "kind": c["kind"], "seed": c["seed"], "T0": T0, "n_cal": 2048, "n_test": 1024,
       "edges": full["edges"].tolist(), "calibration_check": check}
fresh = {s: states(s, "transfer", T0, 1024) for s in (0.5, 0.6, 0.7, 0.8, 0.9)}
res["fresh"] = [{"sigma": s, **score(full, fresh[s], s)} for s in fresh]
res["between"] = [{"sigma": s, **score(full, states(s, "transfer", T0, 1024), s)}
                  for s in (0.525, 0.625, 0.725, 0.825, 0.925)]
res["beyond"] = [{"sigma": s, **score(low, fresh[s], s), "full_sensor": score(full, fresh[s], s)["sensor"]}
                 for s in (0.7, 0.8, 0.9)]
for T in (24, 48):
    res[f"T{T}"] = [{"sigma": s, **score(full, states(s, "transfer", T, 1024), s)} for s in GRID]
for part in ("fresh", "between", "beyond", "T24", "T48"):
    for r in res[part]:
        print(part, json.dumps(r), flush=True)
json.dump(res, open(out, "w"), indent=1)
