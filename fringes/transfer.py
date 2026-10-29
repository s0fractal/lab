"""Do the honest decoders transfer? (post hoc play; the prediction is in README before the runs)
The decoders of readout.py are fit once on its calibration set (T = 32, sigma 0, 0.05, ..., 1.2, 2048 per sigma) and
then frozen. They are scored on data none of them saw, from a new namespace ("transfer", 1024 sequences per cell):

  fresh      the same sigma grid points 0.5-0.9 at T = 32: a new corpus and new noise
  between    sigma halfway between grid points (0.525-0.925): the sigma-aware decoder takes the nearest grid cell
  beyond     decoders fit only on sigma <= 0.6, scored at 0.7, 0.8, 0.9: outside their calibration range
  T24, T48   the whole grid at hold length T = 24 and T = 48 with the T = 32 decoders: the clock has run for a
             different time; the model's own readout shows where its anti-memory dip moves
Decoders: fixed (the model's readout), pooled (one LDA), sensor (16 bins of D = <cos 2 theta>), sigma (LDA of the
nearest calibration cell; knows sigma, not T), oracle (LDA on one half of the test cell, a bound); all scored on the
other half. The calibration states are readout.py's own, so the decoders are the frozen ones: the run checks its
sensor-bin edges and calibration sensor means against artifacts/readout-2026-10-28 and stops if they differ.
Usage: transfer.py LAB CKPT OUT [threads] [--cells T:sigma,...]
--cells scores only the model's readout and the oracle on the listed test cells (no calibration, a few seconds each),
which is what artifacts/verify.py replays."""
import json
import sys

import numpy as np
import torch

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
    # the calibration set must be readout.py's, state for state, so its noise seed is formatted as there ("0.70");
    # the new test cells need three decimals for the halfway sigmas. An earlier version used three decimals for the
    # calibration too, which drew other calibration states and so refit the decoder (Codex review of #8, 1dec5d4)
    seed = tasks.derive_seed(f"{namespace}-noise", T, f"{sigma:.2f}" if namespace == "calib-readout" else f"{sigma:.3f}")
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
    return (z @ m[0] + m[1]).argmax(1)


def fit(cells):
    """The frozen decoders from calibration cells {sigma: (z, pred, y, D)}."""
    Z = np.concatenate([cells[s][0] for s in cells])
    Y = np.concatenate([cells[s][2] for s in cells])
    D = np.concatenate([cells[s][3] for s in cells])
    edges = np.quantile(D, np.linspace(0, 1, 17)[1:-1])
    b = np.digitize(D, edges)
    return {"pooled": lda_fit(Z, Y), "edges": edges, "bins": [lda_fit(Z[b == k], Y[b == k]) for k in range(16)],
            "cells": {s: lda_fit(cells[s][0], cells[s][2]) for s in cells}}


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
cal = {s: states(s, "calib-readout", T0, 2048) for s in GRID}              # the calibration set of readout.py
full = fit(cal)
low = fit({s: cal[s] for s in GRID if s <= 0.6})
# the decoder must be the frozen one: its sensor-bin edges and the mean sensor of every calibration cell are compared
# with the readout snapshot, and the run stops if they differ
ref = json.load(open(f"{LAB}/fringes/artifacts/readout-2026-10-28/results/readout/{c['kind']}-s{c['seed']}.json"))
d_edges = float(np.max(np.abs(full["edges"] - np.asarray(ref["edges"]))))
d_dcal = max(abs(float(cal[r["sigma"]][3].mean()) - r["D_cal"]) for r in ref["rows"])
print(f"calibration matches readout.py: edges off by {d_edges:.1e}, mean calibration sensor off by {d_dcal:.1e}",
      flush=True)
assert d_edges <= 1e-6 and d_dcal <= 1e-6, "transfer.py would fit another decoder than the frozen readout.py one"
res = {"ckpt": ckpt, "kind": c["kind"], "seed": c["seed"], "T0": T0, "n_cal": 2048, "n_test": 1024,
       "edges": full["edges"].tolist(), "calibration_check": {"edges": d_edges, "D_cal": d_dcal}}
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
