"""Honest decoders for the clock memory (post hoc play; the prediction is in README before the runs).
States z_T of one frozen checkpoint on a calibration set and a separate test set, 2048 sequences per sigma each.
Every decoder but the oracle learns on the calibration set only:

  fixed    the model's own readout
  pooled   one LDA on the whole calibration set: blind to sigma, linear
  sensor   16 quantile bins of the label-free sensor D = <cos 2 theta> (read from the state itself), one LDA per bin;
           a test state takes the LDA of its bin: blind to sigma and to the test labels
  sigma    LDA on the calibration cell of the same sigma: knows sigma, not the test labels
  oracle   LDA on one half of the test cell (the old probe): a diagnostic bound, not a decoder
All five are scored on the same other half of the test cell.
Also per sigma: the mean of D and of the kinetic sensor K = <P^2> on the test set, so their use as thermometers shows.
Usage: readout.py LAB CKPT OUT [n] [threads]"""
import json
import sys

import numpy as np
import torch

LAB, ckpt, out = sys.argv[1], sys.argv[2], sys.argv[3]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
torch.set_num_threads(int(sys.argv[5]) if len(sys.argv) > 5 else 1)
sys.path.insert(0, f"{LAB}/phs-soliton")
from phs import tasks  # noqa: E402
from phs.models import build  # noqa: E402
from phs.train import to_torch  # noqa: E402

c = torch.load(ckpt)
spec = tasks.TASKS["A"]
model = build(c["kind"], spec["vocab"], spec["classes"], c["cfg"], torch.Generator().manual_seed(0))
model.load_state_dict(c["state"])
model.eval()
T, N, d, mp, K = c["T"], model.N, model.d, model.core.m_pin, spec["classes"]
grab = {}
model.readout.register_forward_hook(lambda mod, inp, o: grab.__setitem__("z", inp[0].detach()))
SIG = [round(0.05 * i, 2) for i in range(25)]


@torch.no_grad()
def states(sigma, namespace):
    b = tasks.test_set("A", T, sigma, n=n, namespace=namespace)
    x, tg, sg = to_torch(b)
    seed = tasks.derive_seed(f"{namespace}-noise", T, f"{sigma:.2f}")
    logits = model(x, sg, generator=torch.Generator().manual_seed(seed))
    z = grab["z"].double().numpy()
    zz = z.reshape(len(z), N, 3 * d)                                   # cos | sin | P per node (phs/models.py)
    theta = np.arctan2(zz[..., d:2 * d], zz[..., :d])
    D = np.cos(mp * theta).mean((1, 2))
    Kin = (zz[..., 2 * d:] ** 2).mean((1, 2))
    return z, logits[:, -1].double().numpy().argmax(1), tg[:, -1].numpy(), D, Kin


def lda_fit(z, y, lam=1e-3):
    mu = np.stack([z[y == k].mean(0) for k in range(K)])
    r = z - mu[y]
    S = r.T @ r / len(z)
    S = S + lam * S.diagonal().mean() * np.eye(S.shape[0])
    W = np.linalg.solve(S, mu.T)
    return W, -0.5 * (mu * W.T).sum(1)


def lda_predict(model_, z):
    W, b = model_
    return (z @ W + b).argmax(1)


cal = {s: states(s, "calib-readout") for s in SIG}
test = {s: states(s, "play") for s in SIG}
Zc = np.concatenate([cal[s][0] for s in SIG])
Yc = np.concatenate([cal[s][2] for s in SIG])
Dc = np.concatenate([cal[s][3] for s in SIG])
pooled = lda_fit(Zc, Yc)
edges = np.quantile(Dc, np.linspace(0, 1, 17)[1:-1])                 # 16 bins of the sensor, from calibration only
bins = np.digitize(Dc, edges)
per_bin = [lda_fit(Zc[bins == k], Yc[bins == k]) for k in range(16)]
rows = []
for s in SIG:
    z, fixed_pred, y, D, Kin = test[s]
    tb = np.digitize(D, edges)
    sensor_pred = np.empty(len(z), dtype=int)
    for k in range(16):
        if (tb == k).any():
            sensor_pred[tb == k] = lda_predict(per_bin[k], z[tb == k])
    sig_model = lda_fit(cal[s][0], cal[s][2])
    half = len(y) // 2
    oracle = lda_fit(z[:half], y[:half])
    ev = slice(half, None)                                             # every decoder is scored on the same half
    acc = lambda pred: float((pred[ev] == y[ev]).mean())  # noqa: E731
    row = {"sigma": s, "fixed": acc(fixed_pred), "pooled": acc(lda_predict(pooled, z)), "sensor": acc(sensor_pred),
           "sigma_lda": acc(lda_predict(sig_model, z)), "oracle": float((lda_predict(oracle, z[ev]) == y[ev]).mean()),
           "D": float(D.mean()), "K": float(Kin.mean()), "D_cal": float(cal[s][3].mean())}
    rows.append(row)
    print(json.dumps(row), flush=True)
json.dump({"ckpt": ckpt, "kind": c["kind"], "seed": c["seed"], "n": n, "bins": 16, "edges": edges.tolist(), "rows": rows},
          open(out, "w"), indent=1)
