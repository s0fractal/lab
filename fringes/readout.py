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
Usage: readout.py LAB CKPT OUT [n] [threads] [--decoder DEC.npz]
--decoder also writes the fitted decoders, which transfer.py scores on unseen data instead of refitting them: pooled,
the sensor-bin edges with one LDA per bin, one LDA per calibration sigma, and the pooled and sensor decoders fit on
sigma <= 0.6 only (transfer.py's out-of-range test), with the mean calibration sensor per sigma. The JSON is the same
with or without it. The file is a plain npz with a fixed timestamp, so the same decoder gives the same bytes."""
import io
import json
import sys
import zipfile

import numpy as np
import torch

dec_out = None
if "--decoder" in sys.argv:
    i = sys.argv.index("--decoder")
    dec_out = sys.argv[i + 1]
    del sys.argv[i:i + 2]
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
    # a class absent from an LDA's training states has NaN scores and is never predicted (argmax would pick the first
    # NaN for every state); on the six frozen models none of the decoders this script scores lacks a class
    W, b = model_
    s = z @ W + b
    return np.where(np.isnan(s), -np.inf, s).argmax(1)


def fit_sensor(sigmas):
    """Pooled LDA, sensor-bin edges and one LDA per bin, from the calibration cells of these sigmas."""
    Zc = np.concatenate([cal[s][0] for s in sigmas])
    Yc = np.concatenate([cal[s][2] for s in sigmas])
    Dc = np.concatenate([cal[s][3] for s in sigmas])
    edges_ = np.quantile(Dc, np.linspace(0, 1, 17)[1:-1])            # 16 bins of the sensor, from calibration only
    bins = np.digitize(Dc, edges_)
    return lda_fit(Zc, Yc), edges_, [lda_fit(Zc[bins == k], Yc[bins == k]) for k in range(16)]


def save_npz(path, arrays):
    """np.savez, but with a fixed timestamp and no compression: the bytes depend on the arrays only."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        for name, a in arrays.items():
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.asanyarray(a), allow_pickle=False)
            zf.writestr(zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0)), buf.getvalue())


cal = {s: states(s, "calib-readout") for s in SIG}
test = {s: states(s, "play") for s in SIG}
pooled, edges, per_bin = fit_sensor(SIG)
cells = {}
rows = []
for s in SIG:
    z, fixed_pred, y, D, Kin = test[s]
    tb = np.digitize(D, edges)
    sensor_pred = np.empty(len(z), dtype=int)
    for k in range(16):
        if (tb == k).any():
            sensor_pred[tb == k] = lda_predict(per_bin[k], z[tb == k])
    sig_model = cells[s] = lda_fit(cal[s][0], cal[s][2])
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
if dec_out:
    low = [s for s in SIG if s <= 0.6]
    low_pooled, low_edges, low_bins = fit_sensor(low)
    arrays = {"sigmas": np.array(SIG), "edges": edges, "D_cal": np.array([float(cal[s][3].mean()) for s in SIG]),
              "low_sigmas": np.array(low), "low_edges": low_edges}
    models = ([("pooled", pooled), ("low_pooled", low_pooled)] + [(f"bin{k:02d}", m) for k, m in enumerate(per_bin)]
              + [(f"low_bin{k:02d}", m) for k, m in enumerate(low_bins)]
              + [(f"cell{i:02d}", cells[s]) for i, s in enumerate(SIG)])
    for name, (W, b) in models:
        arrays[f"{name}_W"], arrays[f"{name}_b"] = W, b
    save_npz(dec_out, arrays)
