"""Fringe probe of one checkpoint (post hoc play). For each sigma on a play test set of task A:

  acc_fixed   the model's own readout, as in phs-soliton
  acc_probe   a fresh regularised LDA on the final state z_T = [cos Theta, sin Theta, P], fit on one half, scored
              on the other: is the marker still decodable, whatever the trained readout thinks?
  phi, coh    memory phase: c^y_ik(sigma) = E[exp(i theta_ik) | y] - mean_y, the class signal of node i, channel k;
              phi = arg <c(sigma), c(0)>, coh = |<c(sigma), c(0)>| / (|c(sigma)| |c(0)|)
  acc_unwound the trained readout after the counter-rotation theta -> theta - phi(sigma) (P untouched)
  zmeans      class means of z_T minus their mean over classes (7 x 3Nd), for the rotation plane in dial.py
"""
import json
import math
import sys

import torch

LAB, ckpt, out = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, f"{LAB}/phs-soliton")
from phs import tasks  # noqa: E402
from phs.models import build  # noqa: E402
from phs.train import to_torch  # noqa: E402

torch.set_num_threads(int(sys.argv[4]) if len(sys.argv) > 4 else 2)
n = int(sys.argv[5]) if len(sys.argv) > 5 else 2048
c = torch.load(ckpt)
spec = tasks.TASKS["A"]
model = build(c["kind"], spec["vocab"], spec["classes"], c["cfg"], torch.Generator().manual_seed(0))
model.load_state_dict(c["state"])
model.eval()
T, N, d = c["T"], model.N, model.d
grab = {}
model.readout.register_forward_hook(lambda mod, inp, o: grab.__setitem__("z", inp[0].detach()))


@torch.no_grad()
def final_state(sigma):
    b = tasks.test_set("A", T, sigma, n=n, namespace="play")
    x, tg, sg = to_torch(b)
    logits = model(x, sg, generator=torch.Generator().manual_seed(tasks.derive_seed("playnoise", T, f"{sigma:.2f}")))
    z = grab["z"].double()
    return z, logits[:, -1].double(), tg[:, -1]


def lda(ztr, ytr, zte, yte, K=7, lam=1e-3):
    mu = torch.stack([ztr[ytr == k].mean(0) for k in range(K)])
    r = ztr - mu[ytr]
    S = r.T @ r / len(ztr)
    S = S + lam * S.diagonal().mean() * torch.eye(S.shape[0], dtype=S.dtype)
    W = torch.linalg.solve(S, mu.T)                              # (D, K)
    score = zte @ W - 0.5 * (mu * W.T).sum(1)
    return float((score.argmax(1) == yte).double().mean())


def class_signal(z, y, K=7):
    th = torch.atan2(z[:, N * d:2 * N * d], z[:, :N * d])       # theta from (cos, sin)
    e = torch.exp(1j * th)
    m = torch.stack([e[y == k].mean(0) for k in range(K)])      # (K, N d)
    return m - m.mean(0, keepdim=True)


sigmas = [round(0.05 * i, 2) for i in range(41)]
rows, c0 = [], None
for s in sigmas:
    z, logit, y = final_state(s)
    half = len(y) // 2
    sig = class_signal(z, y)
    if c0 is None:
        c0 = sig
    ov = (sig * c0.conj()).sum()
    phi = float(torch.angle(ov))
    coh = float(ov.abs() / (sig.abs().pow(2).sum().sqrt() * c0.abs().pow(2).sum().sqrt()))
    th = torch.atan2(z[:, N * d:2 * N * d], z[:, :N * d]) - phi
    zu = torch.cat([torch.cos(th), torch.sin(th), z[:, 2 * N * d:]], 1)
    acc_unwound = float((model.readout.double()(zu).argmax(1) == y).double().mean())
    model.readout.float()
    conf = torch.zeros(7, 7)
    for a, b in zip(y.tolist(), logit.argmax(1).tolist()):
        conf[a, b] += 1
    shift = [int(((conf[k].argmax() - k) % 7)) for k in range(7)]
    row = {"sigma": s, "acc_fixed": float((logit.argmax(1) == y).double().mean()),
           "acc_probe": lda(z[:half], y[:half], z[half:], y[half:]), "phi": phi, "coh": coh,
           "acc_unwound": acc_unwound, "pred_shift": shift}
    zm = torch.stack([z[y == k].mean(0) for k in range(7)])
    row_z = (zm - zm.mean(0, keepdim=True)).round(decimals=5).tolist()
    print(json.dumps(row), flush=True)
    row["zmeans"] = row_z
    rows.append(row)
json.dump({"ckpt": ckpt, "kind": c["kind"], "T": T, "step": c["step"], "n": n, "rows": rows}, open(out, "w"), indent=1)
