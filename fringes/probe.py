"""Fringe probe of one checkpoint (post hoc play). For each sigma on a play test set of task A:

  acc_fixed   the model's own readout, as in phs-soliton
  acc_probe   a fresh regularised LDA on the final state z_T = [cos Theta, sin Theta, P], fit on one half, scored
              on the other: is the marker still decodable, whatever the trained readout thinks?
  phi, coh    memory phase: c^y_ik(sigma) = E[exp(i theta_ik) | y] - mean_y, the class signal of node i, channel k;
              phi = arg <c(sigma), c(0)>, coh = |<c(sigma), c(0)>| / (|c(sigma)| |c(0)|)
  acc_unwound the trained readout after the counter-rotation theta -> theta - phi(sigma) (P untouched)
  zmeans      class means of z_T minus their mean over classes (7 x 3Nd), for the rotation plane in dial.py

The readout input is built node by node, cat([cos, sin, P], -1) on (B, N, d), then flattened (phs/models.py), so
z.reshape(B, N, 3d) splits it back. Until the Codex review (F3) this file read z as three contiguous blocks of N d,
which mixed nodes and momenta: phi, coh and acc_unwound of every earlier probe-*.json are wrong; acc_fixed,
acc_probe and zmeans never unpacked z and stand. probe.py --selftest checks the layout without a checkpoint.
"""
import json
import sys

import torch


def unpack(z, N, d):
    """(B, 3Nd) readout input -> theta (B, N, d), P (B, N, d)."""
    z = z.reshape(len(z), N, 3 * d)
    return torch.atan2(z[..., d:2 * d], z[..., :d]), z[..., 2 * d:]


def pack(theta, P):
    return torch.cat([torch.cos(theta), torch.sin(theta), P], -1).reshape(len(theta), -1)


def class_signal(z, y, N, d, K=7):
    e = torch.exp(1j * unpack(z, N, d)[0]).reshape(len(z), -1)
    m = torch.stack([e[y == k].mean(0) for k in range(K)])      # (K, N d)
    return m - m.mean(0, keepdim=True)


def overlap(sig, ref):
    ov = (sig * ref.conj()).sum()
    return float(torch.angle(ov)), float(ov.abs() / (sig.abs().pow(2).sum().sqrt() * ref.abs().pow(2).sum().sqrt()))


def unwind(z, phi, N, d):
    th, P = unpack(z, N, d)
    return pack(th - phi, P)


def selftest():
    g = torch.Generator().manual_seed(1)
    n, N, d, K = 700, 5, 2, 7
    y = torch.arange(n) % K
    th = torch.randn(n, N, d, generator=g, dtype=torch.float64) * 0.3 + 0.9 * y[:, None, None]
    P = torch.randn(n, N, d, generator=g, dtype=torch.float64) * 3
    z = pack(th, P)
    errs = {"zero rotation is the identity": float((unwind(z, 0.0, N, d) - z).abs().max())}
    a = 1.234
    phi, coh = overlap(class_signal(pack(th + a, P), y, N, d), class_signal(z, y, N, d))
    errs["a global shift a reads as phi = a"] = abs(phi - a)
    errs["... with coh = 1"] = abs(coh - 1)
    zu = unwind(pack(th + a, P), a, N, d)
    errs["unwinding the shift returns z"] = float((zu - z).abs().max())
    errs["P is untouched"] = float((unpack(zu, N, d)[1] - P).abs().max())
    for k, v in errs.items():
        print(f"{'ok ' if v < 1e-9 else 'BAD'} {k}: {v:.2e}")
    # bite: the old contiguous-block reading fails the identity
    old = torch.atan2(z[:, N * d:2 * N * d], z[:, :N * d])
    bite = float((torch.cat([old.cos(), old.sin(), z[:, 2 * N * d:]], 1) - z).abs().max())
    print(f"{'ok ' if bite > 0.1 else 'BAD'} the old block layout fails the identity: {bite:.2f}")
    sys.exit(0 if max(errs.values()) < 1e-9 and bite > 0.1 else 1)


if sys.argv[1] == "--selftest":
    selftest()

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
if c.get("arch") == "homeostatic":
    sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
    from homeo_forward import bind
    bind(model, c.get("ema", 0.3))
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


# each sigma has its own test set and noise seed, so a subset (argv 6, comma-separated, starting at 0) reproduces
# the same rows as the full scan; artifacts/verify.py uses this to replay a frozen result from the weights
sigmas = ([float(v) for v in sys.argv[6].split(",")] if len(sys.argv) > 6 else [round(0.05 * i, 2) for i in range(41)])
rows, c0 = [], None
for s in sigmas:
    z, logit, y = final_state(s)
    half = len(y) // 2
    sig = class_signal(z, y, N, d)
    if c0 is None:
        c0 = sig
    phi, coh = overlap(sig, c0)
    zu = unwind(z, phi, N, d)
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
json.dump({"version": 2, "ckpt": ckpt, "kind": c["kind"], "T": T, "step": c["step"], "n": n, "rows": rows}, open(out, "w"),
          indent=1)
