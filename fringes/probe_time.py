"""Memory phase along the sequence (post hoc play): does the class signal precess while it is being held?
For each sigma, phi(t, sigma) = arg <c(t, sigma), c(t, 0)> with c^y_ik(t) = E[exp(i theta_ik(t)) | y] - mean_y.
A clock memory gives phi growing linearly in t at a rate proportional to sigma^2; a readout transient gives a
phase that appears only at the last (query) token."""
import json
import sys

import torch

LAB, ckpt, out = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, f"{LAB}/phs-soliton")
from phs import tasks  # noqa: E402
from phs.core import Drive, internal_force, wrap  # noqa: E402
from phs.models import build  # noqa: E402
from phs.train import to_torch  # noqa: E402

torch.set_num_threads(int(sys.argv[4]) if len(sys.argv) > 4 else 2)
n = int(sys.argv[5]) if len(sys.argv) > 5 else 2048
T_eval = int(sys.argv[6]) if len(sys.argv) > 6 else None
c = torch.load(ckpt)
spec = tasks.TASKS["A"]
model = build(c["kind"], spec["vocab"], spec["classes"], c["cfg"], torch.Generator().manual_seed(0))
model.load_state_dict(c["state"])
T = T_eval or c["T"]


@torch.no_grad()
def trajectory(sigma):
    b = tasks.test_set("A", T, sigma, n=n, namespace="play")
    x, tg, sg = to_torch(b)
    gen = torch.Generator().manual_seed(tasks.derive_seed("playnoise", T, f"{sigma:.2f}"))
    core, phys = model.core, model.core.physics()
    G = core.rotation(phys)
    c1, c2 = core.ou_coefficients(sg)
    theta = torch.zeros(n, model.N, model.d)
    P = torch.zeros_like(theta)
    f = internal_force(theta, phys)
    thetas, hits = [], None
    for t in range(T):
        tok = x[:, t]
        drive = Drive(phi=wrap(model.E_phi[tok]), u=model.u_max * torch.sigmoid(model.E_u[tok]))
        theta, P, f = core.token(theta, P, f, phys, G, drive, c1, c2, gen)
        thetas.append(theta.reshape(n, -1).double())
    z = torch.cat([torch.cos(theta), torch.sin(theta), P], -1).reshape(n, -1)
    hits = float((model.readout(z).argmax(1) == tg[:, -1]).double().mean())
    return torch.stack(thetas, 1), tg[:, -1], hits


def signal(th, y, K=7):
    e = torch.exp(1j * th)                                         # (n, T, Nd)
    m = torch.stack([e[y == k].mean(0) for k in range(K)])         # (K, T, Nd)
    return m - m.mean(0, keepdim=True)


sigmas = [0.0, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
ref, rows = None, []
for s in sigmas:
    th, y, acc = trajectory(s)
    sig = signal(th, y)
    if ref is None:
        ref = sig
    ov = (sig * ref.conj()).sum((0, 2))                            # (T,)
    norm = sig.abs().pow(2).sum((0, 2)).sqrt() * ref.abs().pow(2).sum((0, 2)).sqrt()
    phi = torch.angle(ov)
    phi_unwrapped = torch.cumsum(torch.cat([phi[:1], wrap(phi[1:] - phi[:-1])]), 0)
    rows.append({"sigma": s, "acc": acc, "phi": [round(float(v), 4) for v in phi_unwrapped],
                 "coh": [round(float(v), 4) for v in ov.abs() / norm],
                 "amp": [round(float(v), 5) for v in sig.abs().pow(2).sum((0, 2)).sqrt()]})
    print(f"sigma {s:.2f} acc {acc:.3f} phi(t=2,T/2,T-2,T-1) = "
          f"{phi_unwrapped[2]:+.2f} {phi_unwrapped[T // 2]:+.2f} {phi_unwrapped[T - 2]:+.2f} {phi_unwrapped[T - 1]:+.2f}"
          f"  coh(T-2) {float(ov.abs()[T - 2] / norm[T - 2]):.2f}", flush=True)
json.dump({"ckpt": ckpt, "T": T, "rows": rows}, open(out, "w"))
