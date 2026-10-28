"""Thermal softening of the springs (post hoc play). At the end of the hold (token T-2, before the query) measure
per sigma the Debye-Waller-like factors of the three cosine springs, each relative to its sigma = 0 value:

  D_pin = E[cos(m theta)]                 pinning, m = m_pin
  D_lat = E[cos(theta_{i+1} - theta_i)]   lattice bonds
  D_att = E[C_ij], C_ij = (1/d) sum_k cos(theta_ik - theta_jk)   attention overlap

If the precession is a thermal softening of the springs, the dial angle of dial.py is a function of the softening
1 - D(sigma)/D(0) (linear at first), not of sigma^2 itself."""
import json
import sys

import torch

LAB, ckpt, out = sys.argv[1], sys.argv[2], sys.argv[3]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
sys.path.insert(0, f"{LAB}/phs-soliton")
from phs import tasks  # noqa: E402
from phs.core import Drive, internal_force, wrap  # noqa: E402
from phs.models import build  # noqa: E402
from phs.train import to_torch  # noqa: E402

torch.set_num_threads(int(sys.argv[5]) if len(sys.argv) > 5 else 1)
c = torch.load(ckpt)
spec = tasks.TASKS["A"]
model = build(c["kind"], spec["vocab"], spec["classes"], c["cfg"], torch.Generator().manual_seed(0))
model.load_state_dict(c["state"])
T, mp = c["T"], model.core.m_pin


SIGMAS = [round(0.05 * i, 2) for i in range(25)]


@torch.no_grad()
def hold_end():
    """Every sigma cell in one batch (sigma is per sequence); state after token T-2."""
    bs = [to_torch(tasks.test_set("A", T, s, n=n, namespace="play")) for s in SIGMAS]
    x, sg = torch.cat([b[0] for b in bs]), torch.cat([b[2] for b in bs])
    B = x.shape[0]
    gen = torch.Generator().manual_seed(17)
    core, phys = model.core, model.core.physics()
    G = core.rotation(phys)
    c1, c2 = core.ou_coefficients(sg)
    theta = torch.zeros(B, model.N, model.d)
    P = torch.zeros_like(theta)
    f = internal_force(theta, phys)
    for t in range(T - 1):
        tok = x[:, t]
        drive = Drive(phi=wrap(model.E_phi[tok]), u=model.u_max * torch.sigmoid(model.E_u[tok]))
        theta, P, f = core.token(theta, P, f, phys, G, drive, c1, c2, gen)
    return theta.double().reshape(len(SIGMAS), n, model.N, model.d), P.double().reshape(len(SIGMAS), n, model.N, model.d)


TH, PP = hold_end()
rows = []
for s, th, P in zip(SIGMAS, TH, PP):
    d = th.shape[-1]
    cs, sn = torch.cos(th), torch.sin(th)
    Cij = (cs @ cs.transpose(-1, -2) + sn @ sn.transpose(-1, -2)) / d
    off = ~torch.eye(th.shape[1], dtype=torch.bool)
    rows.append({"sigma": s,
                 "pin": float(torch.cos(mp * th).mean()),
                 "lat": float(torch.cos(torch.roll(th, -1, dims=1) - th).mean()),
                 "att": float(Cij[:, off].mean()),
                 "kinetic": float((P ** 2).mean())})
    r = rows[-1]
    print(f"sigma {s:.2f}  <cos m theta> {r['pin']:+.3f}  <cos dtheta> {r['lat']:+.3f}  <C_ij> {r['att']:+.3f}  "
          f"<P^2> {r['kinetic']:.3f}", flush=True)
json.dump({"kind": c["kind"], "step": c["step"], "rows": rows}, open(out, "w"))
