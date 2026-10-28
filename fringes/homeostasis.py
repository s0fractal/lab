"""Homeostatic clock (post hoc play): can the Soliton-PHS memory correct its own rate without being told sigma?
The model reads its own temperature from its state: D(t) = mean_i,k cos(m theta_ik), the Debye-Waller factor of the
pinning spring, and at every token rescales g_pin -> g_pin * D0(t) / D(t), with D0(t) the sigma = 0 profile from a
separate calibration set. Nothing in the loop knows sigma.

  none        the trained model
  self        D(t) per sequence, smoothed by an EMA over tokens: one sequence regulates itself
  population  D(t) averaged over the 1024 sequences of the cell: a shared thermometer (still blind to sigma)"""
import json
import sys

import torch
import torch.nn.functional as F

LAB, ckpt, out = sys.argv[1], sys.argv[2], sys.argv[3]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
EMA = float(sys.argv[5]) if len(sys.argv) > 5 else 0.3
sys.path.insert(0, f"{LAB}/phs-soliton")
from phs import tasks  # noqa: E402
from phs.core import Drive, Physics, internal_force, wrap  # noqa: E402
from phs.models import build  # noqa: E402
from phs.train import to_torch  # noqa: E402

torch.set_num_threads(int(sys.argv[6]) if len(sys.argv) > 6 else 1)
MODES = sys.argv[7].split(",") if len(sys.argv) > 7 else ["none", "self", "population"]
c = torch.load(ckpt)
spec = tasks.TASKS["A"]
model = build(c["kind"], spec["vocab"], spec["classes"], c["cfg"], torch.Generator().manual_seed(0))
model.load_state_dict(c["state"])
model.eval()
T, mp = c["T"], model.core.m_pin
SIGMAS = [round(0.05 * i, 2) for i in range(25)]


@torch.no_grad()
def run(x, sg, mode, D0=None, groups=None, seed=13):
    B = x.shape[0]
    core, phys = model.core, model.core.physics()
    G = core.rotation(phys)
    c1, c2 = core.ou_coefficients(sg)
    gen = torch.Generator().manual_seed(seed)
    theta = torch.zeros(B, model.N, model.d)
    P = torch.zeros_like(theta)
    Dhat = torch.ones(B)
    prof = []
    for t in range(T):
        D = torch.cos(mp * theta).mean((1, 2))
        if mode == "self":
            Dhat = D if t == 0 else (1 - EMA) * Dhat + EMA * D
            ref = D0[t]
        elif mode == "population":
            Dhat = D.reshape(groups, -1).mean(1).repeat_interleave(B // groups)
            ref = D0[t]
        if mode in ("self", "population"):
            fac = (ref / Dhat.clamp_min(0.05)).clamp(0.5, 2.0)
            ph = Physics(kappa=phys.kappa, g_lat=phys.g_lat, g_pin=phys.g_pin * fac[:, None, None], m_pin=phys.m_pin,
                         mass=phys.mass, J=phys.J, lattice=phys.lattice)
        else:
            ph = phys
        f = internal_force(theta, ph)
        tok = x[:, t]
        drive = Drive(phi=wrap(model.E_phi[tok]), u=model.u_max * torch.sigmoid(model.E_u[tok]))
        theta, P, f = core.token(theta, P, f, ph, G, drive, c1, c2, gen)
        prof.append(float(D.mean()))
    z = torch.cat([torch.cos(theta), torch.sin(theta), P], -1).reshape(B, -1)
    return model.readout(z), prof


# calibration: the sigma = 0 profile of D(t), on its own sequences
xc, _, sc = to_torch(tasks.test_set("A", T, 0.0, n=n, namespace="calib-homeo"))
_, D0 = run(xc, sc, "none")
D0 = [1.0] + D0[:-1]                     # D(t) is read before token t: the profile of the state after token t-1
cells = [to_torch(tasks.test_set("A", T, s, n=n, namespace="comp")) for s in SIGMAS]
X, Y, S = torch.cat([b[0] for b in cells]), torch.cat([b[1][:, -1] for b in cells]), torch.cat([b[2] for b in cells])
res = {}
for mode in MODES:
    logits, _ = run(X, S, mode, D0=D0, groups=len(SIGMAS))
    res[mode] = [round(float(v), 4) for v in (logits.argmax(1) == Y).double().reshape(len(SIGMAS), n).mean(1)]
    print(mode.ljust(10), " ".join(f"{a:.2f}" for a in res[mode][::2]), flush=True)
print("sigma     ", " ".join(f"{s:.2f}" for s in SIGMAS[::2]))
json.dump({"kind": c["kind"], "ckpt": ckpt, "n": n, "ema": EMA, "sigma": SIGMAS, "D0": D0, "acc": res}, open(out, "w"))
