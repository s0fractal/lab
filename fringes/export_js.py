"""Export a trained PHS model for the in-browser engine (post hoc play): physics, drives, the Cayley rotation G,
readout, plus reference logits from PyTorch for a fixed sequence at sigma = 0, so the JS port can be checked.
Usage: export_js.py LAB ckpt out.json [pin_mode] [q]   pin_mode: plain | qp | homeo"""
import json
import math
import sys

import torch

LAB, ckpt, out = sys.argv[1], sys.argv[2], sys.argv[3]
mode = sys.argv[4] if len(sys.argv) > 4 else "plain"
q = int(sys.argv[5]) if len(sys.argv) > 5 else 0
sys.path.insert(0, f"{LAB}/phs-soliton")
import phs.core as core  # noqa: E402
from phs import tasks  # noqa: E402
from phs.models import build  # noqa: E402

c = torch.load(ckpt)
cfg = c["cfg"]
spec = tasks.TASKS["A"]
kind = c["kind"]
m = build(kind, spec["vocab"], spec["classes"], cfg, torch.Generator().manual_seed(0))
m.load_state_dict(c["state"])
m.eval()
N, d = m.N, m.d
psi = [2 * math.pi * q * i / N for i in range(N)] if mode == "qp" else [0.0] * N
if mode == "qp":
    P_ = torch.tensor(psi, dtype=torch.float32)[:, None]

    def lf(theta, g_lat, g_pin, mp):
        fwd = torch.roll(theta, -1, dims=-2) - theta
        bwd = theta - torch.roll(theta, 1, dims=-2)
        return g_lat * (torch.sin(fwd) - torch.sin(bwd)) - mp * g_pin * torch.sin(mp * (theta - P_))
    core.lattice_force = lf
with torch.no_grad():
    ph = m.core.physics()
    G = core.cayley_rotation(ph, 0.5 * m.core.h)
    R = lambda t, k=6: [round(float(v), k) for v in t.flatten().tolist()]  # noqa: E731
    exp = {"kind": kind, "mode": mode, "N": N, "d": d, "S": m.core.S, "h": m.core.h, "gamma": m.core.gamma,
           "kappa": float(ph.kappa), "g_lat": float(ph.g_lat), "g_pin": float(ph.g_pin), "m_pin": ph.m_pin,
           "mass": R(ph.mass), "G": R(G), "psi": [round(v, 6) for v in psi], "ema": c.get("ema", 0.3),
           "phi": R(core.wrap(m.E_phi)), "u": R(m.u_max * torch.sigmoid(m.E_u)),
           "W": R(m.readout.weight), "b": R(m.readout.bias), "seed": c["seed"], "step": c["step"],
           "physics": m.learned_physics()}
    # reference: one fixed task-A sequence at sigma = 0, logits of the last token (plain and qp modes only)
    x = torch.tensor([[3, 3] + [0, 0, 5, 0, 0, 0, 2, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0]
                      + [8]])
    if mode != "homeo":
        exp["ref"] = {"x": x[0].tolist(), "logits": R(m(x, torch.zeros(1))[0, -1])}
json.dump(exp, open(out, "w"), separators=(",", ":"))
print(out, kind, mode, "N", N, "logits" if "ref" in exp else "", exp.get("ref", {}).get("logits"))
