"""Homeostatic Soliton-PHS, trained end to end (post hoc play). The pinning force is normalised by its own order
parameter, read from the state of each sequence:

    F_pin = -m (g_pin / D_t) sin(m theta),   D_t = EMA over tokens of mean_ik cos(m theta_ik)   (clamped >= 0.3)

so the Debye-Waller-dressed stiffness g_pin * D / D_t stays at its trained value whatever the temperature. D_t is a
detached sensor reading (a reflex, no gradient through it). Everything else is Soliton-PHS with the PREREG v1 settings
of task A; training sees sigma in [0, 0.35] as before. homeostasis.py did this at evaluation only; here the model
learns with the reflex in place.

Prediction (written before the runs): no anti-recall and no fringes on sigma in [0, 1.5], and sigma_crit (Acc >= 0.8)
above the plain Soliton-PHS of the same seeds (0.43 on seeds 900-902), at or above Pure-PHS (0.68).
Usage: homeo_arch.py LAB seed steps out_dir [threads] [ema]"""
import json
import sys
import time

import torch

LAB, seed, steps, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
torch.set_num_threads(int(sys.argv[5]) if len(sys.argv) > 5 else 1)
EMA = float(sys.argv[6]) if len(sys.argv) > 6 else 0.3
sys.path.insert(0, f"{LAB}/phs-soliton")
from phs import tasks  # noqa: E402
from phs.core import Drive, Physics, internal_force, q_top, wrap  # noqa: E402
from phs.models import build  # noqa: E402
from phs.train import loss_and_hits, to_torch  # noqa: E402
from tools.confirm import model_config  # noqa: E402


def homeostatic_forward(self, x, sigma, generator=None, diagnostics=False):
    B, T = x.shape
    core = self.core
    phys = core.physics()
    G = core.rotation(phys)
    c1, c2 = core.ou_coefficients(sigma.to(self.E_u.dtype))
    theta = torch.zeros(B, self.N, self.d, dtype=self.E_u.dtype)
    P = torch.zeros_like(theta)
    Dhat = torch.ones(B, dtype=theta.dtype)
    logits, charges, sensor = [], [], []
    for t in range(T):
        D = torch.cos(core.m_pin * theta.detach()).mean((1, 2))
        Dhat = D if t == 0 else (1 - EMA) * Dhat + EMA * D
        g_eff = phys.g_pin / Dhat.clamp_min(0.3)[:, None, None]
        ph = Physics(kappa=phys.kappa, g_lat=phys.g_lat, g_pin=g_eff, m_pin=phys.m_pin, mass=phys.mass, J=phys.J,
                     lattice=True)
        f = internal_force(theta, ph)
        tok = x[:, t]
        drive = Drive(phi=wrap(self.E_phi[tok]), u=self.u_max * torch.sigmoid(self.E_u[tok]))
        theta, P, f = core.token(theta, P, f, ph, G, drive, c1, c2, generator)
        z = torch.cat([torch.cos(theta), torch.sin(theta), P], dim=-1).reshape(B, -1)
        logits.append(self.readout(z))
        if diagnostics:
            charges.append(q_top(theta.detach()))
            sensor.append(Dhat)
    out_ = torch.stack(logits, dim=1)
    if not diagnostics:
        return out_
    return out_, {"q_top": torch.stack(charges, dim=1), "theta_T": theta.detach(), "D": torch.stack(sensor, 1)}


p = json.load(open(f"{LAB}/phs-soliton/prereg.json"))
cfg = model_config(p, "soliton", "A")
spec = tasks.TASKS["A"]
tag = f"homeoarch-s{seed}"
model = build("soliton", spec["vocab"], spec["classes"], cfg,
              torch.Generator().manual_seed(tasks.derive_seed("init", "soliton", "A", 32, seed)))
model.forward = homeostatic_forward.__get__(model)
noise = torch.Generator().manual_seed(tasks.derive_seed("noise", "soliton", "A", 32, seed))
opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
stream = tasks.train_stream("A", 32, seed)
t0 = time.time()
for step in range(1, steps + 1):
    x, tg, sg = to_torch(tasks.batch("A", stream, cfg["batch"], 32))
    loss, hits = loss_and_hits("A", model(x, sg, generator=noise), tg)
    if not torch.isfinite(loss):
        print(f"diverged at {step}", flush=True)
        break
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
    opt.step()
    if step % 200 == 0:
        print(f"{tag} step {step} loss {float(loss.detach()):.4f} acc {float(hits.double().mean()):.3f} "
              f"{time.time() - t0:.0f}s {json.dumps(model.learned_physics())}", flush=True)
torch.save({"kind": "soliton", "arch": "homeostatic", "ema": EMA, "T": 32, "seed": seed, "step": step, "cfg": cfg,
            "state": model.state_dict()}, f"{out}/{tag}-step{step:05d}.pt")

# evaluation on the same fresh cells as homeostasis.py ("comp", 1024 per sigma), all sigma in one batch
model.eval()
SIG = [round(0.05 * i, 2) for i in range(31)]
n = 1024
bs = [to_torch(tasks.test_set("A", 32, s, n=n, namespace="comp")) for s in SIG]
X, Y, S = torch.cat([b[0] for b in bs]), torch.cat([b[1][:, -1] for b in bs]), torch.cat([b[2] for b in bs])
with torch.no_grad():
    logits, diag = model(X, S, generator=torch.Generator().manual_seed(13), diagnostics=True)
acc = (logits[:, -1].argmax(1) == Y).double().reshape(len(SIG), n).mean(1)
th = diag["theta_T"].double().reshape(len(SIG), n, model.N, model.d)
cs, sn = torch.cos(th), torch.sin(th)
C = (cs @ cs.transpose(-1, -2) + sn @ sn.transpose(-1, -2)) / model.d
off = ~torch.eye(model.N, dtype=torch.bool)
Dend = diag["D"][:, -1].double().reshape(len(SIG), n)
rows = [{"sigma": s, "acc": round(float(acc[j]), 4), "C": round(float(C[j][:, off].mean()), 4),
         "pin": round(float(torch.cos(2 * th[j]).mean()), 4), "sensor": round(float(Dend[j].mean()), 4)}
        for j, s in enumerate(SIG)]
crit = -0.05
for r in rows:
    if r["acc"] >= 0.8:
        crit = r["sigma"]
    else:
        break
print("sigma  " + " ".join(f"{r['sigma']:.2f}" for r in rows[::2]))
print("acc    " + " ".join(f"{r['acc']:.2f}" for r in rows[::2]))
print("C      " + " ".join(f"{r['C']:.2f}" for r in rows[::2]))
print("sensor " + " ".join(f"{r['sensor']:.2f}" for r in rows[::2]))
print(f"sigma_crit {crit}  min Acc {min(r['acc'] for r in rows):.3f}")
json.dump({"tag": tag, "seed": seed, "ema": EMA, "physics": model.learned_physics(), "sigma_crit": crit, "rows": rows},
          open(f"{out}/{tag}.json", "w"))
