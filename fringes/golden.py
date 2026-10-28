"""Golden-twist chain (post hoc play): a Soliton-PHS whose lattice bonds prefer a twist delta instead of 0,

    V_FK = g_lat sum_i (1 - cos(theta_{i+1} - theta_i - delta)) + g_pin sum_i (1 - cos(m theta_i)),

on a ring of N = 34 (a Fibonacci number) with delta = 2 pi * 21/34, the Fibonacci approximant of the golden angle
(neighbours want to sit like successive leaves of a sunflower). Against the pinning (m = 2) this is a Frenkel-Kontorova
chain with near-golden incommensurability: Aubry's last chain to pin. The control is the same ring with delta = 0.

Prediction: if the twisted chain stays unpinned (an Aubry sliding phase), its memory has a zero-frequency phason and
no clock, so its accuracy decays without fringes; if it is pinned, the clock and the fringes come back.

The patch replaces phs.core.lattice_force in this process only; phs-soliton is not changed.
Usage: golden.py LAB q N seed steps out_dir [threads]   (q = 0 is the untwisted control)"""
import json
import math
import sys
import time

import torch

LAB = sys.argv[1]
q, N, seed, steps, out = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
torch.set_num_threads(int(sys.argv[7]) if len(sys.argv) > 7 else 1)
sys.path.insert(0, f"{LAB}/phs-soliton")
import phs.core as core  # noqa: E402
from phs import tasks  # noqa: E402
from phs.models import build  # noqa: E402
from phs.train import loss_and_hits, to_torch  # noqa: E402
from tools.confirm import model_config  # noqa: E402

DELTA = 2 * math.pi * q / N


def twisted_lattice_force(theta, g_lat, g_pin, m_pin):
    fwd = torch.roll(theta, -1, dims=-2) - theta - DELTA
    bwd = theta - torch.roll(theta, 1, dims=-2) - DELTA
    return g_lat * (torch.sin(fwd) - torch.sin(bwd)) - m_pin * g_pin * torch.sin(m_pin * theta)


core.lattice_force = twisted_lattice_force

# the force is the gradient of the twisted potential (checked once, float64, by autograd)
th = torch.randn(3, N, 2, dtype=torch.float64, requires_grad=True)
gl, gp = torch.tensor(1.3, dtype=torch.float64), torch.tensor(0.7, dtype=torch.float64)
V = (gl * (1 - torch.cos(torch.roll(th, -1, dims=-2) - th - DELTA)) + gp * (1 - torch.cos(2 * th))).sum()
(grad,) = torch.autograd.grad(V, th)
assert torch.allclose(-grad, twisted_lattice_force(th.detach(), gl, gp, 2), atol=1e-10), "twisted force != -grad V"

p = json.load(open(f"{LAB}/phs-soliton/prereg.json"))
cfg = dict(model_config(p, "soliton", "A"), N=N)
spec = tasks.TASKS["A"]
tag = f"golden-q{q}-N{N}-s{seed}"
model = build("soliton", spec["vocab"], spec["classes"], cfg, torch.Generator().manual_seed(tasks.derive_seed("init", tag)))
noise = torch.Generator().manual_seed(tasks.derive_seed("noise", tag))
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
torch.save({"kind": "soliton", "T": 32, "seed": seed, "step": step, "cfg": cfg, "q": q, "state": model.state_dict()},
           f"{out}/{tag}-step{step:05d}.pt")

# evaluation: accuracy over sigma (all cells in one batch) and the order of the state at the end of the hold
model.eval()
SIG = [round(0.05 * i, 2) for i in range(31)]
n = 1024
bs = [to_torch(tasks.test_set("A", 32, s, n=n, namespace="golden")) for s in SIG]
X, Y, S = torch.cat([b[0] for b in bs]), torch.cat([b[1][:, -1] for b in bs]), torch.cat([b[2] for b in bs])
grab = {}
model.readout.register_forward_hook(lambda mod, inp, o: grab.__setitem__("z", inp[0].detach()))
with torch.no_grad():
    logits, diag = model(X, S, generator=torch.Generator().manual_seed(13), diagnostics=True)
acc = (logits[:, -1].argmax(1) == Y).double().reshape(len(SIG), n).mean(1)
th = diag["theta_T"].double().reshape(len(SIG), n, N, -1)
cs, sn = torch.cos(th), torch.sin(th)
C = (cs @ cs.transpose(-1, -2) + sn @ sn.transpose(-1, -2)) / th.shape[-1]
off = ~torch.eye(N, dtype=torch.bool)
bond = core.wrap(torch.roll(th, -1, dims=2) - th)
rows = []
for j, s in enumerate(SIG):
    rows.append({"sigma": s, "acc": round(float(acc[j]), 4),
                 "C": round(float(C[j][:, off].mean()), 4),
                 "pin": round(float(torch.cos(2 * th[j]).mean()), 4),
                 "twist": round(float(torch.cos(bond[j] - DELTA).mean()), 4),
                 "q_top": round(float(diag["q_top"][j * n:(j + 1) * n, -1].abs().mean()), 3)})
print("sigma " + " ".join(f"{r['sigma']:.2f}" for r in rows[::2]))
print("acc   " + " ".join(f"{r['acc']:.2f}" for r in rows[::2]))
print("C     " + " ".join(f"{r['C']:.2f}" for r in rows[::2]))
print("twist " + " ".join(f"{r['twist']:.2f}" for r in rows[::2]))
print("pin   " + " ".join(f"{r['pin']:.2f}" for r in rows[::2]))
print("|Q|   " + " ".join(f"{r['q_top']:.1f}" for r in rows[::2]))
json.dump({"tag": tag, "q": q, "N": N, "delta": DELTA, "physics": model.learned_physics(), "rows": rows},
          open(f"{out}/{tag}.json", "w"))
