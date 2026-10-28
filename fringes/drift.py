"""Does floating-point drift matter for a clock memory? (post hoc play)

1. float32 vs float64: the same trained model and the same 4096 sequences at sigma = 0, run in both precisions;
   argmax disagreements and the largest logit difference.
2. Neutral directions: in float64, perturb theta by eps * (random unit vector) right after the marker (token 2) and
   follow |delta theta(t)| to the query, sigma = 0. A contracting system forgets the perturbation (rounding drift
   cannot accumulate); a clock has a neutral phase direction along which it persists (drift random-walks along it).
Usage: drift.py LAB ckpt [n]"""
import json
import sys

import torch

LAB, ckpt = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
sys.path.insert(0, f"{LAB}/phs-soliton")
from phs import tasks  # noqa: E402
from phs.core import Drive, internal_force, wrap  # noqa: E402
from phs.models import build  # noqa: E402
from phs.train import to_torch  # noqa: E402

torch.set_num_threads(2)
c = torch.load(ckpt)
spec = tasks.TASKS["A"]


def load(dtype):
    m = build(c["kind"], spec["vocab"], spec["classes"], c["cfg"], torch.Generator().manual_seed(0))
    m.load_state_dict(c["state"])
    return m.to(dtype).eval()


x, tg, sg = to_torch(tasks.test_set("A", 32, 0.0, n=4096, namespace="comp"))
with torch.no_grad():
    l32 = load(torch.float32)(x, sg)[:, -1].double()
    l64 = load(torch.float64)(x, sg.double())[:, -1]
flips = int((l32.argmax(1) != l64.argmax(1)).sum())
print(f"{c['kind']} seed {c['seed']}: float32 vs float64 at sigma 0, 4096 seqs: argmax flips {flips}, "
      f"max |logit diff| {float((l32 - l64).abs().max()):.2e}, acc32 {float((l32.argmax(1) == tg[:, -1]).double().mean()):.4f}")

m = load(torch.float64)
core, phys = m.core, m.core.physics()
G = core.rotation(phys)
x, _, _ = to_torch(tasks.test_set("A", 32, 0.0, n=n, namespace="drift"))
c1, c2 = core.ou_coefficients(torch.zeros(n, dtype=torch.float64))
eps = 1e-6
gen = torch.Generator().manual_seed(5)
v = torch.randn(n, m.N, m.d, generator=gen, dtype=torch.float64)
v = v / v.flatten(1).norm(dim=1)[:, None, None]


@torch.no_grad()
def traj(perturb):
    th = torch.zeros(n, m.N, m.d, dtype=torch.float64)
    P = torch.zeros_like(th)
    f = internal_force(th, phys)
    out = []
    for t in range(32):
        if perturb and t == 2:
            th = wrap(th + eps * v)
            f = internal_force(th, phys)
        tok = x[:, t]
        drive = Drive(phi=wrap(m.E_phi[tok]), u=m.u_max * torch.sigmoid(m.E_u[tok]))
        th, P, f = core.token(th, P, f, phys, G, drive, c1, c2, None)
        out.append(th.clone())
    return out


a, b = traj(False), traj(True)
gain = [float((wrap(bb - aa)).flatten(1).norm(dim=1).median() / eps) for aa, bb in zip(a, b)]
print("   |delta theta(t)| / eps, median over sequences, t = 2, 4, 8, 16, 24, 30, 31:",
      " ".join(f"{gain[t]:.3g}" for t in (2, 4, 8, 16, 24, 30, 31)))
json.dump({"kind": c["kind"], "seed": c["seed"], "flips": flips, "gain": gain}, open(sys.argv[4] if len(sys.argv) > 4
          else "/dev/null", "w"))
