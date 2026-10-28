"""Interference map (post hoc play): accuracy of the trained readout over sigma x a scale factor f of one
stiffness (g_pin, g_lat or kappa), set at evaluation with no retraining. If the memory phase is
a sigma^2 + b (f - 1), the fringes are parabolas in this plane and the ridge f*(sigma) of best accuracy is the
compensation curve: the stiffness that keeps the clock on time at temperature sigma^2 (Harrison's gridiron
pendulum, for a reservoir)."""
import json
import sys

import torch
import torch.nn.functional as F

LAB, ckpt, which, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
n = int(sys.argv[5]) if len(sys.argv) > 5 else 512
sys.path.insert(0, f"{LAB}/phs-soliton")
from phs import tasks  # noqa: E402
from phs.core import inv_softplus  # noqa: E402
from phs.models import build  # noqa: E402
from phs.train import to_torch  # noqa: E402

torch.set_num_threads(int(sys.argv[6]) if len(sys.argv) > 6 else 1)
c = torch.load(ckpt)
spec = tasks.TASKS["A"]
m = build(c["kind"], spec["vocab"], spec["classes"], c["cfg"], torch.Generator().manual_seed(0))
m.load_state_dict(c["state"])
m.eval()
w = getattr(m.core, {"pin": "w_pin", "lat": "w_lat", "kappa": "w_kappa"}[which])
g0 = float(F.softplus(w))
sigmas = [round(0.05 * i, 2) for i in range(25)]
factors = [round(0.4 + 0.1 * i, 2) for i in range(17)]          # 0.4 .. 2.0
cells = {s: tasks.test_set("A", c["T"], s, n=n, namespace="map") for s in sigmas}
# every sigma cell in one batch (sigma is per sequence), so the token loop runs once per factor
batches = [to_torch(cells[s]) for s in sigmas]
X = torch.cat([b[0] for b in batches])
Y = torch.cat([b[1][:, -1] for b in batches])
S = torch.cat([b[2] for b in batches])
grid = []
for f in factors:
    with torch.no_grad():
        w.fill_(inv_softplus(f * g0))
        lg = m(X, S, generator=torch.Generator().manual_seed(11))[:, -1]
    hit = (lg.argmax(1) == Y).double().reshape(len(sigmas), n)
    row = [round(float(v), 4) for v in hit.mean(1)]
    grid.append(row)
    print(f"{c['kind']} {which} x{f:.1f}: " + " ".join(f"{a:.2f}" for a in row[::2]), flush=True)
json.dump({"kind": c["kind"], "step": c["step"], "which": which, "g0": g0, "n": n, "sigma": sigmas,
           "factor": factors, "acc": grid}, open(out, "w"))
