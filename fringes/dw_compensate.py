"""Debye-Waller compensation with no fitting (post hoc play). Rescale springs by their own measured thermal
factors from probe_soften.py, g -> g * D(0) / D(sigma), so each spring's effective stiffness g * D stays at its
sigma = 0 value, and score the trained readout on the fresh "comp" namespace.

  variant pin    g_pin only (D = <cos m theta>)
  variant all    g_pin, g_lat (D = <cos dtheta>) and kappa (D = <C_ij>)
  variant other  g_lat and kappa only: the control; if the pinning spring is the clock's hairspring, this one
                 should not bring the memory back"""
import json
import sys

import torch
import torch.nn.functional as F

LAB, ckpt, softfile, variant, out = sys.argv[1:6]
n = int(sys.argv[6]) if len(sys.argv) > 6 else 1024
sys.path.insert(0, f"{LAB}/phs-soliton")
from phs import tasks  # noqa: E402
from phs.core import inv_softplus  # noqa: E402
from phs.models import build  # noqa: E402
from phs.train import to_torch  # noqa: E402

torch.set_num_threads(1)
c = torch.load(ckpt)
spec = tasks.TASKS["A"]
m = build(c["kind"], spec["vocab"], spec["classes"], c["cfg"], torch.Generator().manual_seed(0))
m.load_state_dict(c["state"])
m.eval()
soft = {r["sigma"]: r for r in json.load(open(softfile))["rows"]}
d0 = soft[0.0]
springs = {"pin": ("w_pin", "pin"), "lat": ("w_lat", "lat"), "kappa": ("w_kappa", "att")}
use = {"pin": ["pin"], "all": ["pin", "lat", "kappa"], "other": ["lat", "kappa"]}[variant]
g0 = {k: float(F.softplus(getattr(m.core, springs[k][0]))) for k in springs}
rows = []
for s in sorted(soft):
    fac = {k: (d0[springs[k][1]] / soft[s][springs[k][1]] if k in use else 1.0) for k in springs}
    x, tg, sg = to_torch(tasks.test_set("A", c["T"], s, n=n, namespace="comp"))
    with torch.no_grad():
        for k in springs:
            getattr(m.core, springs[k][0]).fill_(inv_softplus(fac[k] * g0[k]))
        lg = m(x, sg, generator=torch.Generator().manual_seed(13))[:, -1]
    a = float((lg.argmax(1) == tg[:, -1]).double().mean())
    rows.append({"sigma": s, "acc": a, "factors": fac})
    print(f"{variant:5s} sigma {s:.2f}  " + " ".join(f"{k} x{fac[k]:.3f}" for k in springs) + f"  Acc {a:.3f}", flush=True)
json.dump({"variant": variant, "n": n, "rows": rows}, open(out, "w"))
