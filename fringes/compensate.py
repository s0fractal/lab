"""Harrison compensation (post hoc play): take the ridge f*(sigma) of an interference map (probe_map.py; the best
factor per sigma, followed continuously from f = 1 at sigma = 0) and score the trained readout at that factor on a
fresh test namespace, so the choice of f* does not see the cells it is scored on. Compared with f = 1 (no
compensation) on the same cells."""
import json
import sys

import torch
import torch.nn.functional as F

LAB, ckpt, mapfile, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
n = int(sys.argv[5]) if len(sys.argv) > 5 else 1024
sys.path.insert(0, f"{LAB}/phs-soliton")
from phs import tasks  # noqa: E402
from phs.core import inv_softplus  # noqa: E402
from phs.models import build  # noqa: E402
from phs.train import to_torch  # noqa: E402

torch.set_num_threads(1)
mp = json.load(open(mapfile))
c = torch.load(ckpt)
spec = tasks.TASKS["A"]
m = build(c["kind"], spec["vocab"], spec["classes"], c["cfg"], torch.Generator().manual_seed(0))
m.load_state_dict(c["state"])
m.eval()
w = getattr(m.core, {"pin": "w_pin", "lat": "w_lat", "kappa": "w_kappa"}[mp["which"]])
g0 = float(F.softplus(w))
F_, A = mp["factor"], mp["acc"]                                   # A[i_factor][j_sigma]


def ridge(window=0.3):
    j_f = F_.index(1.0)
    out = []
    for j, s in enumerate(mp["sigma"]):
        cand = [i for i, f in enumerate(F_) if abs(f - F_[j_f]) <= window + 1e-9]
        j_f = max(cand, key=lambda i: A[i][j])
        out.append(F_[j_f])
    return out


fstar = ridge()
rows = []
for s, f in zip(mp["sigma"], fstar):
    x, tg, sg = to_torch(tasks.test_set("A", c["T"], s, n=n, namespace="comp"))
    accs = {}
    for name, fac in (("f1", 1.0), ("fstar", f)):
        with torch.no_grad():
            w.fill_(inv_softplus(fac * g0))
            lg = m(x, sg, generator=torch.Generator().manual_seed(13))[:, -1]
        accs[name] = float((lg.argmax(1) == tg[:, -1]).double().mean())
    rows.append({"sigma": s, "fstar": f, "acc_f1": accs["f1"], "acc_fstar": accs["fstar"]})
    print(f"{c['kind']} {mp['which']} sigma {s:.2f}  f* {f:.1f}  Acc f=1 {accs['f1']:.3f}  Acc f* {accs['fstar']:.3f}", flush=True)
json.dump({"kind": c["kind"], "which": mp["which"], "n": n, "rows": rows}, open(out, "w"))
