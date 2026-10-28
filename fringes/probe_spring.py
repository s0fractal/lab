"""Which spring sets the clock (post hoc play): scale g_pin, g_lat or kappa of a trained Soliton-PHS at evaluation and
locate the first accuracy minimum sigma_1 of the trained readout. If the precession is a thermal softening of
that spring, sigma_1 moves with its scale while Acc(0) stays high."""
import json
import math
import sys

import torch

LAB, ckpt, which, factor = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
out = sys.argv[5] if len(sys.argv) > 5 else None
sys.path.insert(0, f"{LAB}/phs-soliton")
from phs import tasks  # noqa: E402
from phs.core import inv_softplus  # noqa: E402
from phs.models import build  # noqa: E402
from phs.train import to_torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.set_num_threads(1)
c = torch.load(ckpt)
spec = tasks.TASKS["A"]
m = build(c["kind"], spec["vocab"], spec["classes"], c["cfg"], torch.Generator().manual_seed(0))
m.load_state_dict(c["state"])
m.eval()
w = getattr(m.core, {"pin": "w_pin", "lat": "w_lat", "kappa": "w_kappa"}[which])
with torch.no_grad():
    w.fill_(inv_softplus(factor * float(F.softplus(w))))
accs = []
for s in [round(0.05 * i, 2) for i in range(25)]:
    b = tasks.test_set("A", c["T"], s, n=768, namespace="play")
    x, tg, sg = to_torch(b)
    with torch.no_grad():
        lg = m(x, sg, generator=torch.Generator().manual_seed(7))[:, -1]
    accs.append((s, float((lg.argmax(1) == tg[:, -1]).double().mean())))
i0 = next((i for i, (_, a) in enumerate(accs) if a < 1 / 7), None)
s1 = None
if i0 is not None:
    j = i0
    while j + 1 < len(accs) and accs[j + 1][1] <= accs[j][1]:
        j += 1
    s1 = accs[j][0]
rec = {"kind": c["kind"], "step": c["step"], "which": which, "factor": factor, "acc0": accs[0][1], "sigma_1": s1,
       "min_acc": min(a for _, a in accs), "curve": accs}
print(json.dumps(rec), flush=True)
if out:
    json.dump(rec, open(out, "w"))
