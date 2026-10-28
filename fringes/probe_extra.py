"""Interventions on one checkpoint (post hoc play), per sigma:
  acc_fixed     the trained readout
  acc_reflect   the trained readout on z' = 2 zbar - z (zbar: mean final state over the cell), i.e. the class
                signal with its sign flipped: high where the memory has precessed by pi
  acc_noJ       the trained model with the gyroscopic block J set to 0 at evaluation"""
import json
import sys

import torch

LAB, ckpt, out = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, f"{LAB}/phs-soliton")
from phs import tasks  # noqa: E402
from phs.models import build  # noqa: E402
from phs.train import to_torch  # noqa: E402

torch.set_num_threads(1)
n = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
c = torch.load(ckpt)
spec = tasks.TASKS["A"]


def load(noJ=False):
    m = build(c["kind"], spec["vocab"], spec["classes"], c["cfg"], torch.Generator().manual_seed(0))
    m.load_state_dict(c["state"])
    if noJ:
        with torch.no_grad():
            m.core.w_skew.zero_()
    return m.eval()


models = {"base": load(), "noJ": load(True)}
grab = {}
models["base"].readout.register_forward_hook(lambda mod, inp, o: grab.__setitem__("z", inp[0].detach()))
rows = []
for s in [round(0.1 * i, 1) for i in range(16)]:
    b = tasks.test_set("A", c["T"], s, n=n, namespace="play")
    x, tg, sg = to_torch(b)
    y = tg[:, -1]
    with torch.no_grad():
        lg = models["base"](x, sg, generator=torch.Generator().manual_seed(7))[:, -1]
        z = grab["z"]
        zr = 2 * z.mean(0, keepdim=True) - z
        lr = models["base"].readout(zr)
        ln = models["noJ"](x, sg, generator=torch.Generator().manual_seed(7))[:, -1]
    row = {"sigma": s, "acc_fixed": float((lg.argmax(1) == y).double().mean()),
           "acc_reflect": float((lr.argmax(1) == y).double().mean()),
           "acc_noJ": float((ln.argmax(1) == y).double().mean())}
    rows.append(row)
    print(f"{c['kind']:7s} step {c['step']} sigma {s:.1f}  fixed {row['acc_fixed']:.3f}  reflect {row['acc_reflect']:.3f}"
          f"  noJ {row['acc_noJ']:.3f}", flush=True)
json.dump({"ckpt": ckpt, "rows": rows}, open(out, "w"))
