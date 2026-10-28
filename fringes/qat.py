"""Digital memory (post hoc play): train a PHS model with its phase held on a grid of 2^A points from the start, with
a straight-through gradient, instead of quantising a float model afterwards (intring.py did that and found Soliton
alive at 8 bits, Pure dead, both dead below 8).

Every wrap() in the dynamics (the two A-steps per sub-step, the drive phases) returns round(theta * 2^A / 2 pi) on the
circle; the backward pass sees the identity. Momenta stay float (Q16 in the integer ring is far finer than the phase).
After training, the model is evaluated (a) in this quantised torch forward and (b) in the integer ring at A = B, which
has no float at all, through export_js.py's format.

Prediction (written before the runs): trained this way, Soliton-PHS learns task A at A = 6 and A = 7 bits
(Acc >= 0.9 at sigma 0 in the integer ring at the same A), where the post hoc ring gave 0.16 and 0.10; Pure-PHS at
A = 7 stays far below (Acc < 0.5), because its memory needs a fine continuous state.
Usage: qat.py LAB kind A seed steps out_dir [threads]"""
import json
import math
import sys
import time

import torch

LAB, kind, A, seed, steps, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
torch.set_num_threads(int(sys.argv[7]) if len(sys.argv) > 7 else 1)
sys.path.insert(0, f"{LAB}/phs-soliton")
import phs.core as core  # noqa: E402
import phs.models as models  # noqa: E402
from phs import tasks  # noqa: E402
from phs.train import loss_and_hits, to_torch  # noqa: E402
from tools.confirm import model_config  # noqa: E402

TAU, STEP = 2 * math.pi, 2 * math.pi / (1 << A)
_wrap = core.wrap


def qwrap(x):
    y = _wrap(x)
    q = _wrap(torch.round(y / STEP) * STEP)
    return y + (q - y).detach()           # straight-through: forward on the grid, identity backward


core.wrap = qwrap
models.wrap = qwrap

p = json.load(open(f"{LAB}/phs-soliton/prereg.json"))
cfg = model_config(p, kind, "A")
spec = tasks.TASKS["A"]
tag = f"qat-{kind}-A{A}-s{seed}"
model = models.build(kind, spec["vocab"], spec["classes"], cfg,
                     torch.Generator().manual_seed(tasks.derive_seed("init", kind, "A", 32, seed)))
noise = torch.Generator().manual_seed(tasks.derive_seed("noise", kind, "A", 32, seed))
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
torch.save({"kind": kind, "qat_bits": A, "T": 32, "seed": seed, "step": step, "cfg": cfg, "state": model.state_dict()},
           f"{out}/{tag}-step{step:05d}.pt")

model.eval()
SIG = [round(0.1 * i, 1) for i in range(13)]
n = 512
bs = [to_torch(tasks.test_set("A", 32, s, n=n, namespace="comp")) for s in SIG]
X, Y, S = torch.cat([b[0] for b in bs]), torch.cat([b[1][:, -1] for b in bs]), torch.cat([b[2] for b in bs])
with torch.no_grad():
    lg = model(X, S, generator=torch.Generator().manual_seed(13))[:, -1]
acc = (lg.argmax(1) == Y).double().reshape(len(SIG), n).mean(1)
print("torch-quantised  " + " ".join(f"{s:.1f}:{a:.2f}" for s, a in zip(SIG, acc.tolist())), flush=True)
json.dump({"tag": tag, "kind": kind, "A": A, "seed": seed, "physics": model.learned_physics(),
           "acc_torch_q": acc.tolist(), "sigma": SIG}, open(f"{out}/{tag}.json", "w"))
