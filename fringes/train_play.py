"""Short play training of one phs-soliton model (task A) with checkpoints, for the fringe probe. Not a
confirmatory run: fewer steps than PREREG v1, seed outside 0-4, PyPI torch build."""
import json
import sys
import time

import torch

LAB = sys.argv[1]
sys.path.insert(0, f"{LAB}/phs-soliton")
from tools.confirm import model_config  # noqa: E402
from phs import tasks  # noqa: E402
from phs.models import build  # noqa: E402
from phs.train import loss_and_hits, to_torch  # noqa: E402

kind, T, seed, steps, out = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
torch.set_num_threads(int(sys.argv[7]) if len(sys.argv) > 7 else 2)
p = json.load(open(f"{LAB}/phs-soliton/prereg.json"))
cfg = model_config(p, kind, "A")
spec = tasks.TASKS["A"]
model = build(kind, spec["vocab"], spec["classes"], cfg, torch.Generator().manual_seed(tasks.derive_seed("init", kind, "A", T, seed)))
noise = torch.Generator().manual_seed(tasks.derive_seed("noise", kind, "A", T, seed))
opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
stream = tasks.train_stream("A", T, seed)
t0 = time.time()
for step in range(1, steps + 1):
    x, tg, sg = to_torch(tasks.batch("A", stream, cfg["batch"], T))
    loss, hits = loss_and_hits("A", model(x, sg, generator=noise), tg)
    if not torch.isfinite(loss):
        print(f"diverged at {step}", flush=True)
        break
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
    opt.step()
    if step % 100 == 0:
        print(f"{kind} T={T} step {step} loss {float(loss):.4f} acc {float(hits.double().mean()):.3f} "
              f"{time.time() - t0:.0f}s {json.dumps(model.learned_physics())}", flush=True)
    if step % 1000 == 0:
        torch.save({"kind": kind, "T": T, "seed": seed, "step": step, "cfg": cfg, "state": model.state_dict()},
                   f"{out}/{kind}-T{T:03d}-s{seed}-step{step:05d}.pt")
