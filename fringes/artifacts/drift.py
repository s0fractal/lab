"""How far do last-bit differences move the transfer accuracies? (post hoc; the evidence behind verify.py's bounds)
Every model parameter of Soliton seed 900 is multiplied by (1 + eps N(0,1)) (float32: eps = 1e-7 is about one ulp, as
another CPU rounds differently), then the T = 48 transfer cells are rescored with transfer.py's own definitions and the
frozen decoders. Prints, per sigma and at worst, the change of each accuracy from artifacts/transfer-2026-10-29b in
units of one binomial standard error of 512 (at least 3 sequences): the frozen decoders and the oracle, which is refit
on each cell and so moves more.
Usage: artifacts/drift.py LAB EPS   (from fringes/, with the phs-soliton venv's python -I)"""
import json
import sys

import torch

LAB, eps = sys.argv[1], float(sys.argv[2])
sys.argv = ["transfer.py", LAB, f"{LAB}/fringes/artifacts/pr7-2026-10-28/checkpoints/soliton-T032-s900-step04000.pt",
            "/dev/null", "1"]
ns = {}                                     # transfer.py's definitions, up to where its command-line modes begin
src = open(f"{LAB}/fringes/transfer.py").read().split('if "--cells" in sys.argv:')[0]
exec(compile(src, "transfer.py", "exec"), ns)
full, low, _ = ns["frozen_decoders"]()
frozen = json.load(open(f"{LAB}/fringes/artifacts/transfer-2026-10-29b/results/transfer/soliton-s900.json"))
g = torch.Generator().manual_seed(1)
with torch.no_grad():
    for p in ns["model"].parameters():
        p.mul_(1 + eps * torch.randn(p.shape, generator=g, dtype=p.dtype))
keys = ("fixed", "pooled", "sensor", "sigma_lda", "oracle")
worst = {k: (-1.0, 0.0) for k in keys}
for row in frozen["T48"]:
    s = row["sigma"]
    z, fp, y, D = ns["states"](s, "transfer", 48, 1024)
    sc = ns["score"](full, (z, fp, y, D), s)
    gaps = {k: abs(sc[k] - row[k]) * 512 / max(3.0, (row[k] * (1 - row[k]) * 512) ** 0.5) for k in keys}
    for k in keys:
        worst[k] = max(worst[k], (gaps[k], s))
    print(f"eps {eps:.0e} sigma {s:.2f}: " + ", ".join(f"{k} {gaps[k]:.2f}" for k in keys), flush=True)
print(f"eps {eps:.0e} worst: " + ", ".join(f"{k} {w[0]:.2f} (sigma {w[1]})" for k, w in worst.items()))
