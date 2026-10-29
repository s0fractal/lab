"""How far do last-bit differences move the replayed numbers? (post hoc; the evidence behind verify.py's bounds)
Every model parameter of Soliton seed 900 is multiplied by (1 + eps N(0,1)) (float32: eps = 1e-7 is about one ulp, as
another CPU rounds differently), then cells are recomputed with the producer's own definitions:
  default      the T = 48 transfer cells, rescored with the frozen decoders. Prints, per sigma and at worst, the change
               of each accuracy from artifacts/transfer-2026-10-29b in units of one binomial standard error of 512 (at
               least 3 sequences): the frozen decoders and the oracle, which is refit on each cell and so moves more;
               and of the cell mean of the sensor D, in its own standard errors (the spread of D over the cell's 1024
               sequences) and relative
  --decouple   decouple.py's n = 0 and n = 63 at T = 128 and sigma 0, 0.6: the change of the model's own readout and of
               the same-sigma LDA (refit) in standard errors of 1024 (at least 3), and of the overlap r and the
               relative norm, from artifacts/decouple-2026-10-29
  --kinds      kinds.py's J63 and P63 at sigma 0, 0.6, the same numbers, from artifacts/kinds-2026-10-29
Usage: artifacts/drift.py LAB EPS [--decouple | --kinds]   (from fringes/, with the phs-soliton venv's python -I)"""
import json
import sys

import torch

LAB, eps = sys.argv[1], float(sys.argv[2])
if "--decouple" in sys.argv:
    sys.argv = ["decouple.py", LAB, f"{LAB}/fringes/artifacts/pr7-2026-10-28/checkpoints/soliton-T032-s900-step04000.pt",
                "/dev/null", "1"]
    ns = {}                                 # decouple.py's definitions, up to where its command-line modes begin
    src = open(f"{LAB}/fringes/decouple.py").read().split('if "--selftest" in sys.argv:')[0]
    exec(compile(src, "decouple.py", "exec"), ns)
    frozen = json.load(open(f"{LAB}/fringes/artifacts/decouple-2026-10-29/results/decouple/soliton-s900.json"))
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in ns["model"].parameters():
            p.mul_(1 + eps * torch.randn(p.shape, generator=g, dtype=p.dtype))
    for n in (0, 63):
        ref = None
        for s in (0.0, 0.6):
            zc, _, _, yc = ns["cell"](n, s, "decouple-cal")
            z, _, pred, y = ns["cell"](n, s, "decouple")
            sig = ns["class_signal"](z, y)
            ref = sig if s == 0.0 else ref
            w, j = frozen[f"dd{n}"], frozen["sigma"].index(s)
            aware = ns["lda_predict"](ns["lda_fit"](zc, yc), z)
            got = {"own": float((pred == y).mean()), "aware": float((aware == y).mean()), "r": ns["cos"](sig, ref),
                   "norm": float(sig @ sig) ** 0.5 / float(ref @ ref) ** 0.5}
            se = {k: abs(got[k] - w[k][j]) * 1024 / max(3.0, (w[k][j] * (1 - w[k][j]) * 1024) ** 0.5)
                  for k in ("own", "aware")}
            print(f"eps {eps:.0e} dd{n} sigma {s:.1f}: own {se['own']:.2f} SE, aware LDA {se['aware']:.2f} SE, "
                  f"r {abs(got['r'] - w['r'][j]):.4f}, norm {abs(got['norm'] - w['norm'][j]):.4f}", flush=True)
    sys.exit(0)
if "--kinds" in sys.argv:
    sys.argv = ["kinds.py", LAB, f"{LAB}/fringes/artifacts/pr7-2026-10-28/checkpoints/soliton-T032-s900-step04000.pt",
                "/dev/null", "1"]
    ns = {}                                 # kinds.py's definitions, up to where its command-line modes begin
    src = open(f"{LAB}/fringes/kinds.py").read().split('if "--scales" in sys.argv:')[0]
    exec(compile(src, "kinds.py", "exec"), ns)
    frozen = json.load(open(f"{LAB}/fringes/artifacts/kinds-2026-10-29/results/kinds/soliton-s900.json"))
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in ns["model"].parameters():
            p.mul_(1 + eps * torch.randn(p.shape, generator=g, dtype=p.dtype))
    for name in ("J63", "P63"):
        ref = None
        for s in (0.0, 0.6):
            zc, _, _, yc = ns["cell"](name, s, "decouple-cal")
            z, _, pred, y = ns["cell"](name, s, "decouple")
            sig = ns["class_signal"](z, y)
            ref = sig if s == 0.0 else ref
            w, j = frozen[name], frozen["sigma"].index(s)
            aware = ns["lda_predict"](ns["lda_fit"](zc, yc), z)
            got = {"own": float((pred == y).mean()), "aware": float((aware == y).mean()), "r": ns["cos"](sig, ref),
                   "norm": float(sig @ sig) ** 0.5 / float(ref @ ref) ** 0.5}
            se = {k: abs(got[k] - w[k][j]) * 1024 / max(3.0, (w[k][j] * (1 - w[k][j]) * 1024) ** 0.5)
                  for k in ("own", "aware")}
            print(f"eps {eps:.0e} {name} sigma {s:.1f}: own {se['own']:.2f} SE, aware LDA {se['aware']:.2f} SE, "
                  f"r {abs(got['r'] - w['r'][j]):.4f}, norm {abs(got['norm'] - w['norm'][j]):.4f}", flush=True)
    sys.exit(0)
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
    gaps["D"] = abs(sc["D"] - row["D"]) / (D.std() / len(D) ** 0.5)  # the cell mean of D, in its own standard errors
    gaps["D_rel"] = abs(sc["D"] - row["D"]) / abs(row["D"])
    for k in keys + ("D", "D_rel"):
        worst[k] = max(worst.get(k, (-1.0, 0.0)), (gaps[k], s))
    print(f"eps {eps:.0e} sigma {s:.2f}: " + ", ".join(f"{k} {gaps[k]:.2f}" for k in keys)
          + f", D {gaps['D']:.2f} SE ({gaps['D_rel']:.1e} relative)", flush=True)
print(f"eps {eps:.0e} worst: " + ", ".join(f"{k} {w[0]:.2g} (sigma {w[1]})" for k, w in worst.items()))
