"""Can unbiased rounding replace the healing noise? (post hoc play; the prediction is in README before the runs)
At 6-7 bits of state the integer ring stalls: a phase step below 1/2 LSB rounds to zero and the rotors freeze, and
heat (sigma 0.1-0.3) brought Pure from 0.05 back to 0.88. Is that the noise itself, or only that noise unbiases the
rounding? Same frozen export, same sequences, same physical-noise draws for every mode; only the phase rounding
changes (stochastic rounding takes its offsets from a separate stream).

  floor       rnd 0, the -1/2 LSB seam of #6
  nearest     rnd 1, the default: unbiased per step, but every step below 1/2 LSB is lost
  stochastic  rnd 2, a uniform sub-LSB offset: unbiased in expectation, adds its own noise of up to 1 LSB
  feedback    rnd 3, each rotor carries its sub-LSB remainder (error feedback): unbiased over time, deterministic
  nearest+p   rnd 1 with products and sums also rounded to nearest (prnd): are the floors in P and forces a bias?
  dither:W    rnd 2 with a dither of width W LSB around 1/2 (W = 1 is stochastic, W = 0 is nearest)

Per sigma: accuracy, and over every phase step the mean signed rounding error (LSB), its RMS, the share of steps that
should move but round to zero (frozen) and the share below 1 LSB; per rotor, the rounding error accumulated over the
whole sequence (RMS over rotors, and the part that differs between the nodes of one ring).
Usage: rounding.py LAB EXPORT A B MODE OUT [n]"""
import json
import sys

import numpy as np

from intring import IntRing

LAB, exp_path, A, B, mode, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
n = int(sys.argv[7]) if len(sys.argv) > 7 else 512
sys.path.insert(0, f"{LAB}/phs-soliton")
from phs import tasks  # noqa: E402

MODES = {"floor": (0, 0), "nearest": (1, 0), "stochastic": (2, 0), "feedback": (3, 0), "nearest+p": (1, 1)}
rnd, prnd = MODES[mode] if not mode.startswith("dither:") else (2, 0)
ring = IntRing(json.load(open(exp_path)), A=A, B=B, rnd=rnd, prnd=prnd)
if mode.startswith("dither:"):
    ring.sr_w = int(round(float(mode.split(":")[1]) * (1 << 20)))
SIG = [round(0.1 * i, 1) for i in range(13)]
rows = []
for s in SIG:
    b = tasks.test_set("A", 32, s, n=n, namespace="comp")
    sq = np.round(b["sigma"] * (1 << ring.F)).astype(np.int64)
    ring.stats = {}
    lg, th, P, _ = ring.run(b["x"], sq, seed=3)
    st = ring.stats
    acc = float((lg.argmax(1) == b["targets"][:, -1]).mean())
    rows.append({"sigma": s, "acc": acc, "bias_lsb": st["err"] / st["n"], "rms_lsb": (st["err2"] / st["n"]) ** 0.5,
                 "frozen": st["frozen"] / st["n"], "sub_lsb": st["sub"] / st["n"], "cum_rms_lsb": st["cum_rms"],
                 "cum_rel_rms_lsb": st["cum_rel_rms"]})
    r = rows[-1]
    print(f"{mode:10s} A={A} B={B} sigma {s:.1f}  acc {acc:.3f}  bias {r['bias_lsb']:+.3f}  rms {r['rms_lsb']:.3f}  "
          f"frozen {r['frozen']:.3f}  sub-LSB {r['sub_lsb']:.3f}  cum {r['cum_rms_lsb']:.2f} (rel {r['cum_rel_rms_lsb']:.2f})",
          flush=True)
json.dump({"export": exp_path, "A": A, "B": B, "mode": mode, "n": n, "rows": rows}, open(out, "w"), indent=1)
