"""Synthetic integer-ring cases for CI: no checkpoint, no torch, only numpy. Usage: synth_cases.py OUTDIR

A small random export (N=4, d=2, V=7, negative bias: a trained readout has one, and b << F was undefined in C99) is
run through intring.py over the grid plain/homeo x lattice off/on x (A, B) = (8, 8), (24, 16) x rnd 0/1, and each case
is written with the Python states and logits for c/check_c.py and web/check_intring.js. The vectors are synthetic,
not the trained models: they pin the arithmetic contract, not any accuracy.
Two self-checks run first and must bite: the 2^53 range check refuses an export built to overflow it, and two
different seeds give different transcripts (a generator that ignored its input would pass every engine check)."""
import json
import os
import sys

import numpy as np

from intring import IntRing, write_case


def export(seed=42, N=4, d=2, V=7, u=(0.2, 1.0), mass=1.0):
    rs = np.random.default_rng(seed)
    return dict(N=N, d=d, S=4, mode="plain", h=0.05, gamma=0.3, mass=[mass] * N, G=np.eye(N).tolist(), kappa=1.0,
                g_lat=0.5, g_pin=0.3, m_pin=2, psi=[0.0] * N, phi=rs.uniform(-3, 3, (9, N, d)).tolist(),
                u=rs.uniform(*u, (9, N, d)).tolist(), W=rs.normal(0, 0.1, (V, 3 * N * d)).tolist(),
                b=(-0.1 - 0.05 * np.arange(V)).tolist())


def main(out):
    os.makedirs(out, exist_ok=True)
    try:   # range bite: a light ring hammered by a strong drive with 30 phase bits
        write_case(IntRing(export(u=(4000, 8000), mass=0.01), A=30, B=16), os.path.join(out, "overflow.json"), "x")
        sys.exit("range check FAILED to bite: an export built to pass 2^53 was accepted")
    except AssertionError as err:
        print(f"range bite ok: {err}")
    names = []
    for mode in ("plain", "homeo"):
        for lattice in (False, True):
            for A, B in ((8, 8), (24, 16)):
                for rnd in (0, 1):
                    e = dict(export(), mode=mode)
                    if not lattice:
                        e.update(g_lat=0.0, g_pin=0.0)
                    name = f"synth-{mode}-{'lat' if lattice else 'free'}-{A}-{B}-r{rnd}"
                    write_case(IntRing(e, A=A, B=B, rnd=rnd), os.path.join(out, name + ".json"), name)
                    names.append(name)
    a = json.load(open(os.path.join(out, names[-1] + ".json")))
    write_case(IntRing(dict(export(seed=7), mode="homeo"), A=24, B=16, rnd=1), os.path.join(out, "seed7.json"), "s7")
    c = json.load(open(os.path.join(out, "seed7.json")))
    os.remove(os.path.join(out, "seed7.json"))
    assert a["seqs"][-1]["th"] != c["seqs"][-1]["th"], "generator does not depend on its export"
    peaks = [json.load(open(os.path.join(out, n + ".json")))["peak_bits"] for n in names]
    print(f"{len(names)} synthetic cases in {out}, peak intermediate <= 2^{max(peaks)}")


if __name__ == "__main__":
    main(sys.argv[1])
