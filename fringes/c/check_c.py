"""Bit-for-bit check of c/intring.c against intring.py: convert a case file (intring.py --case) to the C text
format, run the binary, compare every phase, momentum and logit. Usage: check_c.py ./intring case.json [--bite]
--bite feeds the binary the other phase rounding and requires a difference (a check must be able to fail)."""
import json
import subprocess
import sys

binary, case_path = sys.argv[1], sys.argv[2]
bite = "--bite" in sys.argv
cs = json.load(open(case_path))
e = cs["export"]
out = [e["A"], e["B"], e["F"], e["N"], e["d"], e["S"], e["V"], int(e["mode"] == "homeo"), int(e["lattice"]), e["m_pin"],
       e["EXP_RANGE"], e["EXP_STEP"], (1 - e.get("rnd", 0)) if bite else e.get("rnd", 0), e["h2"], e["c1"], e["c2u"], e["kappa"], e["g_lat"], e["g_pin"], e["ema"], e["ph_mul"]]
out += e["SIN"] + e["EXP"] + e["invm"] + e["sqm"] + e["G"] + [len(e["phi"])] + e["phi"] + e["u"] + e["psi"] + e["W"] + e["b"]
out += [cs["seed"], len(cs["seqs"]), len(cs["seqs"][0]["x"])]
for s in cs["seqs"]:
    out += s["x"] + [s["sigmaQ"]]
txt = " ".join(str(int(v)) for v in out)
res = subprocess.run([binary], input=txt, capture_output=True, text=True, check=True).stdout.strip().splitlines()
bad = 0
for s, line in zip(cs["seqs"], res):
    th, P, lg = ([int(v) for v in part.split()] for part in line.split("|"))
    if th != s["th"] or P != s["P"] or lg != [int(v) for v in s["logits"]]:
        bad += 1
if bite:
    print(f"{cs['name']}: bite {'ok' if bad else 'FAILED'}: {bad}/{len(cs['seqs'])} sequences differ with the other rounding")
    sys.exit(0 if bad else 1)
print(f"{cs['name']}: {len(cs['seqs']) - bad}/{len(cs['seqs'])} sequences bit-identical in C (A={e['A']}, B={e['B']})")
sys.exit(1 if bad else 0)
