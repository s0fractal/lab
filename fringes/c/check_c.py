"""Bit-for-bit check of c/intring.c against intring.py: convert a case file (intring.py --case) to the C text
format, run the binary, compare every phase, momentum and logit.
Usage: check_c.py ./intring case.json [--bite|--selftest]
--bite feeds the binary the other phase rounding and requires a difference (a check must be able to fail).
--selftest runs the real binary once, then feeds the comparison doctored transcripts (empty, truncated, one line too
many, a malformed line, a changed value, a missing field) and requires every one to be refused: a checker that only
compares the lines it happens to get accepts `/usr/bin/true` (Codex review F1, primitives PATTERNS.md)."""
import json
import subprocess
import sys


def to_text(cs, rnd):
    e = cs["export"]
    out = [e["A"], e["B"], e["F"], e["N"], e["d"], e["S"], e["V"], int(e["mode"] == "homeo"), int(e["lattice"]), e["m_pin"],
           e["EXP_RANGE"], e["EXP_STEP"], rnd, e["h2"], e["c1"], e["c2u"], e["kappa"], e["g_lat"], e["g_pin"], e["ema"], e["ph_mul"]]
    out += e["SIN"] + e["EXP"] + e["invm"] + e["sqm"] + e["G"] + [len(e["phi"])] + e["phi"] + e["u"] + e["psi"] + e["W"] + e["b"]
    out += [cs["seed"], len(cs["seqs"]), len(cs["seqs"][0]["x"])]
    for s in cs["seqs"]:
        out += s["x"] + [s["sigmaQ"]]
    return " ".join(str(int(v)) for v in out)


def compare(cs, stdout):
    """(number of differing sequences, None) for a complete, well-formed transcript; (None, reason) otherwise."""
    e = cs["export"]
    Nd, V = e["N"] * e["d"], e["V"]
    lines = stdout.splitlines()
    if len(lines) != len(cs["seqs"]):
        return None, f"{len(lines)} output lines for {len(cs['seqs'])} sequences"
    bad = 0
    for k, (s, line) in enumerate(zip(cs["seqs"], lines)):
        parts = line.split("|")
        if len(parts) != 3:
            return None, f"line {k}: {len(parts)} fields, expected th|P|logits"
        try:
            th, P, lg = ([int(v) for v in part.split()] for part in parts)
        except ValueError:
            return None, f"line {k}: not integers"
        if (len(th), len(P), len(lg)) != (Nd, Nd, V):
            return None, f"line {k}: shape {len(th)},{len(P)},{len(lg)} expected {Nd},{Nd},{V}"
        if th != s["th"] or P != s["P"] or lg != [int(v) for v in s["logits"]]:
            bad += 1
    return bad, None


def run(binary, cs, rnd):
    return subprocess.run([binary], input=to_text(cs, rnd), capture_output=True, text=True, check=True).stdout


def selftest(cs, stdout):
    lines = stdout.splitlines()
    first = lines[0].split("|")
    changed = " ".join(str(int(v) + 1) if i == 0 else v for i, v in enumerate(first[0].split()))
    doctored = {
        "empty": "",
        "truncated": "\n".join(lines[:-1]),
        "extra line": stdout + lines[-1] + "\n",
        "malformed": "\n".join(["garbage"] + lines[1:]),
        "missing field": "\n".join(["|".join(first[:2])] + lines[1:]),
        "short logits": "\n".join(["|".join(first[:2] + [" ".join(first[2].split()[:-1])])] + lines[1:]),
        "changed value": "\n".join(["|".join([changed] + first[1:])] + lines[1:]),
    }
    failed = []
    for name, text in doctored.items():
        bad, reason = compare(cs, text)
        if reason is None and not bad:
            failed.append(name)
    return failed


def main():
    binary, case_path = sys.argv[1], sys.argv[2]
    cs = json.load(open(case_path))
    e = cs["export"]
    if not cs.get("seqs"):
        print(f"{cs.get('name', case_path)}: REFUSED: empty corpus")
        sys.exit(1)
    rnd = e.get("rnd", 0)
    if "--bite" in sys.argv:
        bad, reason = compare(cs, run(binary, cs, 1 - rnd))
        if reason:
            print(f"{cs['name']}: bite REFUSED: {reason}")
            sys.exit(1)
        print(f"{cs['name']}: bite {'ok' if bad else 'FAILED'}: {bad}/{len(cs['seqs'])} sequences differ with the other rounding")
        sys.exit(0 if bad else 1)
    stdout = run(binary, cs, rnd)
    bad, reason = compare(cs, stdout)
    if reason:
        print(f"{cs['name']}: REFUSED: {reason}")
        sys.exit(1)
    if "--selftest" in sys.argv:
        if bad:
            print(f"{cs['name']}: selftest needs a passing binary, {bad} sequences differ")
            sys.exit(1)
        failed = selftest(cs, stdout)
        print(f"{cs['name']}: selftest {'FAILED, accepted: ' + ', '.join(failed) if failed else 'ok'}: 7 doctored transcripts")
        sys.exit(1 if failed else 0)
    print(f"{cs['name']}: {len(cs['seqs']) - bad}/{len(cs['seqs'])} sequences bit-identical in C (A={e['A']}, B={e['B']})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
