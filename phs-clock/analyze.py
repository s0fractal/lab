#!/usr/bin/env python3
"""The frozen analysis of PREREG phs-clock v1. Reads prereg.json and results/manifests/*.json and writes
results/results.json: every run's measures, the learnability gate, the verdicts of H1, H2 and H3, the GRU control and
the continuation rule. Its SHA-256 is pinned in prereg.json. It refuses to run if this file, prereg.json or
fringes/fringe_scan.py differ from what the PREREG tag names, or if any run lacks an admissible manifest.

    ../phs-soliton/.venv/bin/python analyze.py      # from phs-clock/

Only numpy and the standard library; no model code is imported. A revival is located by fringe_scan.py's own
fringes(), executed from its pinned bytes.
"""
from __future__ import annotations

import ast
import hashlib
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent
EPS = 1e-9
CHANCE = 1 / 7
PHS = ("soliton", "pure")
CAL_TOP = 1.2   # readout.py's calibration grid ends here; an honest sigma_crit at or above it is counted, not trusted


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(s: float) -> str:
    return f"{s:.2f}"


def sigma_crit(acc: list, grid: list, threshold: float, below: float) -> float:
    """max{sigma_g : Acc(sigma') >= threshold for every grid sigma' <= sigma_g}; `below` if Acc(grid[0]) < threshold
    (phs-soliton analyze.py, unchanged in meaning)."""
    best = below
    for s, a in zip(grid, acc):
        if a >= threshold:
            best = s
        else:
            break
    return best


def frozen_fringes(data: bytes):
    """fringe_scan.py's fringes(), executed alone from the given bytes (its module code reads manifests)."""
    tree = ast.parse(data, "fringes/fringe_scan.py")
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "fringes"]
    if len(nodes) != 1:
        raise SystemExit("refused: fringes/fringe_scan.py does not define fringes() once")
    env = {"CHANCE": CHANCE}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "fringes/fringe_scan.py", "exec"), env)
    return env["fringes"]


def revival(grid: list, acc: list, fringes, margin: float) -> dict | None:
    """As fringe_scan.py reads a curve: after Acc first falls under chance, the extrema alternate minimum, peak, ...;
    sigma_1 is the first minimum, sigma_2 the first later peak above chance + margin (a peak at the grid edge is
    censored by fringes()). None without both."""
    ext = fringes(grid, acc) or []
    mins = [e for k, e in enumerate(ext) if k % 2 == 0]
    peaks = [e for k, e in enumerate(ext) if k % 2 == 1 and e[1] > CHANCE + margin]
    if not (mins and peaks):
        return None
    (s1, a1), (s2, a2) = mins[0], peaks[0]
    return {"sigma1": s1, "acc1": a1, "sigma2": s2, "acc2": a2, "x": (s2 / s1) ** 2}


_SPLITS: dict = {}


def splits(n: int, k: int) -> np.ndarray:
    if (n, k) not in _SPLITS:
        _SPLITS[(n, k)] = np.array(list(itertools.combinations(range(n), k)), dtype=np.int64).reshape(-1, k)
    return _SPLITS[(n, k)]


def split_diffs(a: list, b: list) -> np.ndarray:
    """mean(first) - mean(second) for every split of the pooled values into groups of len(a) and len(b)."""
    pooled = np.asarray(list(a) + list(b), float)
    n, k = len(pooled), len(a)
    s = pooled[splits(n, k)].sum(1)
    return s / k - (pooled.sum() - s) / (n - k)


def stratified_p(strata: list) -> tuple[float, float, int]:
    """Two-sided exact permutation test of the stratified difference, the mean over strata of mean(a) - mean(b):
    every relabelling within each stratum (the product of the strata's splits). Returns (p, statistic, splits).
    One or two strata."""
    if not 1 <= len(strata) <= 2:
        raise ValueError("one or two strata")
    stat = float(np.mean([np.mean(a) - np.mean(b) for a, b in strata]))
    c = abs(stat) * len(strata) - EPS          # |sum over strata| >= |observed sum|, with float slack
    d = [split_diffs(a, b) for a, b in strata]
    total = int(np.prod([len(x) for x in d]))
    if c <= 0:
        return 1.0, stat, total
    if len(d) == 1:
        return float(np.count_nonzero(np.abs(d[0]) >= c) / total), stat, total
    other = np.sort(d[1])
    hi = len(other) - np.searchsorted(other, c - d[0], side="left")       # d0 + d1 >= c
    lo = np.searchsorted(other, -c - d[0], side="right")                  # d0 + d1 <= -c
    return float((hi.sum() + lo.sum()) / total), stat, total


def shifted(strata: list, theta: float) -> list:
    """Strata under H0: Delta = theta (the first group moved by -theta)."""
    return [([x - theta for x in a], b) for a, b in strata]


def interval(strata: list, alpha: float, step: float, bound: float = 1.0) -> list:
    """Descriptive: [lowest, highest] of the shifts theta on a grid of `step` in [-bound, bound], contiguous around
    the observed Delta, that the two-sided test does not reject at alpha (an end at +-bound may be cut there)."""
    centre = min(bound, max(-bound, round(stratified_p(strata)[1] / step) * step))
    ends = []
    for direction in (-1, 1):
        t = centre
        while abs(t + direction * step) <= bound + EPS \
                and stratified_p(shifted(strata, t + direction * step))[0] >= alpha:
            t += direction * step
        ends.append(round(t, 6))
    return ends


def margin_test(strata: list, margin: float, alpha: float, supported_side: str) -> dict:
    """H1 (supported_side "above") and H2 ("below") at the minimal effect: the two-sided exact stratified test of
    H0: Delta = -margin. With p < alpha the side of Delta + margin decides: SUPPORTED on the hypothesis's side,
    CONTRADICTED on the other; INCONCLUSIVE otherwise."""
    p_m, s_m, n = stratified_p(shifted(strata, -margin))           # s_m = Delta + margin
    p_0, delta, _ = stratified_p(strata)
    if p_m < alpha:
        side = "above" if s_m > 0 else "below"
        verdict = "SUPPORTED" if side == supported_side else "CONTRADICTED"
    else:
        verdict = "INCONCLUSIVE"
    return {"delta": delta, "p_at_margin": p_m, "p_at_zero": p_0, "splits": n, "verdict": verdict}


def sign_test(x: list, m0: float) -> tuple[float, int, int]:
    """Two-sided exact sign test of H0: the median of x is m0 (values equal to m0 dropped). (p, above, below)."""
    above = sum(1 for v in x if v > m0 + EPS)
    below = sum(1 for v in x if v < m0 - EPS)
    n, k = above + below, min(above, below)
    if n == 0:
        return 1.0, above, below
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n), above, below


def interval_test(x: list, lo: float, hi: float, alpha: float, min_runs: int) -> dict:
    """H3: SUPPORTED when the median of x lies significantly above lo and significantly below hi (two-sided exact
    sign tests at each end, p < alpha, on the side the counts show); CONTRADICTED when it lies significantly outside
    the interval at one end; INCONCLUSIVE otherwise, or with fewer than min_runs values. A sign test needs no
    symmetry and no model of the outliers a misread extremum makes."""
    out = {"n": len(x), "x": list(x)}
    if len(x) < min_runs:
        return {**out, "verdict": "INCONCLUSIVE", "why": f"{len(x)} runs with a visible revival, fewer than {min_runs}"}
    (p_lo, a_lo, b_lo), (p_hi, a_hi, b_hi) = sign_test(x, lo), sign_test(x, hi)
    if p_lo < alpha and a_lo > b_lo and p_hi < alpha and b_hi > a_hi:
        verdict = "SUPPORTED"
    elif (p_lo < alpha and b_lo > a_lo) or (p_hi < alpha and a_hi > b_hi):
        verdict = "CONTRADICTED"
    else:
        verdict = "INCONCLUSIVE"
    return {**out, "median": float(np.median(x)), "mean": float(np.mean(x)),
            "inside": sum(1 for v in x if lo <= v <= hi), "p_at_lo": p_lo, "p_at_hi": p_hi, "verdict": verdict}


def accuracy(v) -> bool:
    """A recorded accuracy: a finite real number in [0, 1] (a bool or a string is not one)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and 0 <= v <= 1


def admit(prereg: dict, prereg_sha: str, tag_commit: str, named: list) -> list:
    """Every reason the manifests (file name, manifest) cannot stand for the runs of this PREREG; [] admits them. Each
    run of the design needs exactly one manifest, under its own name, from the tag, with the registered configuration
    and environment, its attempt records (its own and those of attempts killed before a manifest; any checkpoint those
    left reproduced), a reason if FAILED, and if OK: the registered test cells, a checkpoint with the recorded SHA-256,
    the registered decoders and every accuracy a finite number in [0, 1]."""
    d, env = prereg["design"], prereg["environment"]
    expected = {(k, T, s) for k in d["models"] for T in d["T"] for s in d["seeds"]}
    cells = {key(s) for s in d["sigma"]}
    cal = [round(0.05 * i, 2) for i in range(25)]
    attempts = ROOT / "results" / "attempts"
    problems, seen = [], {}
    for fname, m in named:
        try:
            ident = (m["kind"], m["T"], m["seed"])
        except (KeyError, TypeError):
            problems.append(f"{fname}: no run identity")
            continue
        if ident not in expected or m.get("task") != "A":
            problems.append(f"{fname}: {ident} is not a run of the design")
            continue
        stem = f"A-{ident[0]}-T{ident[1]:03d}-s{ident[2]}"
        if fname != stem + ".json":
            problems.append(f"{fname}: names run {stem}")
        if ident in seen:
            problems.append(f"{fname}: duplicate of {seen[ident]}")
        seen[ident] = fname
        if m.get("prereg_sha256") != prereg_sha or m.get("tag_commit") != tag_commit:
            problems.append(f"{fname}: not a run of this PREREG at its tag")
        if m.get("cfg") != prereg["models"][ident[0]]["config"]:
            problems.append(f"{fname}: configuration differs from PREREG")
        for k in ("python", "torch", "numpy"):
            if m.get("environment", {}).get(k) != env[k]:
                problems.append(f"{fname}: {k} {m.get('environment', {}).get(k)} is not {env[k]}")
        listed = [m.get("attempt")] + list(m.get("interrupted_attempts") or [])
        if sorted(p.name for p in attempts.glob(f"{stem}.*.json")) != sorted(str(a) for a in listed):
            problems.append(f"{fname}: attempt records are not this run's attempt and its interrupted ones")
        if m.get("status") == "FAILED":
            if not str(m.get("reason") or "").strip():
                problems.append(f"{fname}: FAILED without a reason")
            continue
        if m.get("status") != "OK":
            problems.append(f"{fname}: status {m.get('status')!r}")
            continue
        if m.get("interrupted_state_sha256") not in (None, m.get("state_sha256")):
            problems.append(f"{fname}: the weights differ from those an interrupted attempt saved")
        got = m.get("cells", {})
        if m.get("cell_namespace") != "test" or set(got) != cells:
            problems.append(f"{fname}: native cells are not the grid of namespace 'test'")
            continue
        for c in sorted(cells):
            if got[c].get("digest") != prereg["test_cells"]["digests"][f"A/{ident[1]}/{c}"] \
                    or got[c].get("n") != d["n_test"] or not accuracy(got[c].get("acc")):
                problems.append(f"{fname}: native cell {c} is not the registered one or has no accuracy")
        ckpt = ROOT / "results" / "checkpoints" / f"{stem}.pt"
        if not ckpt.exists() or sha256(ckpt) != m.get("checkpoint_sha256"):
            problems.append(f"{fname}: checkpoint missing or not the one the run saved")
        h = m.get("honest", {})
        want = ["fixed", "pooled", "sensor", "oracle"] if ident[0] in PHS else ["fixed", "pooled", "oracle"]
        rows = h.get("rows", [])
        if h.get("decoders") != want or [r.get("sigma") for r in rows] != d["sigma"] \
                or any(not accuracy(r.get(k)) for r in rows for k in want):
            problems.append(f"{fname}: honest rows are not the grid with an accuracy of each of {want}")
        play = h.get("play", [])
        if ident[0] in PHS and (h.get("play_check") != "exact" or [r.get("sigma") for r in play] != cal
                                or any(not accuracy(r.get(k)) for r in play for k in ("pooled", "sensor"))):
            problems.append(f"{fname}: readout.py's play rows missing or not reproduced exactly")
    missing = sorted(expected - set(seen))
    problems += [f"no manifest for run A-{k}-T{T:03d}-s{s}" for k, T, s in missing]
    return problems


def measures(m: dict, prereg: dict, fringes) -> dict:
    """One run: the gate, native and honest sigma_crit (and per decoder), the decoder choice, the revival."""
    rule, grid = prereg["rule"], prereg["design"]["sigma"]
    thr, below = rule["acc_threshold"], rule["sigma_crit_below_grid"]
    out = {"kind": m["kind"], "T": m["T"], "seed": m["seed"], "status": m["status"],
           "interrupted_attempts": len(m.get("interrupted_attempts") or [])}
    if m["status"] != "OK":
        return {**out, "gate": False, "reason": m.get("reason")}
    native = [m["cells"][key(s)]["acc"] for s in grid]
    out.update(acc0=native[0], gate=bool(native[0] >= rule["gate_acc"] - EPS),
               native=sigma_crit(native, grid, thr, below))
    rows = m["honest"]["rows"]
    for dec in m["honest"]["decoders"]:
        out[f"honest_{dec}"] = sigma_crit([r[dec] for r in rows], grid, thr, below)
    if m["kind"] in PHS:
        play = m["honest"]["play"]
        cal = [r["sigma"] for r in play]
        out["play_sensor"] = sigma_crit([r["sensor"] for r in play], cal, thr, below)
        out["play_pooled"] = sigma_crit([r["pooled"] for r in play], cal, thr, below)
        out["choice"] = "sensor" if out["play_sensor"] >= out["play_pooled"] else "pooled"
    else:
        out["choice"] = "pooled"
    out["honest"] = out[f"honest_{out['choice']}"]
    out["revival"] = revival(grid, native, fringes, rule["h3"]["revival_margin"])
    out["nonfinite_z"] = sum(r.get("nonfinite_z", 0) for r in rows)
    return out


def strata_of(runs: list, a: str, b: str, metric: str, Ts: list, min_runs: int) -> tuple[list, dict]:
    """(strata, by T) of gated runs: T enters only with at least min_runs gated runs of each arm."""
    strata, by_T = [], {}
    for T in Ts:
        xa = [r[metric] for r in runs if r["kind"] == a and r["T"] == T and r["gate"]]
        xb = [r[metric] for r in runs if r["kind"] == b and r["T"] == T and r["gate"]]
        enters = len(xa) >= min_runs and len(xb) >= min_runs
        by_T[str(T)] = {a: xa, b: xb, "enters": enters,
                        **({"mean_" + a: float(np.mean(xa)), "mean_" + b: float(np.mean(xb))} if enters else {})}
        if enters:
            strata.append((xa, xb))
    return strata, by_T


def compare(runs: list, a: str, b: str, metric: str, prereg: dict, supported_side: str | None) -> dict:
    rule, Ts = prereg["rule"], prereg["design"]["T"]
    strata, by_T = strata_of(runs, a, b, metric, Ts, rule["min_runs_per_arm_per_T"])
    out = {"a": a, "b": b, "metric": metric, "by_T": by_T}
    if not strata:
        return {**out, "verdict": "INCONCLUSIVE", "why": "no T with enough gated runs of both arms"}
    if supported_side:
        out.update(margin_test(strata, rule["margin"], rule["alpha"], supported_side), margin=rule["margin"])
    else:
        p_0, delta, n = stratified_p(strata)
        out.update(delta=delta, p_at_zero=p_0, splits=n)
    out["interval"] = interval(strata, rule["alpha"], rule["interval_step"])
    return out


def adequacy(runs: list, h1: dict, margin: float) -> dict:
    """H1 can be SUPPORTED only if the honest decoder is a competent reader: in every T entering H1, the mean honest
    sigma_crit of Pure-PHS (no clock) is at least its mean native sigma_crit minus the margin."""
    per_T = {}
    for T, s in h1["by_T"].items():
        if s["enters"]:
            native = float(np.mean([r["native"] for r in runs if r["kind"] == "pure" and r["T"] == int(T) and r["gate"]]))
            per_T[T] = {"pure_honest": s["mean_pure"], "pure_native": native,
                        "holds": bool(s["mean_pure"] >= native - margin - EPS)}
    return {"by_T": per_T, "holds": bool(per_T) and all(v["holds"] for v in per_T.values())}


def compute(prereg: dict, manifests: list, fringes) -> dict:
    rule, d = prereg["rule"], prereg["design"]
    runs = [measures(m, prereg, fringes) for m in sorted(manifests, key=lambda m: (m["kind"], m["T"], m["seed"]))]
    gate = {k: {str(T): {"passed": sum(r["gate"] for r in runs if r["kind"] == k and r["T"] == T),
                         "runs": sum(1 for r in runs if r["kind"] == k and r["T"] == T),
                         "failed": sum(r["status"] != "OK" for r in runs if r["kind"] == k and r["T"] == T),
                         "acc0": [r.get("acc0") for r in runs if r["kind"] == k and r["T"] == T]}
                for T in d["T"]} for k in d["models"]}
    h1 = compare(runs, "soliton", "pure", "honest", prereg, "above")
    h1["adequacy"] = adequacy(runs, h1, rule["margin"])
    if h1["verdict"] == "SUPPORTED" and not h1["adequacy"]["holds"]:
        h1.update(verdict="INCONCLUSIVE", why="the honest decoder does not read Pure-PHS about as well as its own "
                                              "readout (adequacy)")
    h2 = compare(runs, "soliton", "pure", "native", prereg, "below")
    xs = [r["revival"]["x"] for r in runs if r["kind"] == "soliton" and r["gate"] and r["revival"]]
    lo, hi = rule["h3"]["interval"]
    h3 = interval_test(xs, lo, hi, rule["alpha"], rule["h3"]["min_runs"])
    h3["visible"] = {k: {str(T): f"{sum(1 for r in runs if r['kind'] == k and r['T'] == T and r['gate'] and r['revival'])}"
                                  f"/{sum(1 for r in runs if r['kind'] == k and r['T'] == T and r['gate'])}"
                         for T in d["T"]} for k in d["models"]}
    control = {f"{a}_vs_{b}_{metric}": compare(runs, a, b, metric, prereg, None)
               for a, b in (("soliton", "gru"), ("pure", "gru")) for metric in ("honest", "native")}
    sensitivity = {f"h1_{dec}_only": compare(runs, "soliton", "pure", f"honest_{dec}", prereg, "above")
                   for dec in ("sensor", "pooled")}
    sensitivity["h1_oracle"] = compare(runs, "soliton", "pure", "honest_oracle", prereg, None)
    lift = {k: {str(T): [r["honest"] - r["native"] for r in runs if r["kind"] == k and r["T"] == T and r["gate"]]
                for T in d["T"]} for k in d["models"]}
    calibration_edge = sum(1 for r in runs if r["gate"] and r["honest"] >= CAL_TOP - EPS)
    line = "CONTINUE" if h1["verdict"] == "SUPPORTED" else "CLOSE"
    return {"runs": runs, "gate": gate, "H1": h1, "H2": h2, "H3": h3, "control": control, "sensitivity": sensitivity,
            "lift": lift, "honest_at_or_above_calibration_edge": calibration_edge,
            "choices": {k: {c: sum(1 for r in runs if r["kind"] == k and r["gate"] and r.get("choice") == c)
                            for c in ("sensor", "pooled")} for k in d["models"]},
            "nonfinite_z": sum(r.get("nonfinite_z", 0) for r in runs), "line": line}


def main() -> int:
    prereg_path = ROOT / "prereg.json"
    prereg = json.loads(prereg_path.read_text())
    if sha256(Path(__file__).resolve()) != prereg["frozen"]["phs-clock/analyze.py"]:
        print("refused: analyze.py differs from the SHA-256 pinned in prereg.json")
        return 2
    try:
        tag_commit = subprocess.run(["git", "rev-parse", "--verify", f"refs/tags/{prereg['tag']}^{{commit}}"],
                                    cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
        tagged = subprocess.run(["git", "show", f"{tag_commit}:phs-clock/prereg.json"], cwd=ROOT, check=True,
                                capture_output=True).stdout
    except subprocess.CalledProcessError:
        print(f"refused: tag {prereg['tag']} or its prereg.json cannot be resolved")
        return 2
    if tagged != prereg_path.read_bytes():
        print(f"refused: prereg.json differs from the one at tag {prereg['tag']}")
        return 2
    scan = (LAB / "fringes" / "fringe_scan.py").read_bytes()
    if hashlib.sha256(scan).hexdigest() != prereg["frozen"]["fringes/fringe_scan.py"]:
        print("refused: fringes/fringe_scan.py differs from the SHA-256 pinned in prereg.json")
        return 2
    named = [(p.name, json.loads(p.read_text())) for p in sorted((ROOT / "results" / "manifests").glob("*.json"))]
    problems = admit(prereg, sha256(prereg_path), tag_commit, named)
    if problems:
        print(f"refused: {len(problems)} admission problems")
        for x in problems[:20]:
            print("  " + x)
        return 2
    out = compute(prereg, [m for _, m in named], frozen_fringes(scan))
    out.update(prereg_sha256=sha256(prereg_path), analyze_sha256=prereg["frozen"]["phs-clock/analyze.py"],
               tag_commit=tag_commit)
    (ROOT / "results" / "results.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    for h in ("H1", "H2"):
        v = out[h]
        print(f"{h} {v['verdict']}: delta {v.get('delta', float('nan')):+.4f}, p at -margin "
              f"{v.get('p_at_margin', float('nan')):.4g}, p at 0 {v.get('p_at_zero', float('nan')):.4g}, "
              f"interval {v.get('interval')}")
    h3 = out["H3"]
    print(f"H3 {h3['verdict']}: n {h3['n']}, median x {h3.get('median', float('nan')):.3f}, inside "
          f"{h3.get('inside')}, p at ends {h3.get('p_at_lo', float('nan')):.4g} / {h3.get('p_at_hi', float('nan')):.4g}")
    print(f"H1 adequacy: {out['H1'].get('adequacy')}")
    print(f"gate: " + "; ".join(f"{k} " + ", ".join(f"T{T} {g['passed']}/{g['runs']}" for T, g in v.items())
                                for k, v in out["gate"].items()))
    print(f"line: {out['line']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
