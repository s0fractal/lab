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


def interval(strata: list, alpha: float, step: float) -> list:
    """Descriptive: [lowest, highest] of the shifts theta on a grid of `step`, contiguous around the observed Delta,
    that the two-sided test does not reject at alpha."""
    centre = round(stratified_p(strata)[1] / step) * step
    ends = []
    for direction in (-1, 1):
        t = centre
        while abs(t - centre) < 2 and stratified_p(shifted(strata, t + direction * step))[0] >= alpha:
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


def signflip_p(x: list, mu0: float) -> float:
    """Two-sided exact sign-flip test of H0: the x are symmetric about mu0; every one of the 2^n sign vectors."""
    y = np.asarray(x, float) - mu0
    n = len(y)
    c = abs(y.sum()) - EPS
    if c <= 0:
        return 1.0
    count, bits = 0, np.arange(n)
    for start in range(0, 2 ** n, 1 << 16):
        rows = np.arange(start, min(start + (1 << 16), 2 ** n))
        signs = ((rows[:, None] >> bits) & 1) * 2 - 1
        count += int(np.count_nonzero(np.abs(signs @ y) >= c))
    return count / 2 ** n


def interval_test(x: list, lo: float, hi: float, alpha: float, min_runs: int) -> dict:
    """H3: SUPPORTED when the mean of x lies significantly above lo and significantly below hi (two-sided exact
    sign-flip tests at each end, p < alpha); CONTRADICTED when it lies significantly outside the interval on one
    side; INCONCLUSIVE otherwise, or with fewer than min_runs values."""
    out = {"n": len(x), "x": list(x)}
    if len(x) < min_runs:
        return {**out, "verdict": "INCONCLUSIVE", "why": f"{len(x)} runs with a visible revival, fewer than {min_runs}"}
    mean = float(np.mean(x))
    p_lo, p_hi = signflip_p(x, lo), signflip_p(x, hi)
    if p_lo < alpha and p_hi < alpha and lo < mean < hi:
        verdict = "SUPPORTED"
    elif (p_lo < alpha and mean < lo) or (p_hi < alpha and mean > hi):
        verdict = "CONTRADICTED"
    else:
        verdict = "INCONCLUSIVE"
    return {**out, "mean": mean, "p_at_lo": p_lo, "p_at_hi": p_hi, "verdict": verdict}


def admit(prereg: dict, prereg_sha: str, tag_commit: str, named: list) -> list:
    """Every reason the manifests (file name, manifest) cannot stand for the runs of this PREREG; [] admits them. Each
    run of the design needs exactly one manifest, under its own name, from the tag, with the registered configuration,
    environment and test cells, and (if OK) a checkpoint with the recorded SHA-256 and complete decoder rows."""
    d, env = prereg["design"], prereg["environment"]
    expected = {(k, T, s) for k in d["models"] for T in d["T"] for s in d["seeds"]}
    cells = {key(s) for s in d["sigma"]}
    cal = [round(0.05 * i, 2) for i in range(25)]
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
        if m.get("status") == "FAILED":
            continue
        if m.get("status") != "OK":
            problems.append(f"{fname}: status {m.get('status')!r}")
            continue
        got = m.get("cells", {})
        if m.get("cell_namespace") != "test" or set(got) != cells:
            problems.append(f"{fname}: native cells are not the grid of namespace 'test'")
            continue
        for c in sorted(cells):
            if got[c].get("digest") != prereg["test_cells"]["digests"][f"A/{ident[1]}/{c}"] \
                    or got[c].get("n") != d["n_test"]:
                problems.append(f"{fname}: native cell {c} is not the registered one")
        ckpt = ROOT / "results" / "checkpoints" / f"{stem}.pt"
        if not ckpt.exists() or sha256(ckpt) != m.get("checkpoint_sha256"):
            problems.append(f"{fname}: checkpoint missing or not the one the run saved")
        h = m.get("honest", {})
        want = ("fixed", "pooled", "sensor", "oracle") if ident[0] in PHS else ("fixed", "pooled", "oracle")
        rows = h.get("rows", [])
        if [r.get("sigma") for r in rows] != d["sigma"] or any(not all(k in r for k in want) for r in rows):
            problems.append(f"{fname}: honest rows are not the grid with {want}")
        if ident[0] in PHS and (h.get("play_check") != "exact" or [r.get("sigma") for r in h.get("play", [])] != cal
                                or any(not all(k in r for k in ("pooled", "sensor")) for r in h.get("play", []))):
            problems.append(f"{fname}: readout.py's play rows missing or not reproduced exactly")
    missing = sorted(expected - set(seen))
    problems += [f"no manifest for run A-{k}-T{T:03d}-s{s}" for k, T, s in missing]
    return problems


def measures(m: dict, prereg: dict, fringes) -> dict:
    """One run: the gate, native and honest sigma_crit (and per decoder), the decoder choice, the revival."""
    rule, grid = prereg["rule"], prereg["design"]["sigma"]
    thr, below = rule["acc_threshold"], rule["sigma_crit_below_grid"]
    out = {"kind": m["kind"], "T": m["T"], "seed": m["seed"], "status": m["status"]}
    if m["status"] != "OK":
        return {**out, "gate": False}
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


def compute(prereg: dict, manifests: list, fringes) -> dict:
    rule, d = prereg["rule"], prereg["design"]
    runs = [measures(m, prereg, fringes) for m in sorted(manifests, key=lambda m: (m["kind"], m["T"], m["seed"]))]
    gate = {k: {str(T): {"passed": sum(r["gate"] for r in runs if r["kind"] == k and r["T"] == T),
                         "runs": sum(1 for r in runs if r["kind"] == k and r["T"] == T),
                         "failed": sum(r["status"] != "OK" for r in runs if r["kind"] == k and r["T"] == T),
                         "acc0": [r.get("acc0") for r in runs if r["kind"] == k and r["T"] == T]}
                for T in d["T"]} for k in d["models"]}
    h1 = compare(runs, "soliton", "pure", "honest", prereg, "above")
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
    print(f"H3 {h3['verdict']}: n {h3['n']}, mean x {h3.get('mean', float('nan')):.3f}, p at ends "
          f"{h3.get('p_at_lo', float('nan')):.4g} / {h3.get('p_at_hi', float('nan')):.4g}")
    print(f"gate: " + "; ".join(f"{k} " + ", ".join(f"T{T} {g['passed']}/{g['runs']}" for T, g in v.items())
                                for k, v in out["gate"].items()))
    print(f"line: {out['line']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
