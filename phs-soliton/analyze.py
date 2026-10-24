#!/usr/bin/env python3
"""S6: the frozen analysis. Reads prereg.json, results/gate.json and results/manifests/*.json and writes
results/results.json with the per-seed scalars, the hypothesis flags and the verdict. Its SHA-256 is pinned in
prereg.json; it refuses to run if the file or prereg.json differ from what the manifests and PREREG name.

    .venv/bin/python analyze.py            # from phs-soliton/

Only numpy and the standard library; no model code is imported.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
EPS = 1e-9


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sigma_crit(acc: dict, grid: list, threshold: float, below: float) -> float:
    """max{sigma_g : Acc(sigma') >= threshold for every sigma' <= sigma_g}; `below` if Acc(grid[0]) < threshold."""
    best = below
    for s in grid:
        if acc[f"{s:.2f}"] >= threshold:
            best = s
        else:
            break
    return best


def slope(xs: list, ys: list) -> float:
    """Ordinary least-squares slope of ys on xs."""
    x, y = np.asarray(xs, float), np.asarray(ys, float)
    return float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())


def permutation_p(x: list, y: list) -> tuple[float, int]:
    """One-sided exact permutation test of mean(x) - mean(y) > 0: every split of the pooled values into groups of
    len(x) and len(y) (252 for 5 vs 5); p = share of splits whose difference is >= the observed one."""
    pooled = np.asarray(list(x) + list(y), float)
    n = len(x)
    observed = float(np.mean(x) - np.mean(y))
    count = total = 0
    for idx in itertools.combinations(range(len(pooled)), n):
        mask = np.zeros(len(pooled), bool)
        mask[list(idx)] = True
        d = pooled[mask].mean() - pooled[~mask].mean()
        count += bool(d >= observed - EPS)
        total += 1
    return float(count / total), total


def holm(ps: dict, alpha: float) -> dict:
    """Holm's step-down rejections within one family {name: p}."""
    order = sorted(ps, key=lambda k: (ps[k], k))
    out, still = {}, True
    for i, k in enumerate(order):
        still = still and ps[k] <= alpha / (len(order) - i)
        out[k] = bool(still)
    return out


def spearman(a: list, b: list) -> float:
    def ranks(v):
        v = np.asarray(v, float)
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v))
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            r[order[i:j + 1]] = (i + j) / 2 + 1
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def expected_config(prereg: dict, kind: str, task: str) -> dict:
    cfg = dict(prereg["optimisation"])
    cfg.update(prereg["models"][kind]["config"])
    if kind == "gru_pm":
        cfg["pm_target"] = prereg["models"]["soliton"]["params"][task]
    return cfg


def run_name(kind: str, task: str, T: int, seed: int) -> str:
    return f"{task}-{kind}-T{T:03d}-s{seed}"


def admit(prereg: dict, prereg_sha: str, named: list, gate: dict | None, tag_commit: str) -> list:
    """Every reason the manifests (file name, manifest) cannot stand for runs of this PREREG; [] admits them.
    A manifest must be one expected run, once, under its own name, from tag_commit (the commit the PREREG tag
    resolves to, as must the gate), with the registered configuration, environment and test cells (namespace,
    digests, sizes)."""
    g, env = prereg["grid"], prereg["environment"]
    expected = {(k, t, T, s) for k in g["models"] for t in ("A", "B") for T in g["T"] for s in g["seeds"]}
    cells = {f"{s:.2f}" for s in g["sigma"]}
    problems, seen = [], {}
    for fname, m in named:
        try:
            ident = (m["kind"], m["task"], m["T"], m["seed"])
        except (KeyError, TypeError):
            problems.append(f"{fname}: no run identity")
            continue
        if ident not in expected:
            problems.append(f"{fname}: {ident} is not a run of the grid")
            continue
        if fname != run_name(*ident) + ".json":
            problems.append(f"{fname}: names run {run_name(*ident)}")
        if ident in seen:
            problems.append(f"{fname}: duplicate of {seen[ident]}")
        seen[ident] = fname
        if m.get("prereg_sha256") != prereg_sha:
            problems.append(f"{fname}: another PREREG")
        if m.get("tag_commit") != tag_commit:
            problems.append(f"{fname}: ran at {m.get('tag_commit')!r}, not at the tag {tag_commit}")
        if m.get("status") not in ("OK", "FAILED"):
            problems.append(f"{fname}: status {m.get('status')!r}")
        if m.get("cfg") != expected_config(prereg, ident[0], ident[1]):
            problems.append(f"{fname}: configuration differs from PREREG")
        for key in ("python", "torch", "numpy"):
            if m.get("environment", {}).get(key) != env[key]:
                problems.append(f"{fname}: {key} {m.get('environment', {}).get(key)} is not {env[key]}")
        if m.get("status") == "OK":
            if m.get("cell_namespace") != "test":
                problems.append(f"{fname}: cells from namespace {m.get('cell_namespace')!r}")
            got = m.get("cells", {})
            if set(got) != cells:
                problems.append(f"{fname}: cells {sorted(set(got) ^ cells)[:3]} differ from the grid")
            for key in cells & set(got):
                c = got[key]
                if c.get("digest") != prereg["tasks"]["digests"].get(f"{ident[1]}/{ident[2]}/{key}"):
                    problems.append(f"{fname}: cell {key} digest differs from PREREG")
                if c.get("n") != prereg["tasks"]["n_test"]:
                    problems.append(f"{fname}: cell {key} has n = {c.get('n')}")
    if gate is not None and gate.get("tag_commit") != tag_commit:
        problems.append(f"the gate ran at {gate.get('tag_commit')!r}, not at the tag {tag_commit}")
    return problems


def compute(prereg: dict, prereg_sha: str, manifest_list: list, gate: dict | None, registered_mutants: int) -> dict:
    """The whole analysis as a function of admitted inputs (tests/test_analyze.py drives it with synthetic
    manifests). registered_mutants is the number of mutants in mutants.json at the analysed tree."""
    rule, grid = prereg["verdict_rule"], prereg["grid"]
    sigmas, Ts, seeds = grid["sigma"], grid["T"], grid["seeds"]
    kinds, verdict_kinds = grid["models"], rule["verdict_models"]
    manifests = {(m["kind"], m["task"], m["T"], m["seed"]): m for m in manifest_list}

    expected = [(k, t, T, s) for k in kinds for t in ("A", "B") for T in Ts for s in seeds]
    missing = [list(e) for e in expected if e not in manifests]
    failed = [list(e) for e in expected if e in manifests and manifests[e]["status"] != "OK"]
    incomplete_verdict = [e for e in missing + failed if e[0] in verdict_kinds]

    def acc(k, task, T, s, sig):
        m = manifests.get((k, task, T, s))
        if m is None or m["status"] != "OK":
            return None
        return m["cells"][f"{sig:.2f}"]["acc"]

    per_seed = {}
    for k in kinds:
        per_seed[k] = {}
        for s in seeds:
            row = {}
            ok_a = all(acc(k, "A", T, s, sig) is not None for T in Ts for sig in sigmas)
            ok_b = all(acc(k, "B", T, s, 0.0) is not None for T in Ts)
            if ok_a:
                sc = {str(T): sigma_crit({f"{sig:.2f}": acc(k, "A", T, s, sig) for sig in sigmas}, sigmas,
                                         rule["acc_threshold"], rule["sigma_crit_below_grid"]) for T in Ts}
                row["sigma_crit_T"] = sc
                row["sigma_crit_mean"] = float(np.mean(list(sc.values())))
                # secondary: the same threshold read on the PDF's grid only (sigma <= train grid top)
                top = max(grid.get("train_sigma", sigmas))
                row["sigma_crit_mean_pdf_grid"] = float(np.mean([min(v, top) for v in sc.values()]))
                row["lambda"] = slope([np.log2(T) for T in Ts], [acc(k, "A", T, s, rule["sigma_ref"]) for T in Ts])
                row["acc_A_T16_sigma0"] = acc(k, "A", Ts[0], s, 0.0)
                row["acc_A_T16_sigma_ref"] = acc(k, "A", Ts[0], s, rule["sigma_ref"])
            if ok_b:
                row["B_mean_acc_sigma0"] = float(np.mean([acc(k, "B", T, s, 0.0) for T in Ts]))
            per_seed[k][str(s)] = row

    def column(k, key):
        vals = [per_seed[k][str(s)].get(key) for s in seeds]
        return None if any(v is None for v in vals) else vals

    fl = rule["lambda_floor"]
    sol_ref = [per_seed["soliton"][str(s)].get("acc_A_T16_sigma_ref") for s in seeds]
    floor_ok = sum(1 for a in sol_ref if a is not None and a >= fl["acc"] - EPS) >= fl["min_seeds"]
    hyp = {}
    families = {"A": ("pure", {"A1": "sigma_crit_mean", "A2": "lambda"}),
                "G": ("gru", {"G1": "sigma_crit_mean", "G2": "lambda"})}
    min_effect = {"sigma_crit_mean": rule["min_effect"]["sigma_crit_mean"], "lambda": rule["min_effect"]["lambda"]}
    for fam, (other, members) in families.items():
        ps = {}
        for name, key in members.items():
            x, y = column("soliton", key), column(other, key)
            if x is None or y is None:
                hyp[name] = {"metric": key, "versus": other, "available": False, "holds": False}
                continue
            p, n = permutation_p(x, y)
            ps[name] = p
            hyp[name] = {"metric": key, "versus": other, "available": True, "soliton": x, "other": y,
                         "delta": float(np.mean(x) - np.mean(y)), "p": p, "splits": n,
                         "min_effect": min_effect[key]}
        rej = holm(ps, rule["alpha"]) if ps else {}
        for name in members:
            if hyp[name]["available"]:
                hyp[name]["holm_reject"] = rej[name]
                hyp[name]["holds"] = bool(rej[name] and hyp[name]["delta"] >= hyp[name]["min_effect"] - EPS)
                if members[name] == "lambda":   # a flat slope at the chance floor is no slower decay
                    hyp[name]["floor_ok"] = floor_ok
                    hyp[name]["holds"] = hyp[name]["holds"] and floor_ok

    b_sol, b_pure = column("soliton", "B_mean_acc_sigma0"), column("pure", "B_mean_acc_sigma0")
    B = {"available": b_sol is not None and b_pure is not None, "margin": rule["B_margin"]}
    if B["available"]:
        B.update(soliton=float(np.mean(b_sol)), pure=float(np.mean(b_pure)))
        B["delta"] = B["soliton"] - B["pure"]
        B["holds"] = bool(B["delta"] >= -rule["B_margin"] - EPS)
    else:
        B["holds"] = False

    v_models = {}
    for k in verdict_kinds:
        a16 = [per_seed[k][str(s)].get("acc_A_T16_sigma0") for s in seeds]
        good = sum(1 for a in a16 if a is not None and a >= rule["V_acc"] - EPS)
        v_models[k] = {"acc_A_T16_sigma0": a16, "seeds_ok": good, "holds": good >= rule["V_min_seeds"]}
    gate_ok = bool(gate and gate.get("tests_ok") is True and gate.get("prereg_check_ok") is True
                   and gate.get("mutants_exit") == 0 and gate.get("prereg_sha256") == prereg_sha
                   and registered_mutants > 0
                   and gate.get("mutants_killed") == gate.get("mutants_total") == registered_mutants)
    V = {"gate": gate, "gate_ok": gate_ok, "models": v_models,
         "holds": gate_ok and all(v["holds"] for v in v_models.values())}

    pure_sc, pure_lam = column("pure", "sigma_crit_mean"), column("pure", "lambda")
    room = {"available": pure_sc is not None and pure_lam is not None}
    if room["available"]:
        room.update(pure_sigma_crit_mean=float(np.mean(pure_sc)), pure_lambda=float(np.mean(pure_lam)),
                    sigma_max=rule["room"]["pure_sigma_crit_max"], lambda_max=rule["room"]["pure_lambda_max"])
        # REFUTED needs one testable conjunct: room on sigma_crit or room on lambda
        room["sigma"] = bool(room["pure_sigma_crit_mean"] <= rule["room"]["pure_sigma_crit_max"] + EPS)
        room["lambda"] = bool(room["pure_lambda"] <= rule["room"]["pure_lambda_max"] + EPS)
        room["holds"] = room["sigma"] or room["lambda"]
    else:
        room["holds"] = False

    h = {k: v["holds"] for k, v in hyp.items()}
    A12 = h["A1"] and h["A2"]
    small = (hyp["A1"].get("available") and hyp["A2"].get("available")
             and hyp["A1"]["delta"] < rule["min_effect"]["sigma_crit_mean"] - EPS
             and hyp["A2"]["delta"] < rule["min_effect"]["lambda"] - EPS)
    if incomplete_verdict:
        verdict, why = "INCONCLUSIVE", f"{len(incomplete_verdict)} runs of the verdict models missing or FAILED"
    elif not V["holds"]:
        verdict, why = "INCONCLUSIVE", "validity V does not hold"
    elif A12 and h["G1"] and h["G2"] and B["holds"]:
        verdict, why = "CONFIRMED", "A1, A2, G1, G2 and B hold"
    elif A12 and B["holds"]:
        verdict, why = "CONFIRMED_VS_PURE_ONLY", "A1, A2 and B hold; G1 and G2 not both"
    elif A12:
        verdict, why = "TRADEOFF", "A1 and A2 hold; B does not"
    elif small and room["holds"]:
        verdict, why = "REFUTED", "A1 and A2 not both; both Soliton - Pure differences below the minimum effect"
    elif small:
        verdict, why = "INCONCLUSIVE", "both differences below the minimum effect, but Pure-PHS leaves no room (ceiling)"
    else:
        verdict, why = "INCONCLUSIVE", "neither confirmation nor refutation; a new PREREG with more seeds is the remedy"

    cells, secondary = {}, {}
    for k in kinds:
        cells[k] = {}
        for task in ("A", "B"):
            cells[k][task] = {}
            for T in Ts:
                cells[k][task][str(T)] = {}
                for sig in sigmas:
                    ms = [manifests.get((k, task, T, s)) for s in seeds]
                    ms = [m for m in ms if m is not None and m["status"] == "OK"]
                    if not ms:
                        continue
                    c = [m["cells"][f"{sig:.2f}"] for m in ms]
                    entry = {"n_seeds": len(c), "acc_mean": float(np.mean([x["acc"] for x in c])),
                             "acc_by_seed": [x["acc"] for x in c]}
                    if k in ("pure", "soliton"):   # absent or None where a cell had no finite sequence
                        rs = [x["r_slip"] for x in c if x.get("r_slip") is not None]
                        er = [x["erank"] for x in c if x.get("erank") is not None]
                        entry["r_slip_mean"] = float(np.mean(rs)) if rs else None
                        entry["erank_mean"] = float(np.mean(er)) if er else None
                        entry["nonfinite"] = sum(int(x.get("nonfinite", 0)) for x in c)
                    cells[k][task][str(T)][f"{sig:.2f}"] = entry
    sol_a = [(e["r_slip_mean"], e["acc_mean"]) for T in Ts for e in cells["soliton"]["A"][str(T)].values()
             if e.get("r_slip_mean") is not None]
    secondary["spearman_rslip_acc_soliton_A"] = spearman([a for a, _ in sol_a], [b for _, b in sol_a]) \
        if len(sol_a) >= 3 else None
    secondary["models"] = {}
    for k in kinds:
        ms = [m for m in manifests.values() if m["kind"] == k and m["status"] == "OK"]
        info = {"params": sorted({(m["task"], m["params"]) for m in ms}),
                "c_step_ms_median": float(np.median([m["c_step_ms"] for m in ms])) if ms else None,
                "flops_per_token": sorted({(m["task"], m["flops_per_token"]) for m in ms})}
        if k in ("pure", "soliton") and ms:
            for key in ("kappa", "g_lat", "g_pin", "mass_min", "mass_max", "J_norm"):
                info[f"{key}_median"] = float(np.median([m["physics"][key] for m in ms]))
            info["gamma"] = sorted({m["physics"]["gamma"] for m in ms})
        if k == "soliton" and ms:
            info["lattice_unused_rate"] = float(np.mean([bool(m.get("lattice_unused")) for m in ms]))
        secondary["models"][k] = info
    if "gru_pm" in kinds:
        for key in ("sigma_crit_mean", "lambda"):
            x, y = column("soliton", key), column("gru_pm", key)
            if x is not None and y is not None:
                p, n = permutation_p(x, y)
                secondary[f"soliton_vs_gru_pm_{key}"] = {"delta": float(np.mean(x) - np.mean(y)), "p": p}

    return {"prereg_sha256": prereg_sha, "analyze_sha256": prereg["frozen"].get("analyze.py"),
            "runs_expected": len(expected), "runs_present": len(expected) - len(missing), "missing": missing,
            "failed": failed, "per_seed": per_seed, "hypotheses": hyp, "B": B, "V": V, "room": room,
            "verdict": verdict, "verdict_reason": why, "cells": cells, "secondary": secondary}


def main() -> int:
    prereg_path = ROOT / "prereg.json"
    prereg = json.loads(prereg_path.read_text())
    if sha256(Path(__file__).resolve()) != prereg["frozen"]["analyze.py"]:
        print("refused: analyze.py differs from the SHA-256 pinned in prereg.json")
        return 2
    prereg_sha = sha256(prereg_path)
    try:
        tag_commit = subprocess.run(["git", "rev-parse", "--verify", f"refs/tags/{prereg['tag']}^{{commit}}"],
                                    cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
        rel = subprocess.run(["git", "rev-parse", "--show-prefix"], cwd=ROOT, check=True, capture_output=True,
                             text=True).stdout.strip()
        tagged = subprocess.run(["git", "show", f"{tag_commit}:{rel}prereg.json"], cwd=ROOT, check=True,
                                capture_output=True).stdout
    except subprocess.CalledProcessError:
        print(f"refused: tag {prereg['tag']} or its prereg.json cannot be resolved")
        return 2
    if tagged != prereg_path.read_bytes():
        print(f"refused: prereg.json differs from the one at tag {prereg['tag']}")
        return 2
    gate_path = ROOT / "results" / "gate.json"
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else None
    named = [(p.name, json.loads(p.read_text())) for p in sorted((ROOT / "results" / "manifests").glob("*.json"))]
    problems = admit(prereg, prereg_sha, named, gate, tag_commit)
    if problems:
        print(f"refused: {len(problems)} admission problems")
        for x in problems[:20]:
            print("  " + x)
        return 2
    registered = len(json.loads((ROOT / "mutants.json").read_text())["mutants"])
    out = compute(prereg, prereg_sha, [m for _, m in named], gate, registered)
    (ROOT / "results" / "results.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    verdict, why, hyp, B, room = out["verdict"], out["verdict_reason"], out["hypotheses"], out["B"], out["room"]
    v_models, gate_ok = out["V"]["models"], out["V"]["gate_ok"]
    print(f"verdict {verdict}: {why}")
    for name, v in hyp.items():
        if v.get("available"):
            print(f"  {name}: delta {v['delta']:+.4f} (min {v['min_effect']}), p {v['p']:.4f}, "
                  f"Holm {v['holm_reject']}, holds {v['holds']}")
    print(f"  B: {B}")
    print(f"  V: gate {gate_ok}, " + ", ".join(f"{k} {v['seeds_ok']}/5" for k, v in v_models.items()))
    print(f"  room: {room}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
