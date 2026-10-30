"""The frozen analysis on synthetic manifests: sigma_crit, the exact tests, the three verdict rules, the gate, the
decoder choice, the revival and admission. Run from phs-clock/: ../phs-soliton/.venv/bin/python -m unittest -v"""
from __future__ import annotations

import copy
import itertools
import json
import shutil
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import analyze  # noqa: E402

GRID = [round(0.05 * i, 2) for i in range(41)]
CAL = [round(0.05 * i, 2) for i in range(25)]
FRINGES = analyze.frozen_fringes((ROOT.parent / "fringes" / "fringe_scan.py").read_bytes())
PREREG = json.loads((ROOT / "prereg.json").read_text())


def step(sc: float, grid=GRID, after=0.5) -> list:
    """Accuracy 1 up to sc, `after` beyond: sigma_crit = sc."""
    return [1.0 if s <= sc + 1e-9 else after for s in grid]


def fringe_curve(sc: float, s1: float, s2: float) -> list:
    """1 up to sc, down to 0.01 at s1, up to 0.40 at s2, down to 0.15 at the grid end: one visible revival."""
    out = []
    for s in GRID:
        if s <= sc + 1e-9:
            out.append(1.0)
        elif s <= s1 + 1e-9:
            out.append(1.0 - 0.99 * (s - sc) / (s1 - sc))
        elif s <= s2 + 1e-9:
            out.append(0.01 + 0.39 * (s - s1) / (s2 - s1))
        else:
            out.append(0.40 - 0.25 * (s - s2) / (2.0 - s2))
    return out


def manifest(kind, T, seed, native, honest=None, play=("same",), acc0=None):
    """A manifest with a native curve and honest rows; honest is {decoder: curve}, play {decoder: curve on CAL}."""
    native = list(native)
    if acc0 is not None:
        native[0] = acc0
    decs = ("fixed", "pooled", "sensor", "oracle") if kind in analyze.PHS else ("fixed", "pooled", "oracle")
    honest = honest or {}
    rows = [{"sigma": s, **{d: honest.get(d, native)[i] for d in decs}} for i, s in enumerate(GRID)]
    m = {"kind": kind, "task": "A", "T": T, "seed": seed, "status": "OK",
         "cells": {analyze.key(s): {"acc": a} for s, a in zip(GRID, native)},
         "honest": {"decoders": list(decs), "rows": rows}}
    if kind in analyze.PHS:
        p = {"sensor": [r["sensor"] for r in rows][:25], "pooled": [r["pooled"] for r in rows][:25]} \
            if play == ("same",) else play
        m["honest"]["play"] = [{"sigma": s, "sensor": p["sensor"][i], "pooled": p["pooled"][i]}
                               for i, s in enumerate(CAL)]
    return m


def design(per_arm: dict) -> list:
    """per_arm[kind] = function(T, i) -> manifest kwargs; ten seeds at T 32 and 64."""
    out = []
    for kind, f in per_arm.items():
        for T in (32, 64):
            for i, seed in enumerate(range(300, 310)):
                out.append(manifest(kind, T, seed, **f(T, i)))
    return out


def jitter(i: int) -> float:
    return 0.05 * (i % 3 - 1)     # -0.05, 0, +0.05


class SigmaCrit(unittest.TestCase):
    def test_contiguous_from_zero_and_threshold_inclusive(self):
        acc = [1.0, 0.9, 0.8, 0.79, 0.95, 0.99] + [0.1] * 35
        self.assertEqual(analyze.sigma_crit(acc, GRID, 0.8, -0.05), 0.1)

    def test_below_grid(self):
        self.assertEqual(analyze.sigma_crit([0.7] + [1.0] * 40, GRID, 0.8, -0.05), -0.05)


class ExactTests(unittest.TestCase):
    @staticmethod
    def brute(strata):
        """Every within-stratum relabelling, by enumeration: two-sided p of the stratified mean difference."""
        def stat(groups):
            return np.mean([np.mean(a) - np.mean(b) for a, b in groups])
        obs = abs(stat(strata))
        options = []
        for a, b in strata:
            pooled = list(a) + list(b)
            options.append([([pooled[i] for i in idx], [pooled[i] for i in range(len(pooled)) if i not in idx])
                            for idx in itertools.combinations(range(len(pooled)), len(a))])
        hits = total = 0
        for combo in itertools.product(*options):
            total += 1
            hits += abs(stat(combo)) >= obs - 1e-9
        return hits / total

    def test_two_strata_match_enumeration_with_ties(self):
        strata = [([0.5, 0.55, 0.55, 0.6], [0.55, 0.7, 0.75]), ([0.4, 0.45, 0.45], [0.45, 0.6, 0.6, 0.65])]
        p, _, n = analyze.stratified_p(strata)
        self.assertEqual(n, 35 * 35)
        self.assertAlmostEqual(p, self.brute(strata), places=12)
        for theta in (-0.1, -0.05, 0.05):
            self.assertAlmostEqual(analyze.stratified_p(analyze.shifted(strata, theta))[0],
                                   self.brute(analyze.shifted(strata, theta)), places=12)

    def test_one_stratum_full_separation_is_two_sided(self):
        p, stat, n = analyze.stratified_p([([0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8])])
        self.assertEqual((p, n), (2 / 70, 70))
        self.assertAlmostEqual(stat, -0.4)

    def test_sign_test_is_exact_two_sided_and_drops_ties(self):
        x = [1.9, 2.1, 1.8, 1.95, 2.3, 1.7, 2.0, 2.25]
        for m0 in (1.75, 2.0, 2.25):
            above = sum(v > m0 + 1e-9 for v in x)
            below = sum(v < m0 - 1e-9 for v in x)
            n = above + below
            # enumeration of every equally likely sign pattern of the n untied values
            obs = abs(above - below)
            hits = sum(abs(2 * sum(bits) - n) >= obs for bits in itertools.product((0, 1), repeat=n))
            self.assertEqual(analyze.sign_test(x, m0), (hits / 2 ** n, above, below))


class Verdicts(unittest.TestCase):
    def strata(self, delta):
        pure = [0.80 + jitter(i) for i in range(10)]
        return [([v + delta for v in pure], pure), ([v + delta - 0.1 for v in pure], [v - 0.1 for v in pure])]

    def test_h1_and_h2_on_either_side_of_the_margin(self):
        far = self.strata(-0.25)
        self.assertEqual(analyze.margin_test(far, 0.05, 0.05, "above")["verdict"], "CONTRADICTED")
        self.assertEqual(analyze.margin_test(far, 0.05, 0.05, "below")["verdict"], "SUPPORTED")
        level = self.strata(0.0)
        self.assertEqual(analyze.margin_test(level, 0.05, 0.05, "above")["verdict"], "SUPPORTED")
        self.assertEqual(analyze.margin_test(level, 0.05, 0.05, "below")["verdict"], "CONTRADICTED")
        at = self.strata(-0.05)
        self.assertEqual(analyze.margin_test(at, 0.05, 0.05, "above")["verdict"], "INCONCLUSIVE")
        self.assertEqual(analyze.margin_test(at, 0.05, 0.05, "below")["verdict"], "INCONCLUSIVE")

    def test_h3_interval(self):
        near = [1.92 + 0.02 * (i % 5 - 2) for i in range(12)]
        self.assertEqual(analyze.interval_test(near, 1.75, 2.25, 0.05, 6)["verdict"], "SUPPORTED")
        linear = [4.0 + 0.1 * (i % 3 - 1) for i in range(12)]
        self.assertEqual(analyze.interval_test(linear, 1.75, 2.25, 0.05, 6)["verdict"], "CONTRADICTED")
        cubic = [1.55 + 0.02 * (i % 3 - 1) for i in range(12)]
        self.assertEqual(analyze.interval_test(cubic, 1.75, 2.25, 0.05, 6)["verdict"], "CONTRADICTED")
        edge = [1.75 + 0.2 * (i % 2 * 2 - 1) for i in range(12)]
        self.assertEqual(analyze.interval_test(edge, 1.75, 2.25, 0.05, 6)["verdict"], "INCONCLUSIVE")
        # Codex r1 F1: a heavy outlier moves the mean out of the interval, not the median the claim is about
        outliers = [(1.15 / 0.8) ** 2] * 18 + [(1.2 / 0.5) ** 2] * 2
        r = analyze.interval_test(outliers, 1.75, 2.25, 0.05, 6)
        self.assertEqual((r["verdict"], r["inside"]), ("SUPPORTED", 18))
        self.assertGreater(r["mean"], 2.25)
        self.assertEqual(analyze.interval_test(near[:5], 1.75, 2.25, 0.05, 6)["verdict"], "INCONCLUSIVE")
        self.assertEqual(analyze.interval_test(near[:7], 1.75, 2.25, 0.05, 6)["verdict"], "SUPPORTED")
        self.assertEqual(analyze.interval_test(near[:7], 1.75, 2.25, 0.05, 8)["verdict"], "INCONCLUSIVE")


class Revival(unittest.TestCase):
    def test_visible_revival(self):
        r = analyze.revival(GRID, fringe_curve(0.5, 0.8, 1.15), FRINGES, 0.03)
        self.assertEqual((r["sigma1"], r["sigma2"]), (0.8, 1.15))
        self.assertAlmostEqual(r["x"], (1.15 / 0.8) ** 2)

    def test_no_revival(self):
        decay = [max(1 / 7, 1.0 - 0.6 * s) for s in GRID]
        self.assertIsNone(analyze.revival(GRID, decay, FRINGES, 0.03))
        low = [min(a, 1 / 7 + 0.02) if s > 0.8 else a for s, a in zip(GRID, fringe_curve(0.5, 0.8, 1.15))]
        self.assertIsNone(analyze.revival(GRID, low, FRINGES, 0.03))
        edge = fringe_curve(0.5, 1.4, 1.99)[:-1] + [0.45]       # still rising at the grid edge: censored
        self.assertIsNone(analyze.revival(GRID, edge, FRINGES, 0.03))


class Compute(unittest.TestCase):
    def runs(self, sol_honest_delta=0.0, sol_acc0=None, x_s2=1.15):
        def sol(T, i):
            sc = 0.45 - (0.15 if T == 64 else 0) + jitter(i)
            h = 0.80 - (0.15 if T == 64 else 0) + jitter(i) + sol_honest_delta
            return {"native": fringe_curve(sc, 0.8 if T == 32 else 0.55, x_s2 if T == 32 else 0.8),
                    "honest": {"sensor": step(h), "pooled": step(sc), "oracle": step(h + 0.05)},
                    "acc0": sol_acc0(T, i) if sol_acc0 else None}

        def pure(T, i):
            sc = 0.80 - (0.15 if T == 64 else 0) + jitter(i)
            return {"native": [max(1 / 7, a) for a in step(sc, after=0.3)],
                    "honest": {"sensor": step(sc - 0.05), "pooled": step(sc), "oracle": step(sc + 0.05)}}

        def gru(T, i):
            sc = 0.40 + jitter(i)
            return {"native": step(sc), "honest": {"pooled": step(sc + 0.05)}}
        return design({"soliton": sol, "pure": pure, "gru": gru})

    def test_kept_memory(self):
        out = analyze.compute(PREREG, self.runs(), FRINGES)
        self.assertEqual(out["H1"]["verdict"], "SUPPORTED")
        self.assertEqual(out["H2"]["verdict"], "SUPPORTED")
        self.assertEqual((out["H3"]["n"], out["H3"]["verdict"]), (20, "SUPPORTED"))
        self.assertEqual(out["H3"]["visible"]["pure"], {"32": "0/10", "64": "0/10"})
        self.assertEqual(out["line"], "CONTINUE")
        self.assertEqual(out["choices"]["pure"], {"sensor": 0, "pooled": 20})
        self.assertEqual(out["choices"]["soliton"], {"sensor": 20, "pooled": 0})

    def test_lost_memory_closes_the_line(self):
        out = analyze.compute(PREREG, self.runs(sol_honest_delta=-0.2), FRINGES)
        self.assertEqual(out["H1"]["verdict"], "CONTRADICTED")
        self.assertEqual(out["line"], "CLOSE")

    def test_gate_excludes_runs_and_strata(self):
        fail = lambda T, i: 0.9 if (T == 64 and i < 7) or (T == 32 and i == 0) else None  # noqa: E731
        out = analyze.compute(PREREG, self.runs(sol_acc0=fail), FRINGES)
        self.assertEqual(out["gate"]["soliton"]["64"]["passed"], 3)
        self.assertEqual(out["gate"]["soliton"]["32"]["passed"], 9)
        self.assertFalse(out["H1"]["by_T"]["64"]["enters"])
        self.assertEqual(len(out["H1"]["by_T"]["32"]["soliton"]), 9)
        self.assertEqual(out["H3"]["n"], 12)                      # gated Soliton runs only: 9 at T 32, 3 at T 64

    def test_unreadable_decoder_makes_h1_inconclusive(self):
        """Codex r1 F3: both honest decoders read nothing, so both arms tie at -0.05; parity is no readable memory."""
        ms = self.runs()
        for m in ms:
            for r in m["honest"]["rows"]:
                r.update({k: 0.5 for k in m["honest"]["decoders"] if k != "fixed"})
            for r in m["honest"].get("play", []):
                r.update(sensor=0.5, pooled=0.5)
        out = analyze.compute(PREREG, ms, FRINGES)
        self.assertFalse(out["H1"]["adequacy"]["holds"])
        self.assertEqual((out["H1"]["verdict"], out["line"]), ("INCONCLUSIVE", "CLOSE"))

    def test_interval_stays_in_its_bounds(self):
        self.assertEqual(analyze.interval([([2.0] * 4, [0.0] * 4)], 0.05, 0.005), [1.0, 1.0])

    def test_failed_reason_is_kept(self):
        m = {"kind": "pure", "T": 32, "seed": 300, "status": "FAILED", "reason": "training: RuntimeError: x"}
        self.assertEqual(analyze.measures(m, PREREG, FRINGES).get("reason"), "training: RuntimeError: x")

    def test_decoder_choice_tie_takes_sensor(self):
        m = manifest("pure", 32, 300, step(0.8), {"sensor": step(0.7), "pooled": step(0.8)},
                     play={"sensor": step(0.9, CAL), "pooled": step(0.9, CAL)})
        r = analyze.measures(m, PREREG, FRINGES)
        self.assertEqual((r["choice"], r["honest"]), ("sensor", 0.7))
        m["honest"]["play"][-7]["sensor"] = 0.5               # play sigma_crit of sensor drops below pooled
        self.assertEqual(analyze.measures(m, PREREG, FRINGES)["choice"], "pooled")


class Admission(unittest.TestCase):
    def setUp(self):
        self.tmp = ROOT / "runs" / "test-admit"
        self.manifests = Compute().runs()
        d = PREREG["design"]
        for m in self.manifests:
            m.update(prereg_sha256="P", tag_commit="C", cfg=PREREG["models"][m["kind"]]["config"],
                     environment=dict(PREREG["environment"]), cell_namespace="test")
            for s in d["sigma"]:
                m["cells"][analyze.key(s)].update(digest=PREREG["test_cells"]["digests"][f"A/{m['T']}/{analyze.key(s)}"],
                                                  n=d["n_test"])
            if m["kind"] in analyze.PHS:
                m["honest"]["play_check"] = "exact"
        self.orig = analyze.ROOT
        analyze.ROOT = self.tmp
        shutil.rmtree(self.tmp, ignore_errors=True)
        for sub in ("checkpoints", "attempts"):
            (self.tmp / "results" / sub).mkdir(parents=True)
        for m in self.manifests:
            stem = f"A-{m['kind']}-T{m['T']:03d}-s{m['seed']}"
            p = self.tmp / "results" / "checkpoints" / f"{stem}.pt"
            p.write_bytes(b"x")
            m.update(checkpoint_sha256=analyze.sha256(p), attempt=f"{stem}.0.json", interrupted_attempts=[],
                     state_sha256="S", interrupted_state_sha256=None)
            (self.tmp / "results" / "attempts" / f"{stem}.0.json").write_text("{}")

    def tearDown(self):
        analyze.ROOT = self.orig

    def named(self, ms):
        return [(f"A-{m['kind']}-T{m['T']:03d}-s{m['seed']}.json", m) for m in ms]

    def test_complete_set_is_admitted(self):
        self.assertEqual(analyze.admit(PREREG, "P", "C", self.named(self.manifests)), [])

    def test_refusals(self):
        cases = []
        ms = copy.deepcopy(self.manifests)
        cases.append(("missing", ms[1:]))
        ms = copy.deepcopy(self.manifests)
        ms[0]["prereg_sha256"] = "other"
        cases.append(("prereg", ms))
        ms = copy.deepcopy(self.manifests)
        ms[0]["cells"]["0.50"]["digest"] = "0" * 64
        cases.append(("digest", ms))
        ms = copy.deepcopy(self.manifests)
        ms[0]["honest"]["play_check"] = "skipped"
        cases.append(("play", ms))
        ms = copy.deepcopy(self.manifests)
        ms[0]["checkpoint_sha256"] = "0" * 64
        cases.append(("checkpoint", ms))
        ms = copy.deepcopy(self.manifests)
        ms[0]["cfg"] = dict(ms[0]["cfg"], steps=4000)
        cases.append(("config", ms))
        ms = copy.deepcopy(self.manifests)
        cases.append(("duplicate", ms + [copy.deepcopy(ms[0])]))
        for label, bad in (("infinite", float("inf")), ("above one", 1.5), ("string", "0.9"), ("bool", True)):
            ms = copy.deepcopy(self.manifests)
            ms[0]["cells"]["0.00"]["acc"] = bad
            cases.append((f"native {label}", ms))
            ms = copy.deepcopy(self.manifests)
            ms[0]["honest"]["rows"][3]["sensor"] = bad
            cases.append((f"honest {label}", ms))
        ms = copy.deepcopy(self.manifests)
        ms[0]["honest"]["decoders"] = ["fixed", "pooled", "sensor"]
        cases.append(("decoder list", ms))
        ms = copy.deepcopy(self.manifests)
        del ms[0]["honest"]["rows"][5]["pooled"]
        cases.append(("missing field", ms))
        ms = copy.deepcopy(self.manifests)
        ms[0] = {k: v for k, v in ms[0].items() if k in ("kind", "task", "T", "seed", "cfg", "environment",
                                                           "prereg_sha256", "tag_commit", "attempt",
                                                           "interrupted_attempts")}
        ms[0]["status"] = "FAILED"
        cases.append(("failed without reason", ms))
        ms = copy.deepcopy(self.manifests)
        ms[0]["interrupted_attempts"] = [ms[0]["attempt"].replace(".0.", ".1.")]
        cases.append(("unrecorded attempt", ms))
        ms = copy.deepcopy(self.manifests)
        ms[0]["interrupted_state_sha256"] = "other"
        cases.append(("redo differs", ms))
        for label, ms in cases:
            with self.subTest(label):
                self.assertNotEqual(analyze.admit(PREREG, "P", "C", self.named(ms)), [])


if __name__ == "__main__":
    unittest.main()
