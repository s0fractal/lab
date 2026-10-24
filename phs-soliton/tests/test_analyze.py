"""The frozen verdict rule (analyze.compute) on synthetic manifests: each verdict of the rule is reachable, and the
guards (completeness, validity, minimum effect, room) change the verdict when they should."""
from __future__ import annotations

import copy
import math
import unittest

import analyze

SIGMAS = [round(0.05 * i, 2) for i in range(8)]
TS = [16, 32, 64, 128]
SEEDS = [0, 1, 2, 3, 4]
KINDS = ["pure", "soliton", "gru", "gru_pm"]

PREREG = {
    "grid": {"models": KINDS, "T": TS, "sigma": SIGMAS, "seeds": SEEDS},
    "frozen": {},
    "verdict_rule": {"verdict_models": ["pure", "soliton", "gru"], "acc_threshold": 0.8, "sigma_crit_below_grid": -0.05,
                     "sigma_ref": 0.10, "alpha": 0.05, "min_effect": {"sigma_crit_mean": 0.05, "lambda": 0.02},
                     "B_margin": 0.05, "V_acc": 0.95, "V_min_seeds": 4,
                     "room": {"pure_sigma_crit_max": 0.30, "pure_lambda_max": -0.02}},
}
GATE = {"tests_ok": True, "mutants_killed": 13, "mutants_total": 13, "prereg_sha256": "x"}


def robust(T, s, seed, edge=0.25):
    """Holds 0.99 up to sigma = edge at every T; no decay."""
    return 0.99 - 0.001 * seed if s <= edge + 1e-9 else 0.5


def fragile(T, s, seed):
    """0.99 at sigma 0; for 0 < sigma <= 0.10 it decays by 0.05 per doubling of T (0.84 at T = 128); 0.5 beyond."""
    if s == 0:
        return 0.99 - 0.001 * seed
    if s <= 0.10 + 1e-9:
        return 0.99 - 0.05 * (math.log2(T) - 4) - 0.001 * seed
    return 0.5


def manifests(a_fn: dict, b_fn: dict | None = None, drop=()):
    b_fn = b_fn or {}
    out = []
    for k in KINDS:
        for task in ("A", "B"):
            for T in TS:
                for seed in SEEDS:
                    if (k, task, T, seed) in drop:
                        continue
                    if task == "A":
                        accs = {f"{s:.2f}": a_fn[k](T, s, seed) for s in SIGMAS}
                    else:
                        accs = {f"{s:.2f}": b_fn.get(k, lambda T, s, seed: 0.9)(T, s, seed) for s in SIGMAS}
                    cells = {key: {"acc": v} for key, v in accs.items()}
                    m = {"kind": k, "task": task, "T": T, "seed": seed, "status": "OK", "cells": cells,
                         "params": 100, "c_step_ms": 1.0, "flops_per_token": 10}
                    if k in ("pure", "soliton"):
                        for c in cells.values():
                            c.update(r_slip=1.0 - c["acc"], erank=2.0)
                        m["physics"] = {"kappa": 1, "g_lat": 1, "g_pin": 1, "mass_min": 1, "mass_max": 1,
                                        "J_norm": 1, "gamma": 0.5}
                        m["lattice_unused"] = False
                    out.append(m)
    return out


def verdict(ms, gate=GATE, prereg=PREREG):
    return analyze.compute(prereg, "x", ms, gate)["verdict"]


BASE = {"pure": fragile, "soliton": robust, "gru": fragile, "gru_pm": fragile}


class TestPrimitives(unittest.TestCase):
    def test_sigma_crit_stops_at_the_first_failure(self):
        acc = {f"{s:.2f}": a for s, a in zip(SIGMAS, [0.9, 0.85, 0.7, 0.9, 0.9, 0.9, 0.9, 0.9])}
        self.assertEqual(analyze.sigma_crit(acc, SIGMAS, 0.8, -0.05), 0.05)
        acc["0.00"] = 0.79
        self.assertEqual(analyze.sigma_crit(acc, SIGMAS, 0.8, -0.05), -0.05)
        self.assertEqual(analyze.sigma_crit({f"{s:.2f}": 0.8 for s in SIGMAS}, SIGMAS, 0.8, -0.05), 0.35)

    def test_slope(self):
        self.assertAlmostEqual(analyze.slope([4, 5, 6, 7], [1.0, 0.9, 0.8, 0.7]), -0.1)

    def test_exact_permutation(self):
        p, n = analyze.permutation_p([5, 6, 7, 8, 9], [0, 1, 2, 3, 4])
        self.assertEqual(n, 252)
        self.assertAlmostEqual(p, 1 / 252)
        p, _ = analyze.permutation_p([1, 1, 1, 1, 1], [1, 1, 1, 1, 1])
        self.assertEqual(p, 1.0)

    def test_holm_step_down(self):
        self.assertEqual(analyze.holm({"a": 0.03, "b": 0.04}, 0.05), {"a": False, "b": False})
        self.assertEqual(analyze.holm({"a": 0.02, "b": 0.04}, 0.05), {"a": True, "b": True})
        self.assertEqual(analyze.holm({"a": 0.02, "b": 0.06}, 0.05), {"a": True, "b": False})


class TestVerdicts(unittest.TestCase):
    def test_confirmed(self):
        self.assertEqual(verdict(manifests(BASE)), "CONFIRMED")

    def test_confirmed_vs_pure_only(self):
        self.assertEqual(verdict(manifests(dict(BASE, gru=robust))), "CONFIRMED_VS_PURE_ONLY")

    def test_tradeoff(self):
        b = {"soliton": lambda T, s, seed: 0.7}
        self.assertEqual(verdict(manifests(BASE, b)), "TRADEOFF")

    def test_refuted(self):
        self.assertEqual(verdict(manifests(dict(BASE, soliton=fragile))), "REFUTED")

    def test_significant_but_below_minimum_effect_is_not_confirmation(self):
        def slightly(T, s, seed):   # sigma_crit 0.15 at T = 16, 32 and 0.10 after: mean +0.025 over fragile
            if 0.10 < s <= 0.15 + 1e-9 and T <= 32:
                return 0.9 - 0.001 * seed
            if 0 < s <= 0.10 + 1e-9:
                return 0.99 - 0.04 * (math.log2(T) - 4) - 0.001 * seed   # lambda +0.01 over fragile
            return fragile(T, s, seed)
        out = analyze.compute(PREREG, "x", manifests(dict(BASE, soliton=slightly)), GATE)
        self.assertTrue(out["hypotheses"]["A1"]["holm_reject"])
        self.assertEqual(out["verdict"], "REFUTED")

    def test_ceiling_is_inconclusive_not_refuted(self):
        flat = {k: (lambda T, s, seed: 0.99) for k in KINDS}
        self.assertEqual(verdict(manifests(flat)), "INCONCLUSIVE")

    def test_missing_verdict_run_is_inconclusive(self):
        ms = manifests(dict(BASE, gru=robust), drop={("gru", "A", 64, 3)})
        self.assertEqual(verdict(ms), "INCONCLUSIVE")

    def test_validity_needs_learning_at_T16(self):
        def weak(T, s, seed):
            return 0.9 if (s == 0 and T == 16 and seed < 2) else fragile(T, s, seed)
        self.assertEqual(verdict(manifests(dict(BASE, pure=weak))), "INCONCLUSIVE")

    def test_validity_needs_the_gate(self):
        self.assertEqual(verdict(manifests(BASE), gate=dict(GATE, mutants_killed=12)), "INCONCLUSIVE")
        self.assertEqual(verdict(manifests(BASE), gate=None), "INCONCLUSIVE")

    def test_results_serialize_as_json(self):
        import json
        out = analyze.compute(PREREG, "x", manifests(BASE), GATE)
        self.assertEqual(json.loads(json.dumps(out))["verdict"], "CONFIRMED")

    def test_gru_pm_is_not_in_the_verdict(self):
        ms = manifests(dict(BASE, gru_pm=robust), drop={("gru_pm", "A", 16, 0)})
        self.assertEqual(verdict(ms), "CONFIRMED")


if __name__ == "__main__":
    unittest.main()
