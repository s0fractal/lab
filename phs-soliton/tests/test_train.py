"""Evaluation policy for non-finite outputs (phs/train.py)."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from phs import tasks
from phs.train import eligible, evaluate


class HalfNaN(torch.nn.Module):
    """Logits that point at class 0 for every sequence; NaN for the first half of each batch."""

    family = "gru"

    def forward(self, x, sigma, generator=None, diagnostics=False):
        B, T = x.shape
        logits = torch.full((B, T, tasks.TASKS["A"]["classes"]), -1.0)
        logits[..., 0] = 1.0
        logits[: B // 2] = float("nan")
        return logits, {}


class TestNonFinite(unittest.TestCase):
    def test_nonfinite_sequences_count_as_wrong(self):
        b = tasks.batch("A", np.random.default_rng(0), 64, 16, sigma=0.0)
        b["targets"][:, -1] = 0           # every sequence's answer is class 0: an argmax over NaN would "hit"
        out = evaluate(HalfNaN(), "A", b, seed=0, chunk=16)
        self.assertEqual(out["nonfinite"], 32)
        self.assertEqual(out["acc"], 0.5)
        self.assertTrue(np.isfinite(out["loss"]))


class Saturated(torch.nn.Module):
    """Finite logits so extreme that the cross-entropy of a wrong class overflows to inf."""

    family = "gru"

    def forward(self, x, sigma, generator=None, diagnostics=False):
        B, T = x.shape
        logits = torch.zeros((B, T, tasks.TASKS["A"]["classes"]))
        logits[..., 0] = torch.finfo(torch.float32).max
        logits[..., 1] = -torch.finfo(torch.float32).max
        return logits, {}


class TestEligibility(unittest.TestCase):
    def test_infinite_validation_loss_is_not_eligible(self):
        b = tasks.batch("A", np.random.default_rng(1), 64, 16, sigma=0.0)
        b["targets"][:, -1] = np.where(np.arange(64) % 2 == 0, 0, 1)
        v = evaluate(Saturated(), "A", b, seed=0)
        self.assertEqual(v["nonfinite"], 0)
        self.assertFalse(np.isfinite(v["loss"]))
        self.assertFalse(eligible(v))
        self.assertTrue(eligible({"nonfinite": 0, "loss": 0.3, "acc": 0.9}))
        self.assertFalse(eligible({"nonfinite": 1, "loss": 0.3, "acc": 0.9}))


if __name__ == "__main__":
    unittest.main()
