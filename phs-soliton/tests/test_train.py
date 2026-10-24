"""Evaluation policy for non-finite outputs (phs/train.py)."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from phs import tasks
from phs.train import evaluate


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


if __name__ == "__main__":
    unittest.main()
