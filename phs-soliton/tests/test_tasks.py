"""Task generators against a plain reference (GOAL decision 5)."""
from __future__ import annotations

import unittest

import numpy as np

from phs.tasks import P_DIST, Q_MOD, SYMBOLS, digest, task_a, task_b, test_set


class TestTaskA(unittest.TestCase):
    def test_structure(self):
        n, T = 20_000, 32
        x, targets = task_a(np.random.default_rng(1), n, T)
        y = x[:, 0]
        self.assertTrue(((y >= 1) & (y <= SYMBOLS)).all())
        self.assertTrue((x[:, 1] == y).all(), "the marker is shown at t = 0 and t = 1")
        self.assertTrue((x[:, T - 1] == SYMBOLS + 1).all(), "the query token closes the sequence")
        hold = x[:, 2:T - 1]
        self.assertTrue(((hold >= 0) & (hold <= SYMBOLS)).all(), "distractors come from the marker alphabet")
        rate = float((hold > 0).mean())
        self.assertLess(abs(rate - P_DIST), 0.005, f"distractor rate {rate:.4f}")
        self.assertTrue((targets[:, T - 1] == y - 1).all())
        self.assertTrue((targets[:, :T - 1] == -100).all(), "loss on the last step only")


class TestTaskB(unittest.TestCase):
    def test_matches_reference(self):
        x, targets = task_b(np.random.default_rng(2), 500, 40)
        for row, tgt in zip(x, targets):
            s = 0
            for t, c in enumerate(row):
                s = (Q_MOD - s) % Q_MOD if c == 3 else (s + c) % Q_MOD
                self.assertEqual(tgt[t], s)
        self.assertEqual(set(np.unique(x)), {0, 1, 2, 3})


class TestTestSets(unittest.TestCase):
    def test_cells_are_fixed_and_distinct(self):
        a = test_set("A", 16, 0.10, n=256)
        self.assertEqual(digest(a), digest(test_set("A", 16, 0.10, n=256)))
        self.assertNotEqual(digest(a), digest(test_set("A", 16, 0.15, n=256)))
        self.assertTrue((a["sigma"] == 0.10).all())


if __name__ == "__main__":
    unittest.main()
