"""The runner's source guard on a throwaway git repository: anything outside the output paths that differs from the
tag is refused, a committed addition included (Codex r1 F2). Run from phs-clock/."""
from __future__ import annotations

import shutil
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import run  # noqa: E402


class SourceGuard(unittest.TestCase):
    def setUp(self):
        self.repo = ROOT / "runs" / "test-source-guard"
        shutil.rmtree(self.repo, ignore_errors=True)
        (self.repo / "phs-clock" / "results").mkdir(parents=True)
        (self.repo / "phs-clock" / "run.py").write_text("x = 1\n")
        (self.repo / "phs-clock" / "results" / "a.json").write_text("{}\n")
        (self.repo / ".gitignore").write_text("ignored/\n")
        self.git("init", "-q")
        self.git("add", "-A")
        self.git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "tag")
        self.tag = self.git("rev-parse", "HEAD")
        self.orig = run.LAB
        run.LAB = self.repo

    def tearDown(self):
        run.LAB = self.orig

    def git(self, *args):
        return subprocess.run(["git", *args], cwd=self.repo, check=True, capture_output=True, text=True).stdout.strip()

    def commit(self):
        self.git("add", "-A")
        self.git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "later")

    def test_clean_tree_and_outputs_pass(self):
        self.assertEqual(run.source_problems(self.tag), [])
        (self.repo / "phs-clock" / "results" / "b.json").write_text("{}\n")
        (self.repo / "phs-clock" / "REPORT.md").write_text("report\n")
        (self.repo / "ignored").mkdir()
        (self.repo / "ignored" / "numpy.py").write_text("\n")
        self.commit()
        self.assertEqual(run.source_problems(self.tag), [])

    def test_committed_addition_is_refused(self):
        (self.repo / "phs-clock" / "numpy.py").write_text("raise SystemExit\n")
        self.commit()
        self.assertTrue(any("numpy.py" in p for p in run.source_problems(self.tag)))

    def test_change_removal_and_untracked_are_refused(self):
        (self.repo / "phs-clock" / "run.py").write_text("x = 2\n")
        self.assertTrue(any("run.py" in p for p in run.source_problems(self.tag)))
        self.git("checkout", "--", ".")
        (self.repo / "phs-clock" / "run.py").unlink()
        self.assertTrue(any("run.py" in p for p in run.source_problems(self.tag)))
        self.git("checkout", "--", ".")
        (self.repo / "fringes.py").write_text("\n")
        self.assertTrue(any("fringes.py" in p for p in run.source_problems(self.tag)))


if __name__ == "__main__":
    unittest.main()
