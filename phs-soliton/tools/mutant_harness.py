#!/usr/bin/env python3
"""Run one unittest test (dotted name, importable from the current directory) and print how it ended as one JSON
line: {"ran": n, "outcomes": [[kind, message], ...]}, kind `kill` for a failed assertion and `crash` for any other
error (which proves nothing). Used by tools/run_mutants.py."""
from __future__ import annotations

import json
import os
import sys
import unittest


class Result(unittest.TestResult):
    def __init__(self) -> None:
        super().__init__()
        self.outcomes: list[list[str]] = []

    def _record(self, err) -> None:  # type: ignore[no-untyped-def]
        kind, value = err[0], err[1]
        if issubclass(kind, AssertionError):
            self.outcomes.append(["kill", str(value).splitlines()[0][:300] if str(value) else kind.__name__])
        else:
            self.outcomes.append(["crash", f"{kind.__name__}: {value}"[:300]])

    def addFailure(self, test, err):  # type: ignore[no-untyped-def]
        super().addFailure(test, err)
        self._record(err)

    def addError(self, test, err):  # type: ignore[no-untyped-def]
        super().addError(test, err)
        self._record(err)

    def addSubTest(self, test, subtest, err):  # type: ignore[no-untyped-def]
        super().addSubTest(test, subtest, err)
        if err is not None:
            self._record(err)


def main(argv: list[str]) -> int:
    sys.path.insert(0, os.getcwd())
    result = Result()
    try:
        suite = unittest.defaultTestLoader.loadTestsFromName(argv[0])
    except Exception as e:  # an import error under a mutant is a crash, never a kill
        print(json.dumps({"ran": 0, "outcomes": [["crash", f"{type(e).__name__}: {e}"[:300]]]}))
        return 0
    suite.run(result)
    if result.testsRun == 0:
        result.outcomes.append(["crash", "no test ran"])
    print(json.dumps({"ran": result.testsRun, "outcomes": result.outcomes}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
