#!/usr/bin/env python3
"""The mutants of mutants.json, one at a time (the form of phs-soliton/tools/run_mutants.py, whose test harness
tools/mutant_harness.py runs each test here unchanged). A mutant replaces one exact string, occurring once, in one
file; EVERY test it names must then fail on an assertion (an import error or other crash proves nothing). Each test
runs in a fresh interpreter with a fresh bytecode cache; the file is restored after every mutant; every named test
must pass unmutated first. From phs-clock/:

    ../phs-soliton/.venv/bin/python mutants.py [ID...]
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HARNESS = ROOT.parent / "phs-soliton" / "tools" / "mutant_harness.py"
CACHE = ROOT / "runs" / ".mutant-cache"


def outcome(test: str, tag: str) -> tuple[str, str]:
    cache = CACHE / tag
    shutil.rmtree(cache, ignore_errors=True)
    cache.mkdir(parents=True)
    try:
        out = subprocess.run([sys.executable, "-X", f"pycache_prefix={cache}", str(HARNESS), test], cwd=ROOT,
                             capture_output=True, text=True)
    finally:
        shutil.rmtree(cache, ignore_errors=True)
    try:
        kinds = json.loads(out.stdout.strip().splitlines()[-1])["outcomes"]
    except (IndexError, ValueError, KeyError):
        return "crash", (out.stderr.strip().splitlines() or ["no output"])[-1][:300]
    for want in ("kill", "crash"):
        hit = [m for k, m in kinds if k == want]
        if hit:
            return want, hit[0]
    return ("pass", "") if not kinds else ("crash", "unknown outcome")


def main(argv: list) -> int:
    mutants = json.loads((ROOT / "mutants.json").read_text())["mutants"]
    chosen = argv or list(mutants)
    if [c for c in chosen if c not in mutants]:
        print(f"unknown mutants: {[c for c in chosen if c not in mutants]}")
        return 2
    tests = sorted({t for c in chosen for t in mutants[c]["kills"]})
    broken = {t: o for t in tests if (o := outcome(t, "baseline"))[0] != "pass"}
    for t, (how, msg) in broken.items():
        print(f"BASELINE {how}: {t}: {msg}")
    if broken or not tests:
        return 1
    survived = 0
    for mid in chosen:
        m = mutants[mid]
        path = ROOT / m["file"]
        original = path.read_text()
        if original.count(m["find"]) != 1:
            rows = [("invalid", "-", f"find string occurs {original.count(m['find'])} times")]
        else:
            try:
                path.write_text(original.replace(m["find"], m["replace"]))
                rows = [(o[0], t, o[1]) for t in m["kills"] for o in [outcome(t, mid)]]
            finally:
                path.write_text(original)
        ok = all(r[0] == "kill" for r in rows)
        survived += not ok
        print(f"{'killed  ' if ok else 'SURVIVED'} {mid}: {m['claim']}", flush=True)
        for how, t, msg in rows:
            print(f"    {how:<7} {t}: {msg[:140]}", flush=True)
    print(f"{len(chosen) - survived}/{len(chosen)} mutants killed on an assertion by every named test")
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
