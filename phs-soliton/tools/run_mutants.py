#!/usr/bin/env python3
"""Run the mutants of mutants.json, one at a time. A mutant replaces one exact string (occurring exactly once) in
one file; then EVERY test it names must fail on an assertion. An import error or any other crash proves nothing.
Each test runs in a fresh interpreter with a fresh bytecode cache (an earlier same-size mutant's cache could
otherwise be loaded), and the file is restored after every mutant. Before any mutant, every named test must pass
on the unmutated tree.

    .venv/bin/python tools/run_mutants.py                    # all
    .venv/bin/python tools/run_mutants.py ID...              # some
    .venv/bin/python tools/run_mutants.py --json status.json # also write the status as JSON
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "runs" / ".mutant-cache"


def outcome(test: str, tag: str) -> tuple[str, str]:
    """('pass', '') if the test passed, ('kill', msg) on a failed assertion, ('crash', msg) otherwise."""
    cache = CACHE / tag
    shutil.rmtree(cache, ignore_errors=True)
    cache.mkdir(parents=True)
    try:
        out = subprocess.run([sys.executable, "-X", f"pycache_prefix={cache}", str(ROOT / "tools" / "mutant_harness.py"),
                              test], cwd=ROOT, capture_output=True, text=True)
    finally:
        shutil.rmtree(cache, ignore_errors=True)
    try:
        doc = json.loads(out.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        return "crash", (out.stderr.strip().splitlines() or ["no output"])[-1][:300]
    kinds = [k for k, _ in doc["outcomes"]]
    if not kinds:
        return "pass", ""
    for want in ("kill", "crash"):
        if want in kinds:
            return want, doc["outcomes"][kinds.index(want)][1]
    return "crash", "unknown outcome"


def run(mid: str, m: dict) -> tuple[bool, list[dict]]:
    path = ROOT / m["file"]
    original = path.read_text()
    n = original.count(m["find"])
    if n != 1:
        return False, [{"test": "-", "outcome": "invalid", "message": f"find string occurs {n} times in {m['file']}"}]
    rows = []
    try:
        path.write_text(original.replace(m["find"], m["replace"]))
        for t in m["kills"]:
            how, msg = outcome(t, mid)
            rows.append({"test": t, "outcome": how, "message": msg})
    finally:
        path.write_text(original)
    return all(r["outcome"] == "kill" for r in rows), rows


def main(argv: list[str]) -> int:
    json_out = None
    if "--json" in argv:
        i = argv.index("--json")
        json_out = Path(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    mutants = json.loads((ROOT / "mutants.json").read_text())["mutants"]
    chosen = argv or list(mutants)
    unknown = [c for c in chosen if c not in mutants]
    if unknown:
        print(f"unknown mutants: {unknown}")
        return 2
    tests = sorted({t for c in chosen for t in mutants[c]["kills"]})
    if not tests:
        print("no tests named: an empty scope proves nothing")
        return 2
    baseline = {t: outcome(t, "baseline") for t in tests}
    broken = {t: o for t, o in baseline.items() if o[0] != "pass"}
    for t, (how, msg) in broken.items():
        print(f"BASELINE {how}: {t}: {msg}")
    if broken:
        return 1
    print(f"baseline: {len(tests)} named tests pass unmutated")
    status, survived = {}, 0
    for mid in chosen:
        ok, rows = run(mid, mutants[mid])
        survived += not ok
        status[mid] = {"killed": ok, "claim": mutants[mid]["claim"], "tests": rows}
        print(f"{'killed  ' if ok else 'SURVIVED'} {mid}", flush=True)
        for r in rows:
            print(f"    {r['outcome']:<7} {r['test']}: {r['message'][:160]}", flush=True)
    print(f"{len(chosen) - survived}/{len(chosen)} mutants killed on an assertion by every named test")
    if json_out:
        json_out.write_text(json.dumps({"killed": len(chosen) - survived, "total": len(chosen), "mutants": status},
                                       indent=1, ensure_ascii=False) + "\n")
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
