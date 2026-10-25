#!/usr/bin/env python3
"""S3 probes: retrain an already-tried calibration configuration (same seed, so the same model bit for bit) and
evaluate it on more sigma values and other T, on the "calib" cells only. A probe is not a new configuration: it
spends no calibration budget and selects nothing; it measures where the selected models break and what a run costs.

    .venv/bin/python tools/probe.py calibration/probes/probe-1.json [--workers 12]

Every job is appended to calibration/probes.jsonl.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.calibrate import CAL, TRIALS, job_config, shared  # noqa: E402

PROBES = CAL / "probes.jsonl"


def work(args):
    probe, cfg, sh = args
    from phs.train import run, setup_determinism
    setup_determinism(1)
    rec = run(probe["kind"], probe["task"], probe["T"], sh["seed"], cfg, sigmas=tuple(probe["sigmas"]),
              n_test=sh["n_cells"], namespace="calib")
    rec.pop("cfg", None)
    rec.pop("curve", None)
    return {"probe": probe["id"], "of_trial": probe["trial"], "kind": probe["kind"], "task": probe["task"],
            "T": probe["T"], "cfg": cfg, "commit": sh["commit"], "rationale": probe["rationale"], **rec}


def main(argv: list[str]) -> int:
    spec = json.loads(Path(argv[0]).read_text())
    workers = int(argv[argv.index("--workers") + 1]) if "--workers" in argv else 1
    sh = shared()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "phs", "tools"], cwd=ROOT, capture_output=True,
                           text=True).stdout.strip()
    if dirty:
        print(f"refused: uncommitted changes in phs/ or tools/\n{dirty}")
        return 2
    trials = {}
    for r in map(json.loads, TRIALS.read_text().splitlines()):
        trials[(r["trial"], r["task"])] = r
    done = {json.loads(x)["probe"] for x in PROBES.read_text().splitlines()} if PROBES.exists() else set()
    jobs = []
    for p in spec["probes"]:
        if p["id"] in done:
            continue
        tried = trials.get((p["trial"], "A")) or trials.get((p["trial"], "B"))
        if tried is None or tried["kind"] != p["kind"]:
            print(f"refused: {p['id']} names {p['trial']}, which is not a tried {p['kind']} configuration")
            return 2
        cfg = job_config(p["kind"], p["task"], tried["overrides"], sh)
        extra = p.get("steps")
        if extra is not None:   # the only setting a probe may change: the shared step count, decided by time
            cfg["steps"] = extra
        jobs.append((p, cfg, dict(sh, commit=head)))
    print(f"{len(jobs)} probes, {workers} workers, commit {head}", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers, maxtasksperchild=1) as pool:
        for rec in pool.imap_unordered(work, jobs):
            with PROBES.open("a") as f:
                f.write(json.dumps(rec, sort_keys=True) + "\n")
            accs = " ".join(f"{k}:{v['acc']:.2f}" for k, v in rec.get("cells", {}).items())
            print(f"[{time.time() - t0:6.0f}s] {rec['probe']} ({rec['of_trial']} {rec['task']} T={rec['T']}): "
                  f"{rec['status']} wall {rec.get('wall_seconds', 0):.0f}s rss {rec['peak_rss_bytes'] / 2**20:.0f}MB "
                  f"{accs}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
