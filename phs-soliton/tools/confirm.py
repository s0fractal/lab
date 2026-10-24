#!/usr/bin/env python3
"""S5: the confirmatory runs of prereg.json, from a clean tree whose experiment files equal the PREREG tag.

    .venv/bin/python tools/confirm.py gate                       # tests + mutants -> results/gate.json
    .venv/bin/python tools/confirm.py run [--workers K]          # every run without a manifest
    .venv/bin/python tools/confirm.py rerun KIND TASK T SEED --reason "..."   # one run again, same seed
    .venv/bin/python tools/confirm.py status

Refuses unless: the tag exists; prereg.json is byte-identical to the tagged one; every tracked file of phs-soliton/
outside the output paths (results/, RESULTS.md, REPORT.md, figures/) equals the tag; none of them has uncommitted
changes. A manifest is written per run (status OK or FAILED with its reason) and never overwritten: a rerun moves
the old manifest to results/failures/ and records why.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUTPUTS = ("results", "RESULTS.md", "REPORT.md", "figures")
RESULTS = ROOT / "results"
MANIFESTS = RESULTS / "manifests"
FAILURES = RESULTS / "failures"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def preconditions() -> dict:
    prereg = json.loads((ROOT / "prereg.json").read_text())
    tag = prereg["tag"]
    try:
        tag_commit = git("rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}")
    except subprocess.CalledProcessError:
        raise SystemExit(f"refused: tag {tag} does not exist")
    head = git("rev-parse", "HEAD")
    excludes = [f":(exclude){p}" for p in OUTPUTS]
    changed = git("diff", "--name-only", tag_commit, "HEAD", "--", ".", *excludes)
    if changed:
        raise SystemExit(f"refused: HEAD differs from {tag} outside the output paths:\n{changed}")
    dirty = git("status", "--porcelain", "--", ".", *excludes)
    if dirty:
        raise SystemExit(f"refused: uncommitted changes outside the output paths:\n{dirty}")
    rel = ROOT.relative_to(Path(git("rev-parse", "--show-toplevel")))
    tagged = subprocess.run(["git", "show", f"{tag_commit}:{rel}/prereg.json"], cwd=ROOT, check=True,
                            capture_output=True).stdout
    here = (ROOT / "prereg.json").read_bytes()
    if tagged != here:
        raise SystemExit("refused: prereg.json differs from the tagged one")
    if sha256_bytes((ROOT / "analyze.py").read_bytes()) != prereg["frozen"]["analyze.py"]:
        raise SystemExit("refused: analyze.py differs from the SHA-256 pinned in prereg.json")
    return {"prereg": prereg, "prereg_sha256": sha256_bytes(here), "tag": tag, "tag_commit": tag_commit,
            "commit": head}


def source_problems(tag_commit: str) -> list:
    """Every tracked experiment file (outside the output paths) must have the tag's bytes, and no untracked file may
    sit beside them: what a worker is about to import is what the tag froze."""
    top = Path(git("rev-parse", "--show-toplevel"))
    rel = ROOT.relative_to(top).as_posix()
    tagged = {}
    for line in git("ls-tree", "-r", "--full-tree", tag_commit, "--", rel).splitlines():
        meta, path = line.split("\t", 1)
        if path[len(rel) + 1:].split("/")[0] in OUTPUTS:
            continue
        tagged[path] = meta.split()[2]
    paths = sorted(tagged)
    out = subprocess.run(["git", "hash-object", "--stdin-paths"], cwd=top, input="\n".join(paths), text=True,
                         capture_output=True, check=True).stdout.split()
    problems = [f"{p} differs from {tag_commit[:12]}" for p, h in zip(paths, out) if tagged[p] != h]
    excludes = [f":(exclude){o}" for o in OUTPUTS]
    untracked = git("status", "--porcelain", "--untracked-files=all", "--", ".", *excludes)
    problems += [f"uncommitted: {line}" for line in untracked.splitlines()]
    return problems


class Lock:
    """One gate, run or rerun at a time: a gate mutates sources while it runs."""

    def __enter__(self):
        RESULTS.mkdir(exist_ok=True)
        self.f = open(RESULTS / ".confirm.lock", "w")
        try:
            fcntl.flock(self.f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("refused: another gate, run or rerun holds results/.confirm.lock")
        return self

    def __exit__(self, *exc):
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


def environment() -> dict:
    import numpy
    import torch
    try:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    except OSError:
        cpu = platform.processor()
    return {"python": platform.python_version(), "torch": torch.__version__, "numpy": numpy.__version__,
            "platform": platform.platform(), "machine": platform.machine(), "cpu": cpu, "cores": os.cpu_count(),
            "device": "cpu", "dtype": "float32", "threads_per_run": 1, "deterministic": True}


def name(kind: str, task: str, T: int, seed: int) -> str:
    return f"{task}-{kind}-T{T:03d}-s{seed}"


def model_config(prereg: dict, kind: str, task: str) -> dict:
    from phs import tasks
    from phs.models import build, count_params
    import torch
    cfg = dict(prereg["optimisation"])
    cfg.update(prereg["models"][kind]["config"])
    if kind == "gru_pm":
        spec = tasks.TASKS[task]
        sol = dict(prereg["optimisation"], **prereg["models"]["soliton"]["config"])
        cfg["pm_target"] = count_params(build("soliton", spec["vocab"], spec["classes"], sol, torch.Generator()))
    return cfg


def grid(prereg: dict) -> list:
    g = prereg["grid"]
    jobs = [(k, t, T, s) for k in g["models"] for t in ("A", "B") for T in g["T"] for s in g["seeds"]]
    cost = {"pure": 20, "soliton": 20, "gru": 1, "gru_pm": 1}
    return sorted(jobs, key=lambda j: (-cost[j[0]] * j[2], j))


def work(args):
    kind, task, T, seed, cfg, ctx = args
    problems = source_problems(ctx["tag_commit"])   # before anything of phs/ is imported in this process
    if problems:
        return {"abort": problems}
    from phs.train import peak_rss_bytes, run, setup_determinism
    setup_determinism(1)
    t0 = time.time()
    try:
        rec = run(kind, task, T, seed, cfg, sigmas=tuple(ctx["sigmas"]), n_test=ctx["n_test"], namespace="test")
    except Exception as e:  # a crash is a recorded failure, not a silent gap
        rec = {"kind": kind, "task": task, "T": T, "seed": seed, "cfg": cfg, "status": "FAILED",
               "reason": f"{type(e).__name__}: {e}", "wall_seconds": time.time() - t0,
               "peak_rss_bytes": peak_rss_bytes()}
    rec.update(commit=ctx["commit"], tag=ctx["tag"], tag_commit=ctx["tag_commit"],
               prereg_sha256=ctx["prereg_sha256"], environment=ctx["environment"],
               started=time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t0)), **ctx.get("extra", {}))
    return rec


def write_manifest(rec: dict) -> Path:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    path = MANIFESTS / (name(rec["kind"], rec["task"], rec["T"], rec["seed"]) + ".json")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    tmp.rename(path)
    return path


def context(pre: dict) -> dict:
    return {"commit": pre["commit"], "tag": pre["tag"], "tag_commit": pre["tag_commit"],
            "prereg_sha256": pre["prereg_sha256"], "environment": environment(),
            "n_test": pre["prereg"]["tasks"]["n_test"], "sigmas": pre["prereg"]["grid"]["sigma"]}


def cmd_gate() -> int:
    with Lock():
        return gate()


def gate() -> int:
    pre = preconditions()
    t0 = time.time()
    tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."], cwd=ROOT,
                           capture_output=True, text=True)
    status_path = RESULTS / "mutants-status.json"
    RESULTS.mkdir(exist_ok=True)
    status_path.unlink(missing_ok=True)   # never read a matrix this gate did not write
    registered = len(json.loads((ROOT / "mutants.json").read_text())["mutants"])
    mut = subprocess.run([sys.executable, "tools/run_mutants.py", "--json", str(status_path)], cwd=ROOT,
                         capture_output=True, text=True)
    status = json.loads(status_path.read_text()) if status_path.exists() else {"killed": 0, "total": 0}
    check = subprocess.run([sys.executable, "tools/prereg.py", "check"], cwd=ROOT, capture_output=True, text=True)
    gate = {"commit": pre["commit"], "tag": pre["tag"], "tag_commit": pre["tag_commit"],
            "prereg_sha256": pre["prereg_sha256"], "tests_ok": tests.returncode == 0,
            "tests_summary": (tests.stderr.strip().splitlines() or [""])[-3:],
            "mutants_killed": status["killed"], "mutants_total": status["total"],
            "mutants_registered": registered, "mutants_exit": mut.returncode, "prereg_check_ok": check.returncode == 0,
            "prereg_check": check.stdout.strip().splitlines()[-3:], "environment": environment(),
            "seconds": time.time() - t0}
    (RESULTS / "gate.json").write_text(json.dumps(gate, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: gate[k] for k in ("tests_ok", "mutants_killed", "mutants_total", "prereg_check_ok")}))
    return 0 if gate_ok(gate, pre) else 1


def gate_ok(gate: dict, pre: dict) -> bool:
    return bool(gate.get("tests_ok") is True and gate.get("prereg_check_ok") is True and gate.get("mutants_exit") == 0
                and gate.get("mutants_killed") == gate.get("mutants_total") == gate.get("mutants_registered") > 0
                and gate.get("prereg_sha256") == pre["prereg_sha256"] and gate.get("tag_commit") == pre["tag_commit"])


def admitted_gate(pre: dict) -> dict:
    gate_path = RESULTS / "gate.json"
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else {}
    if not gate_ok(gate, pre):
        raise SystemExit("refused: no passing gate for this PREREG and tag; run `tools/confirm.py gate`")
    problems = source_problems(pre["tag_commit"])
    if problems:
        raise SystemExit("refused: " + "; ".join(problems[:5]))
    import analyze
    named = [(p.name, json.loads(p.read_text())) for p in sorted(MANIFESTS.glob("*.json"))]
    problems = analyze.admit(pre["prereg"], pre["prereg_sha256"], named, gate)
    if problems:
        raise SystemExit(f"refused: existing manifests are not admissible: {problems[:5]}")
    return gate


def cmd_run(workers: int) -> int:
    with Lock():
        return run_grid(workers)


def run_grid(workers: int) -> int:
    pre = preconditions()
    admitted_gate(pre)
    ctx = context(pre)
    todo = [j for j in grid(pre["prereg"]) if not (MANIFESTS / (name(*j) + ".json")).exists()]
    jobs = [(k, t, T, s, model_config(pre["prereg"], k, t), ctx) for k, t, T, s in todo]
    print(f"{len(jobs)} runs to do, {workers} workers", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers, maxtasksperchild=1) as pool:
        for rec in pool.imap_unordered(work, jobs):
            if "abort" in rec:
                pool.terminate()
                raise SystemExit("aborted: the source changed under the runner: " + "; ".join(rec["abort"][:5]))
            path = write_manifest(rec)
            a = rec.get("cells", {}).get("0.00", {}).get("acc")
            print(f"[{time.time() - t0:7.0f}s] {path.name}: {rec['status']} acc(sigma 0) {a} "
                  f"wall {rec.get('wall_seconds', 0):.0f}s rss {rec['peak_rss_bytes'] / 2**20:.0f}MB"
                  + (f" reason {rec['reason']}" if rec["status"] != "OK" else ""), flush=True)
    return cmd_status()


def cmd_rerun(kind: str, task: str, T: int, seed: int, reason: str) -> int:
    if not reason.strip():
        raise SystemExit("refused: a rerun needs a reason")
    with Lock():
        return rerun(kind, task, T, seed, reason)


def rerun(kind: str, task: str, T: int, seed: int, reason: str) -> int:
    pre = preconditions()
    admitted_gate(pre)
    path = MANIFESTS / (name(kind, task, T, seed) + ".json")
    previous = None
    if path.exists():
        old = json.loads(path.read_text())
        if old["status"] == "OK":
            raise SystemExit("refused: the run has an OK manifest")
        FAILURES.mkdir(parents=True, exist_ok=True)
        n = len(list(FAILURES.glob(path.stem + ".*.json")))
        previous = FAILURES / f"{path.stem}.{n}.json"
        path.rename(previous)
    ctx = context(pre)
    ctx["extra"] = {"rerun_reason": reason,
                    "previous_manifest": str(previous.relative_to(ROOT)) if previous else None}
    rec = work((kind, task, T, seed, model_config(pre["prereg"], kind, task), ctx))
    if "abort" in rec:
        raise SystemExit("aborted: " + "; ".join(rec["abort"][:5]))
    print(write_manifest(rec).name, rec["status"])
    return 0


def cmd_status() -> int:
    prereg = json.loads((ROOT / "prereg.json").read_text())
    jobs = grid(prereg)
    ok = failed = missing = 0
    for j in jobs:
        p = MANIFESTS / (name(*j) + ".json")
        if not p.exists():
            missing += 1
        elif json.loads(p.read_text())["status"] == "OK":
            ok += 1
        else:
            failed += 1
            print("FAILED", p.name, json.loads(p.read_text()).get("reason"))
    print(f"{ok} OK, {failed} FAILED, {missing} missing of {len(jobs)}")
    return 0 if ok == len(jobs) else 1


def main(argv: list[str]) -> int:
    if argv[:1] == ["gate"]:
        return cmd_gate()
    if argv[:1] == ["run"]:
        workers = int(argv[argv.index("--workers") + 1]) if "--workers" in argv else 1
        return cmd_run(workers)
    if argv[:1] == ["rerun"] and "--reason" in argv:
        kind, task, T, seed = argv[1], argv[2], int(argv[3]), int(argv[4])
        return cmd_rerun(kind, task, T, seed, argv[argv.index("--reason") + 1])
    if argv[:1] == ["status"]:
        return cmd_status()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
