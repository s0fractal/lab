#!/usr/bin/env python3
"""The confirmatory runs of prereg.json, from a clean tree at the PREREG tag. From phs-clock/:

    ../phs-soliton/.venv/bin/python run.py run [--workers K]    # every run without a manifest
    ../phs-soliton/.venv/bin/python run.py status
    ../phs-soliton/.venv/bin/python run.py smoke KIND T STEPS   # seed 990, STEPS steps, into runs/smoke/: never a result

A run trains with phs-soliton's train.run (imported by phs-soliton's own loader from bytes checked against the tag
prereg-v1), saves its validation-selected checkpoint, then decodes it honestly (decode.py). `run` refuses unless the
tag exists, prereg.json equals the tagged one, the whole repository outside the output paths equals the tag (no file
changed, added, removed or untracked), the pinned SHA-256 of prereg.json hold and the registered configurations are
phs-soliton's. A worker checks the sources again before it starts and leaves an attempt record before it trains. One
manifest per run (status OK or FAILED with its reason), never overwritten. A run whose earlier attempt was killed
before its manifest runs again from the same seed; its manifest lists the interrupted attempts, and if one of them
left a checkpoint, the new run must reproduce its weights bit for bit or it is FAILED.
"""
from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent
SOLITON = LAB / "phs-soliton"
RESULTS = ROOT / "results"
MANIFESTS = RESULTS / "manifests"
CHECKPOINTS = RESULTS / "checkpoints"
RUNS = ROOT / "runs"
OUTPUTS = ("phs-clock/results", "phs-clock/REPORT.md")          # the only paths that may differ from the tag
SMOKE_SEED = 990


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=LAB, check=True, capture_output=True, text=True).stdout.strip()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def name(kind: str, T: int, seed: int) -> str:
    return f"A-{kind}-T{T:03d}-s{seed}"


def registered_config(prereg: dict, kind: str) -> dict:
    """The phs-soliton PREREG v1 configuration of task A (its analyze.expected_config), checked against ours."""
    v1 = json.loads((SOLITON / "prereg.json").read_text())
    cfg = dict(v1["optimisation"])
    cfg.update(v1["models"][kind]["config"])
    if cfg != prereg["models"][kind]["config"]:
        raise SystemExit(f"refused: the {kind} configuration in prereg.json is not phs-soliton's prereg-v1 one")
    return cfg


def pinned_problems(prereg: dict) -> list:
    return [f"{rel} differs from the SHA-256 pinned in prereg.json" for rel, h in prereg["frozen"].items()
            if sha256((LAB / rel).read_bytes()) != h]


def source_problems(tag_commit: str) -> list:
    """The working tree of the whole repository, outside OUTPUTS, is the tag's: no tracked file changed, added or
    removed since the tag (committed or not), and no untracked file (ignored ones, such as runs/ and the venv, are
    not sources)."""
    scope = ["--", ".", *[f":(exclude){o}" for o in OUTPUTS]]
    changed = git("diff", "--name-status", tag_commit, *scope)
    untracked = git("ls-files", "--others", "--exclude-standard", *scope)
    return ([f"differs from {tag_commit[:12]}: {line}" for line in changed.splitlines()]
            + [f"untracked: {line}" for line in untracked.splitlines()])


def state_sha256(state: dict) -> str:
    """SHA-256 of a state dict's tensors (names, dtypes, shapes, bytes): torch.save's file bytes vary run to run."""
    h = hashlib.sha256()
    for k in sorted(state):
        t = state[k].detach().contiguous().cpu()
        h.update(f"{k}|{t.dtype}|{tuple(t.shape)}|".encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def preconditions() -> dict:
    prereg = json.loads((ROOT / "prereg.json").read_text())
    try:
        tag_commit = git("rev-parse", "--verify", f"refs/tags/{prereg['tag']}^{{commit}}")
        v1_commit = git("rev-parse", "--verify", f"refs/tags/{prereg['reuse']['tag']}^{{commit}}")
    except subprocess.CalledProcessError:
        raise SystemExit(f"refused: tag {prereg['tag']} or {prereg['reuse']['tag']} does not exist")
    if v1_commit != prereg["reuse"]["commit"]:
        raise SystemExit(f"refused: {prereg['reuse']['tag']} is not {prereg['reuse']['commit']}")
    tagged = subprocess.run(["git", "show", f"{tag_commit}:phs-clock/prereg.json"], cwd=LAB, check=True,
                            capture_output=True).stdout
    here = (ROOT / "prereg.json").read_bytes()
    if tagged != here:
        raise SystemExit("refused: prereg.json differs from the tagged one")
    problems = pinned_problems(prereg) + source_problems(tag_commit)
    if problems:
        raise SystemExit("refused: " + "; ".join(problems[:5]))
    return {"prereg": prereg, "prereg_sha256": sha256(here), "tag": prereg["tag"], "tag_commit": tag_commit,
            "v1_commit": v1_commit, "commit": git("rev-parse", "HEAD")}


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


def load_phs(v1_commit: str):
    """phs-soliton's own loader (tools/confirm.py, unchanged): phs/ from bytes checked against prereg-v1."""
    spec = importlib.util.spec_from_file_location("phs_soliton_confirm", SOLITON / "tools" / "confirm.py")
    confirm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(confirm)
    return confirm.load_verified(v1_commit)


def work(job):
    kind, T, seed, cfg, ctx = job
    if not ctx.get("smoke"):
        problems = source_problems(ctx["tag_commit"]) + pinned_problems(ctx["prereg"])
        if problems:
            return {"abort": problems}
    try:
        train = load_phs(ctx["v1_commit"])
    except RuntimeError as e:
        return {"abort": [str(e)]}
    import torch
    import decode
    built = []
    build = train.build

    def capture(*args, **kw):   # train.run builds its model once; keep it to save the selected checkpoint
        built.append(build(*args, **kw))
        return built[-1]
    train.build = capture
    train.setup_determinism(1)
    t0, stage, rec = time.time(), "training", None
    out = Path(ctx["out"])
    # attempts: any record already here belongs to an attempt killed before its manifest (a job runs only without one)
    attempts = out / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    interrupted = sorted(p.name for p in attempts.glob(f"{name(kind, T, seed)}.*.json"))
    attempt = f"{name(kind, T, seed)}.{len(interrupted)}.json"
    (attempts / attempt).write_text(json.dumps({"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "pid": os.getpid(),
                                                "commit": ctx["commit"]}) + "\n")
    ckpt = out / "checkpoints" / f"{name(kind, T, seed)}.pt"
    left = state_sha256(torch.load(ckpt)["state"]) if interrupted and ckpt.exists() else None
    try:
        rec = train.run(kind, "A", T, seed, cfg, sigmas=tuple(ctx["sigmas"]), n_test=ctx["n_test"], namespace="test")
        if rec["status"] == "OK":
            stage = "checkpoint"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            state = built[0].state_dict()
            torch.save({"kind": kind, "T": T, "seed": seed, "step": rec["best_step"], "cfg": cfg, "state": state}, ckpt)
            rec["checkpoint_sha256"] = sha256(ckpt.read_bytes())
            rec["state_sha256"] = state_sha256(state)
            if left is not None and left != rec["state_sha256"]:
                raise RuntimeError("the weights differ from those an interrupted attempt of this run saved")
            stage, t1 = "honest decoding", time.time()
            if kind in ("pure", "soliton"):
                rec["honest"] = decode.phs(LAB, ctx["prereg"]["frozen"], ckpt, Path(ctx["work"]), ctx["sigmas"],
                                           ctx["n_test"])
            else:
                rec["honest"] = decode.gru(LAB, ctx["prereg"]["frozen"], ckpt, ctx["sigmas"], ctx["n_test"])
            rec["honest"]["seconds"] = time.time() - t1
    except (Exception, SystemExit) as e:  # a crash is a recorded failure, not a silent gap
        rec = rec or {"kind": kind, "task": "A", "T": T, "seed": seed, "cfg": cfg}
        rec.update(status="FAILED", reason=f"{stage}: {type(e).__name__}: {e}")
    rec.update(attempt=attempt, interrupted_attempts=interrupted, interrupted_state_sha256=left)
    rec.update(commit=ctx["commit"], tag=ctx["tag"], tag_commit=ctx["tag_commit"], prereg_sha256=ctx["prereg_sha256"],
               environment=ctx["environment"], started=time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t0)),
               wall_seconds_total=time.time() - t0, peak_rss_bytes_total=train.peak_rss_bytes())
    return rec


def write_manifest(rec: dict, where: Path) -> Path:
    where.mkdir(parents=True, exist_ok=True)
    path = where / (name(rec["kind"], rec["T"], rec["seed"]) + ".json")
    if path.exists():
        raise SystemExit(f"refused: {path.name} exists; a manifest is never overwritten")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    tmp.rename(path)
    return path


def grid(prereg: dict) -> list:
    d = prereg["design"]
    cost = {"pure": 20, "soliton": 20, "gru": 1}
    jobs = [(k, T, s) for k in d["models"] for T in d["T"] for s in d["seeds"]]
    return sorted(jobs, key=lambda j: (-cost[j[0]] * j[1], j))


def log(path: Path, rec: dict, t0: float) -> None:
    a = rec.get("cells", {}).get("0.00", {}).get("acc")
    print(f"[{time.time() - t0:7.0f}s] {path.name}: {rec['status']} acc(sigma 0) {a} "
          f"wall {rec.get('wall_seconds_total', 0):.0f}s (decoding {rec.get('honest', {}).get('seconds', 0):.0f}s) "
          f"rss {rec['peak_rss_bytes_total'] / 2**20:.0f}MB" + (f" reason {rec['reason']}" if rec["status"] != "OK"
                                                                else ""), flush=True)


def cmd_run(workers: int) -> int:
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / ".run.lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("refused: another run holds results/.run.lock")
        pre = preconditions()
        prereg = pre["prereg"]
        ctx = {k: pre[k] for k in ("prereg", "prereg_sha256", "tag", "tag_commit", "v1_commit", "commit")}
        ctx.update(environment=environment(), sigmas=prereg["design"]["sigma"], n_test=prereg["design"]["n_test"],
                   out=str(RESULTS), work=str(RUNS / "decoders"))
        todo = [j for j in grid(prereg) if not (MANIFESTS / (name(*j) + ".json")).exists()]
        jobs = [(k, T, s, registered_config(prereg, k), ctx) for k, T, s in todo]
        print(f"{len(jobs)} runs to do, {workers} workers", flush=True)
        t0 = time.time()
        with mp.get_context("spawn").Pool(workers, maxtasksperchild=1) as pool:
            for rec in pool.imap_unordered(work, jobs):
                if "abort" in rec:
                    pool.terminate()
                    raise SystemExit("aborted: the sources changed under the runner: " + "; ".join(rec["abort"][:5]))
                log(write_manifest(rec, MANIFESTS), rec, t0)
    return cmd_status()


def cmd_status() -> int:
    prereg = json.loads((ROOT / "prereg.json").read_text())
    ok = failed = missing = 0
    for j in grid(prereg):
        p = MANIFESTS / (name(*j) + ".json")
        if not p.exists():
            missing += 1
            continue
        m = json.loads(p.read_text())
        if m["status"] == "OK":
            ok += 1
        else:
            failed += 1
            print("FAILED", p.name, m.get("reason"))
    print(f"{ok} OK, {failed} FAILED, {missing} missing of {len(grid(prereg))}")
    return 0 if ok + failed == len(grid(prereg)) else 1


def cmd_smoke(kind: str, T: int, steps: int) -> int:
    """A short run through the same work() on the working tree, without the tag: for cost and end-to-end checks."""
    prereg = json.loads((ROOT / "prereg.json").read_text())
    problems = pinned_problems(prereg)
    if problems:
        raise SystemExit("refused: " + "; ".join(problems))
    cfg = dict(registered_config(prereg, kind), steps=steps)
    v1_commit = git("rev-parse", "--verify", f"refs/tags/{prereg['reuse']['tag']}^{{commit}}")
    smoke = RUNS / "smoke"
    ctx = {"prereg": prereg, "prereg_sha256": "smoke", "tag": "smoke", "tag_commit": "smoke", "v1_commit": v1_commit,
           "commit": git("rev-parse", "HEAD"), "environment": environment(), "sigmas": prereg["design"]["sigma"],
           "n_test": prereg["design"]["n_test"], "out": str(smoke), "work": str(smoke / "decoders"), "smoke": True}
    stem = name(kind, T, SMOKE_SEED)
    for old in [smoke / "manifests" / f"{stem}.json", smoke / "checkpoints" / f"{stem}.pt",
                *(smoke / "attempts").glob(f"{stem}.*.json")]:
        old.unlink(missing_ok=True)                    # a smoke run is redone from scratch, never reconciled
    t0 = time.time()
    with mp.get_context("spawn").Pool(1, maxtasksperchild=1) as pool:
        rec = pool.apply(work, ((kind, T, SMOKE_SEED, cfg, ctx),))
    if "abort" in rec:
        raise SystemExit("aborted: " + "; ".join(rec["abort"]))
    log(write_manifest(rec, smoke / "manifests"), rec, t0)
    return 0 if rec["status"] == "OK" else 1


def main(argv: list) -> int:
    sys.path.insert(0, str(ROOT))
    if argv[:1] == ["run"]:
        return cmd_run(int(argv[argv.index("--workers") + 1]) if "--workers" in argv else 1)
    if argv[:1] == ["status"]:
        return cmd_status()
    if argv[:1] == ["smoke"] and len(argv) == 4:
        return cmd_smoke(argv[1], int(argv[2]), int(argv[3]))
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
