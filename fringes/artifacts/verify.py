"""Check a frozen snapshot and replay results from its weights, without training (post hoc play).
Usage: artifacts/verify.py SNAPSHOT_DIR [--full | --bite | --only STAGE,...]   (uses the phs-soliton venv's python)

  hashes    every file in MANIFEST.json is there with its size and SHA-256, and nothing else is
  ckpts     every checkpoint loads and carries the kind, T, seed and step the manifest says
  exports   export_js.py regenerates each frozen export from its frozen checkpoint: every number within
            2e-6 + 1e-5 |x| (the parameters are rounded to 6 decimals; physics summaries such as the spectral norm of J
            come from LAPACK and differ in the last bits between CPUs), the float reference logits after 32 tokens
            within 1e-3 (the JS port agrees with torch to 5e-4); the worst gap is printed. A NaN, an infinity or a
            non-number on either side fails outright, before any error arithmetic (Codex review F5: a NaN ratio
            compares false both ways, so max() dropped it and the export passed)
  integer   intring.py on the frozen Soliton export at A = B = 8, round, reproduces results/bits-round/soliton-A8-B8
            exactly (integer arithmetic: no tolerance on any platform); --full adds Pure at A = B = 8 with floor, the
            PR #6 seam (results/bits/pure-A8-B8: 0.10 at sigma = 0)
  float     probe.py on the frozen Soliton checkpoint (seed 900, step 4000) at sigma = 0, 0.7, 4096 sequences each,
            reproduces acc_fixed and acc_probe of results/probe2-soliton-4000; --full adds Pure and sigma 0.5, 0.8.
            Bit-equal on the machine that made it; elsewhere float kernels may differ in the last bits, so a gap of
            up to 2 sequences in 4096 is reported and accepted, more fails; a nonfinite accuracy fails.
  --bite    the check must be able to fail, on copies with the manifest updated to match each change: one flipped
            byte (manifest left alone) must fail the hashes; a frozen integer result moved by one sequence must fail
            the replay; exports whose kappa is NaN, +inf, -inf, the literal 1e400 (a float parser makes it inf) or
            1e6 must each fail while the untouched exports pass; a NaN frozen probe accuracy must fail its row while
            the other row passes
Scratch output goes to fringes/runs/verify (ignored)."""
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FR = os.path.dirname(HERE)
LAB = os.path.dirname(FR)
PY = os.path.join(LAB, "phs-soliton", ".venv", "bin", "python")
SNAP = os.path.abspath(sys.argv[1])
FULL = "--full" in sys.argv
ONLY = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
TMP = os.path.join(FR, "runs", "verify")
os.makedirs(TMP, exist_ok=True)


def bite():
    """Doctor two copies of the snapshot; verify must refuse both."""
    def doctored(name):
        d = os.path.join(TMP, name)
        shutil.rmtree(d, ignore_errors=True)
        shutil.copytree(SNAP, d)
        return d
    a = doctored("bite-byte")
    f = os.path.join(a, "results", "collapse.json")
    raw = bytearray(open(f, "rb").read())
    raw[len(raw) // 2] ^= 1
    open(f, "wb").write(raw)
    b = doctored("bite-result")
    f = os.path.join(b, "results", "bits-round", "soliton-A8-B8.json")
    d = json.load(open(f))
    d["acc"][0] += 1 / 256
    json.dump(d, open(f, "w"))
    m = json.load(open(os.path.join(b, "MANIFEST.json")))
    for row in m["files"]:
        if row["path"] == "results/bits-round/soliton-A8-B8.json":
            row.update(bytes=os.path.getsize(f), sha256=hashlib.sha256(open(f, "rb").read()).hexdigest())
    json.dump(m, open(os.path.join(b, "MANIFEST.json"), "w"))
    c = doctored("bite-nonfinite")
    poison = {"qat-pure-A8": float("nan"), "qat-pure-A7": float("inf"), "qat-soliton-A7": float("-inf"),
              "qat-soliton-A6": "1e400", "golden": 1e6}
    changed = []
    for name, v in poison.items():
        f = os.path.join(c, "exports", f"{name}.json")
        d = json.load(open(f))
        d["kappa"] = "__RAW__" if isinstance(v, str) else v
        text = json.dumps(d, separators=(",", ":"))
        open(f, "w").write(text.replace('"__RAW__"', v) if isinstance(v, str) else text)
        changed.append(f"exports/{name}.json")
    f = os.path.join(c, "results", "probe2-soliton-4000.json")
    d = json.load(open(f))
    next(r for r in d["rows"] if abs(r["sigma"] - 0.7) < 1e-9)["acc_probe"] = float("nan")
    json.dump(d, open(f, "w"))
    changed.append("results/probe2-soliton-4000.json")
    for snap, paths in ((b, ["results/bits-round/soliton-A8-B8.json"]), (c, changed)):
        m = json.load(open(os.path.join(snap, "MANIFEST.json")))
        for row in m["files"]:
            if row["path"] in paths:
                f = os.path.join(snap, row["path"])
                row.update(bytes=os.path.getsize(f), sha256=hashlib.sha256(open(f, "rb").read()).hexdigest())
        json.dump(m, open(os.path.join(snap, "MANIFEST.json"), "w"))
    failed = []
    for d, only in ((a, "hashes"), (b, "hashes,integer"), (c, "hashes,exports,float")):
        r = subprocess.run([PY, os.path.abspath(__file__), d, "--only", only], capture_output=True, text=True)
        lines = r.stdout.strip().splitlines()
        ok = r.returncode != 0
        if d == c:   # every poisoned value refused, every untouched export and the other probe row still pass
            want = [f"BAD export {n} " for n in poison] + [f"ok  export {n} " for n in ("soliton", "pure", "homeo")]
            want += ["BAD float soliton sigma 0.70", "ok  float soliton sigma 0.00", "ok  SHA-256"]
            missing = [w for w in want if not any(ln.startswith(w) for ln in lines)]
            ok = ok and not missing
            for ln in lines:
                if ln.startswith("BAD"):
                    print(f"      {ln}")
            if missing:
                print(f"      missing: {missing}")
        print(f"{'ok ' if ok else 'BAD'} bite {os.path.basename(d)}: {lines[-1]}")
        if not ok:
            failed.append(d)
        shutil.rmtree(d)
    sys.exit(1 if failed else 0)


if "--bite" in sys.argv:
    bite()
man = json.load(open(os.path.join(SNAP, "MANIFEST.json")))
by_path = {f["path"]: f for f in man["files"]}
bad = []


def check(ok, what):
    print(f"{'ok ' if ok else 'BAD'} {what}", flush=True)
    if not ok:
        bad.append(what)


def run(*args):
    r = subprocess.run([PY, *args], capture_output=True, text=True, cwd=FR)
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-2000:])
    return r


def stage(name):
    return ONLY is None or name in ONLY


# hashes
if stage("hashes"):
    on_disk = {os.path.relpath(os.path.join(d, n), SNAP) for d, _, ns in os.walk(SNAP) for n in ns} - {"MANIFEST.json",
                                                                                                        "README.md"}
    check(on_disk == set(by_path), f"file set: {len(by_path)} in the manifest, {len(on_disk)} on disk, "
          f"extra {sorted(on_disk - set(by_path))[:3]}, missing {sorted(set(by_path) - on_disk)[:3]}")
    wrong = [p for p, f in by_path.items() if p in on_disk and (os.path.getsize(os.path.join(SNAP, p)) != f["bytes"] or
             hashlib.sha256(open(os.path.join(SNAP, p), "rb").read()).hexdigest() != f["sha256"])]
    check(not wrong, f"SHA-256 and size of {len(by_path)} files" + (f", wrong: {wrong[:3]}" if wrong else ""))

# checkpoints
if stage("ckpts"):
    r = run("-c", "\n".join([
        "import json, sys, torch",
        "man, snap = json.load(open(sys.argv[1])), sys.argv[2]",
        "for f in man['files']:",
        "    if f['path'].endswith('.pt'):",
        "        c = torch.load(snap + '/' + f['path'])",
        "        assert all(c[k] == f[k] for k in ('kind', 'T', 'seed', 'step')), f['path']"]),
        os.path.join(SNAP, "MANIFEST.json"), SNAP)
    check(r.returncode == 0, "every checkpoint loads with the manifest's kind, T, seed and step")

# exports from checkpoints
SRC = {"soliton": ("soliton-T032-s900-step04000.pt",), "pure": ("pure-T032-s900-step04000.pt",),
       "golden": ("golden-qp-q21-N34-s900-step04000.pt", "qp", "21"), "homeo": ("homeoarch-s900-step04000.pt", "homeo")}
for k in ("qat-soliton-A6", "qat-soliton-A7", "qat-pure-A7", "qat-pure-A8"):
    SRC[k] = (f"{k}-s900-step04000.pt",)


def is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def worst(a, b, path=""):
    """(excess over tolerance, path, gap) of the worst number; a structure mismatch, a non-number against a float
    and a NaN or infinity on either side are infinite, so the excess is never NaN and max() cannot drop it."""
    if isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            return (float("inf"), path or "/", "keys")
        return max((worst(a[k], b[k], f"{path}/{k}") for k in a), default=(0.0, path, 0.0))
    if isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            return (float("inf"), path, "length")
        return max((worst(x, y, f"{path}[{i}]") for i, (x, y) in enumerate(zip(a, b))), default=(0.0, path, 0.0))
    if isinstance(a, float) or isinstance(b, float):
        if not (is_num(a) and is_num(b)):
            return (float("inf"), path, "not a number")
        if not (math.isfinite(a) and math.isfinite(b)):
            return (float("inf"), path, f"nonfinite {a} vs {b}")
        gap = abs(a - b)
        tol = 1e-3 if path.startswith("/ref") else 2e-6 + 1e-5 * max(abs(a), abs(b))
        excess = gap / tol
        return (excess if math.isfinite(excess) else float("inf"), path, gap)
    return (0.0 if a == b else float("inf"), path, "differs")


if stage("exports"):
    for name, (ck, *mode) in SRC.items():
        out = os.path.join(TMP, f"{name}.json")
        r = run("-I", "export_js.py", LAB, os.path.join(SNAP, "checkpoints", ck), out, *mode)
        if r.returncode:
            check(False, f"export {name}")
            continue
        new, old = json.load(open(out)), json.load(open(os.path.join(SNAP, "exports", f"{name}.json")))
        w = worst(new, old)
        check(w[0] <= 1, f"export {name} regenerates from {ck} (worst {w[1]}: {w[2]:.1e})" if isinstance(w[2], float)
              else f"export {name} regenerates from {ck} ({w[1]}: {w[2]})")

# integer replay
if stage("integer"):
    INT = [("soliton", 8, 8, 1, "bits-round/soliton-A8-B8.json")]
    if FULL:
        INT.append(("pure", 8, 8, 0, "bits/pure-A8-B8.json"))
    for name, A, B, rnd, ref in INT:
        out = os.path.join(TMP, f"int-{name}-A{A}-B{B}-r{rnd}.json")
        r = run("-I", "intring.py", LAB, os.path.join(SNAP, "exports", f"{name}.json"), str(A), str(B), "256", out, str(rnd))
        got, want = json.load(open(out))["acc"], json.load(open(os.path.join(SNAP, "results", ref)))["acc"]
        finite = all(is_num(v) and math.isfinite(v) for v in got + want)
        check(r.returncode == 0 and finite and got == want,
              f"integer ring {name} A={A} B={B} rnd={rnd} reproduces {ref} exactly (Acc(0) {got[0]:.3f})")

# float replay
if stage("float"):
    FLOAT = [("soliton", "0.0,0.7")] + ([("soliton", "0.0,0.5,0.8"), ("pure", "0.0,0.7")] if FULL else [])
    for kind, sig in FLOAT:
        out = os.path.join(TMP, f"probe-{kind}-{sig.replace(',', '_')}.json")
        r = run("-I", "probe.py", LAB, os.path.join(SNAP, "checkpoints", f"{kind}-T032-s900-step04000.pt"), out, "2", "4096",
                sig)
        if r.returncode:
            check(False, f"probe {kind} {sig}")
            continue
        want = {round(x["sigma"], 2): x for x in json.load(open(os.path.join(SNAP, "results",
                                                                              f"probe2-{kind}-4000.json")))["rows"]}
        for row in json.load(open(out))["rows"]:
            w = want[round(row["sigma"], 2)]
            vals = [row[k] for k in ("acc_fixed", "acc_probe")] + [w[k] for k in ("acc_fixed", "acc_probe")]
            finite = all(is_num(v) and math.isfinite(v) for v in vals)
            gap = max(abs(row[k] - w[k]) * 4096 for k in ("acc_fixed", "acc_probe")) if finite else float("inf")
            check(gap <= 2, f"float {kind} sigma {row['sigma']:.2f}: acc_fixed {row['acc_fixed']:.4f} "
                  f"(frozen {w['acc_fixed']:.4f}), acc_probe {row['acc_probe']:.4f} (frozen {w['acc_probe']:.4f}), "
                  f"gap {gap:.0f} of 4096")

print(f"verify {man['snapshot']}: {'FAILED: ' + '; '.join(bad) if bad else 'ok'}")
sys.exit(1 if bad else 0)
