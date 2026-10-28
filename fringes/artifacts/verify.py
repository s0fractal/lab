"""Check a frozen snapshot and replay results from its weights, without training (post hoc play).
Usage: artifacts/verify.py SNAPSHOT_DIR [--full | --bite | --only STAGE,...]   (uses the phs-soliton venv's python)

A stage runs when its inputs are in the snapshot; a checkpoint or export it needs may live in a sibling snapshot
(the rounding study reuses the checkpoints and seed-900 exports of pr7-2026-10-28).
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
  rounding  rounding.py replays Soliton A = 7 stochastic and Pure A = 6 nearest at sigma = 0 and 0.2 from the seed-900
            exports: accuracy exactly, the phase-step statistics to 1e-9 relative (integer dynamics, the statistics
            are float sums of integers); --full adds Soliton A = 7 feedback and Pure A = 6 stochastic
  readout   with --full only (about 3 minutes): readout.py replays Soliton seed 900 from its frozen checkpoint. The
            states are float and the decoders are fit on 51k of them, so another CPU flips a few predictions: two
            GitHub runs (different runners) differed by up to 3 of 1024 at sigma <= 0.95 and up to 6 above, and in
            one of them the mean sensors drifted by more than 1e-6 at sigma >= 1.1. Each accuracy must agree within
            one binomial standard error of the frozen value (at least 3 of 1024; about 6 at accuracy 0.15, the
            largest gap seen): the replay must stay inside the sampling noise of the frozen estimate, while the
            decoders the README compares differ by 4 standard errors or more. D and
            K agree within 1e-3 relative, 50 times finer than their change between neighbouring sigmas
  --bite    the check must be able to fail, on copies with the manifest updated to match each change: one flipped
            byte (manifest left alone) must fail the hashes; a frozen result moved by one sequence must fail its
            replay; poisoned numbers (NaN, +inf, -inf, the literal 1e400 that a float parser makes inf, a large finite
            value) in exports and results must each fail while the untouched ones pass
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
PR7 = os.path.join(HERE, "pr7-2026-10-28")

# export name -> (checkpoint, export_js mode args); the checkpoint is looked up in this snapshot, then in pr7
SRC = {"soliton": ("soliton-T032-s900-step04000.pt",), "pure": ("pure-T032-s900-step04000.pt",),
       "golden": ("golden-qp-q21-N34-s900-step04000.pt", "qp", "21"), "homeo": ("homeoarch-s900-step04000.pt", "homeo")}
for k in ("qat-soliton-A6", "qat-soliton-A7", "qat-pure-A7", "qat-pure-A8"):
    SRC[k] = (f"{k}-s900-step04000.pt",)
for k in ("soliton", "pure"):
    for s in ("s901", "s902"):
        SRC[f"{k}-{s}"] = (f"{k}-T032-{s}-step04000.pt",)
# rounding replays: (model, bits, mode) -> frozen result; the export is the seed-900 one of pr7
ROUND = [("soliton", 7, "stochastic"), ("pure", 6, "nearest")]
ROUND_FULL = [("soliton", 7, "feedback"), ("pure", 6, "stochastic")]


def find(*parts):
    """A file of this snapshot, or of pr7 when this snapshot reuses it."""
    for root in (SNAP, PR7):
        p = os.path.join(root, *parts)
        if os.path.exists(p):
            return p
    return None


def is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def finite(*vals):
    return all(is_num(v) and math.isfinite(v) for v in vals)


def rehash(snap, paths):
    m = json.load(open(os.path.join(snap, "MANIFEST.json")))
    for row in m["files"]:
        if row["path"] in paths:
            f = os.path.join(snap, row["path"])
            row.update(bytes=os.path.getsize(f), sha256=hashlib.sha256(open(f, "rb").read()).hexdigest())
    json.dump(m, open(os.path.join(snap, "MANIFEST.json"), "w"))


def poison_json(path, setter, raw=None):
    """Apply setter(d) to the JSON at path; raw replaces the placeholder "__RAW__" by a literal such as 1e400."""
    d = json.load(open(path))
    setter(d)
    text = json.dumps(d, separators=(",", ":"))
    open(path, "w").write(text.replace('"__RAW__"', raw) if raw else text)


def bite():
    """Doctor copies of the snapshot; verify must refuse each, and must still pass what was not touched."""
    files = {f["path"] for f in json.load(open(os.path.join(SNAP, "MANIFEST.json")))["files"]}
    rounding = "results/rounding/soliton-A7-B7-stochastic.json" in files
    if "results/readout/soliton-s900.json" in files:
        return bite_readout(files)

    def doctored(name):
        d = os.path.join(TMP, name)
        shutil.rmtree(d, ignore_errors=True)
        shutil.copytree(SNAP, d)
        return d
    plans = []
    a = doctored("bite-byte")
    target = "results/collapse.json" if "results/collapse.json" in files else sorted(p for p in files
                                                                                     if p.startswith("results/"))[0]
    raw = bytearray(open(os.path.join(a, target), "rb").read())
    raw[len(raw) // 2] ^= 1
    open(os.path.join(a, target), "wb").write(raw)
    plans.append((a, "hashes", [], []))

    b = doctored("bite-result")
    if rounding:
        moved = "results/rounding/soliton-A7-B7-stochastic.json"
        poison_json(os.path.join(b, moved), lambda d: d["rows"][0].__setitem__("acc", d["rows"][0]["acc"] + 1 / 512))
        only = "hashes,rounding"
    else:
        moved = "results/bits-round/soliton-A8-B8.json"
        poison_json(os.path.join(b, moved), lambda d: d["acc"].__setitem__(0, d["acc"][0] + 1 / 256))
        only = "hashes,integer"
    rehash(b, [moved])
    plans.append((b, only, [], []))

    c = doctored("bite-nonfinite")
    if rounding:
        poison = {"pure-s901": float("nan"), "pure-s902": float("inf"), "soliton-s901": "1e400"}
        controls = ["soliton-s902"]
        result = "results/rounding/pure-A6-B6-nearest.json"
        poison_json(os.path.join(c, result), lambda d: d["rows"][0].__setitem__("acc", float("nan")))
        want_rows = ["BAD rounding pure A=6 nearest sigma 0.00", "ok  rounding pure A=6 nearest sigma 0.20",
                     "ok  rounding soliton A=7 stochastic sigma 0.00"]
        only = "hashes,exports,rounding"
    else:
        poison = {"qat-pure-A8": float("nan"), "qat-pure-A7": float("inf"), "qat-soliton-A7": float("-inf"),
                  "qat-soliton-A6": "1e400", "golden": 1e6}
        controls = ["soliton", "pure", "homeo"]
        result = "results/probe2-soliton-4000.json"
        poison_json(os.path.join(c, result), lambda d: next(r for r in d["rows"] if abs(r["sigma"] - 0.7) < 1e-9)
                    .__setitem__("acc_probe", float("nan")))
        want_rows = ["BAD float soliton sigma 0.70", "ok  float soliton sigma 0.00"]
        only = "hashes,exports,float"
    for name, v in poison.items():
        poison_json(os.path.join(c, "exports", f"{name}.json"),
                    lambda d, v=v: d.__setitem__("kappa", "__RAW__" if isinstance(v, str) else v),
                    v if isinstance(v, str) else None)
    rehash(c, [f"exports/{n}.json" for n in poison] + [result])
    want = [f"BAD export {n} " for n in poison] + [f"ok  export {n} " for n in controls] + want_rows + ["ok  SHA-256"]
    plans.append((c, only, want, list(poison)))

    failed = []
    for d, only, want, _ in plans:
        r = subprocess.run([PY, os.path.abspath(__file__), d, "--only", only], capture_output=True, text=True)
        lines = r.stdout.strip().splitlines()
        missing = [w for w in want if not any(ln.startswith(w) for ln in lines)]
        ok = r.returncode != 0 and not missing
        if want:
            for ln in lines:
                if ln.startswith("BAD"):
                    print(f"      {ln}")
            if missing:
                print(f"      missing: {missing}")
        print(f"{'ok ' if ok else 'BAD'} bite {os.path.basename(d)}: {lines[-1] if lines else r.stderr[-300:]}")
        if not ok:
            failed.append(d)
        shutil.rmtree(d)
    sys.exit(1 if failed else 0)


def bite_readout(files):
    """One flipped byte must fail the hashes; one replay of a copy with a result moved by 30 sequences (sigma 0.70,
    about 2 standard errors, twice the bound) and a NaN (sigma 0.80) must fail exactly those rows and pass the
    sigma 0 control."""
    a = os.path.join(TMP, "bite-byte")
    shutil.rmtree(a, ignore_errors=True)
    shutil.copytree(SNAP, a)
    f = os.path.join(a, "results/readout/pure-s900.json")
    raw = bytearray(open(f, "rb").read())
    raw[len(raw) // 2] ^= 1
    open(f, "wb").write(raw)
    c = os.path.join(TMP, "bite-readout")
    shutil.rmtree(c, ignore_errors=True)
    shutil.copytree(SNAP, c)
    target = "results/readout/soliton-s900.json"

    def doctor(d):
        rows = {round(r["sigma"], 2): r for r in d["rows"]}
        rows[0.7]["sensor"] += 30 / 1024
        rows[0.8]["oracle"] = float("nan")
    poison_json(os.path.join(c, target), doctor)
    rehash(c, [target])
    plans = [(a, "hashes", []), (c, "hashes,readout", ["BAD readout soliton s900 sigma 0.70",
                                                         "BAD readout soliton s900 sigma 0.80",
                                                         "ok  readout soliton s900 sigma 0.00", "ok  SHA-256"])]
    failed = []
    for d, only, want in plans:
        r = subprocess.run([PY, os.path.abspath(__file__), d, "--only", only], capture_output=True, text=True)
        lines = r.stdout.strip().splitlines()
        missing = [w for w in want if not any(ln.startswith(w) for ln in lines)]
        ok = r.returncode != 0 and not missing
        for ln in lines:
            if want and ln.startswith("BAD"):
                print(f"      {ln}")
        if missing:
            print(f"      missing: {missing}")
        print(f"{'ok ' if ok else 'BAD'} bite {os.path.basename(d)}: {lines[-1] if lines else r.stderr[-300:]}")
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
        if not finite(a, b):
            return (float("inf"), path, f"nonfinite {a} vs {b}")
        gap = abs(a - b)
        tol = 1e-3 if path.startswith("/ref") else 2e-6 + 1e-5 * max(abs(a), abs(b))
        excess = gap / tol
        return (excess if math.isfinite(excess) else float("inf"), path, gap)
    return (0.0 if a == b else float("inf"), path, "differs")


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
if stage("ckpts") and any(p.endswith(".pt") for p in by_path):
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
if stage("exports"):
    for name, (ck, *mode) in SRC.items():
        if f"exports/{name}.json" not in by_path:
            continue
        ckpt = find("checkpoints", ck)
        if ckpt is None:
            check(False, f"export {name}: checkpoint {ck} is in neither snapshot")
            continue
        out = os.path.join(TMP, f"{name}.json")
        r = run("-I", "export_js.py", LAB, ckpt, out, *mode)
        if r.returncode:
            check(False, f"export {name}")
            continue
        new, old = json.load(open(out)), json.load(open(os.path.join(SNAP, "exports", f"{name}.json")))
        w = worst(new, old)
        check(w[0] <= 1, f"export {name} regenerates from {ck} (worst {w[1]}: {w[2]:.1e})" if isinstance(w[2], float)
              else f"export {name} regenerates from {ck} ({w[1]}: {w[2]})")

# integer replay
if stage("integer") and "results/bits-round/soliton-A8-B8.json" in by_path:
    INT = [("soliton", 8, 8, 1, "bits-round/soliton-A8-B8.json")]
    if FULL:
        INT.append(("pure", 8, 8, 0, "bits/pure-A8-B8.json"))
    for name, A, B, rnd, ref in INT:
        out = os.path.join(TMP, f"int-{name}-A{A}-B{B}-r{rnd}.json")
        r = run("-I", "intring.py", LAB, os.path.join(SNAP, "exports", f"{name}.json"), str(A), str(B), "256", out, str(rnd))
        got, want = json.load(open(out))["acc"], json.load(open(os.path.join(SNAP, "results", ref)))["acc"]
        check(r.returncode == 0 and finite(*got, *want) and got == want,
              f"integer ring {name} A={A} B={B} rnd={rnd} reproduces {ref} exactly (Acc(0) {got[0]:.3f})")

# float replay
if stage("float") and "results/probe2-soliton-4000.json" in by_path:
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
            ok = finite(*(row[k] for k in ("acc_fixed", "acc_probe")), *(w[k] for k in ("acc_fixed", "acc_probe")))
            gap = max(abs(row[k] - w[k]) * 4096 for k in ("acc_fixed", "acc_probe")) if ok else float("inf")
            check(gap <= 2, f"float {kind} sigma {row['sigma']:.2f}: acc_fixed {row['acc_fixed']:.4f} "
                  f"(frozen {w['acc_fixed']:.4f}), acc_probe {row['acc_probe']:.4f} (frozen {w['acc_probe']:.4f}), "
                  f"gap {gap:.0f} of 4096")

# rounding replay
if stage("rounding") and any(p.startswith("results/rounding/") for p in by_path):
    for m, b, mode in ROUND + (ROUND_FULL if FULL else []):
        ref = f"results/rounding/{m}-A{b}-B{b}-{mode}.json"
        frozen = {round(r["sigma"], 2): r for r in json.load(open(os.path.join(SNAP, ref)))["rows"]}
        out = os.path.join(TMP, f"rounding-{m}-A{b}-{mode}.json")
        r = run("rounding.py", LAB, find("exports", f"{m}.json"), str(b), str(b), mode, out, "512", "0.0,0.2")
        if r.returncode:
            check(False, f"rounding {m} A={b} {mode}")
            continue
        for row in json.load(open(out))["rows"]:
            w = frozen[round(row["sigma"], 2)]
            keys = [k for k in w if k != "sigma"]
            ok = all(k in row and finite(row[k], w[k]) for k in keys)
            ok = ok and row["acc"] == w["acc"] and all(abs(row[k] - w[k]) <= 1e-9 * max(1.0, abs(w[k])) for k in keys)
            check(ok, f"rounding {m} A={b} {mode} sigma {row['sigma']:.2f}: acc {row['acc']:.4f} "
                  f"(frozen {w['acc']:.4f}), bias {row['bias_lsb']:+.4f}, frozen steps {row['frozen']:.4f}")

# honest-decoder replay (slow: every sigma of calibration and test, so only with --full or --only readout)
if (FULL or (ONLY and "readout" in ONLY)) and "results/readout/soliton-s900.json" in by_path:
    out = os.path.join(TMP, "readout-soliton-s900.json")
    cpu = next((ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name")), "?")
    print(f"--  readout replay on {cpu}", flush=True)
    r = run("readout.py", LAB, find("checkpoints", "soliton-T032-s900-step04000.pt"), out, "2048", "2")
    if r.returncode:
        check(False, "readout soliton s900")
    else:
        frozen = json.load(open(os.path.join(SNAP, "results/readout/soliton-s900.json")))["rows"]
        accs = ("fixed", "pooled", "sensor", "sigma_lda", "oracle")
        for row, w in zip(json.load(open(out))["rows"], frozen):
            if not all(finite(row[k], w[k]) for k in accs + ("D", "K")):
                check(False, f"readout soliton s900 sigma {row['sigma']:.2f}: nonfinite value")
                continue
            tol = {k: max(3.0, math.sqrt(w[k] * (1 - w[k]) * 1024)) for k in accs}
            gaps = {k: abs(row[k] - w[k]) * 1024 for k in accs}
            rel = {k: abs(row[k] - w[k]) / max(abs(w[k]), 1e-12) for k in ("D", "K")}
            k_bad = max(accs, key=lambda k: gaps[k] / tol[k])
            check(all(gaps[k] <= tol[k] for k in accs) and max(rel.values()) <= 1e-3,
                  f"readout soliton s900 sigma {row['sigma']:.2f}: sensor {row['sensor']:.4f} (frozen {w['sensor']:.4f}), "
                  f"tightest {k_bad} gap {gaps[k_bad]:.0f} of bound {tol[k_bad]:.1f}, D and K off by "
                  f"{max(rel.values()):.1e}")

print(f"verify {man['snapshot']}: {'FAILED: ' + '; '.join(bad) if bad else 'ok'}")
sys.exit(1 if bad else 0)
