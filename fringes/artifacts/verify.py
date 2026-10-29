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
  decoder   in a decoder snapshot: every decoder file holds all its arrays, of one shape and finite except where an
            LDA saw no state of a class (a whole NaN column of W and its entry of b, which transfer.py never
            predicts; named in the log), on the 25-point grid (13 points for the decoders fit on sigma <= 0.6), and
            its sensor-bin edges and mean calibration
            sensors equal those of artifacts/readout-2026-10-28, which the same readout.py run wrote (exactly: there
            is no arithmetic between them). In a transfer snapshot: every result carries those same edges, exactly, and
            a calibration check of 0 (Codex review of #8, 1dec5d4: another seed format had drawn other calibration
            states and refit the decoder; 3ad06e9: refitting on another CPU moves the edges by ~3e-5, so the decoder is
            now frozen rather than refit)
  rescore   transfer.py --readout-cells scores the frozen decoders of Soliton seed 900 and Pure seed 901 on readout.py's
            own test cells at sigma = 0, 0.7, 0.8: all five accuracies within the readout bound (one binomial standard
            error of 1024, at least 3) and D within 1e-3 relative; exact on the machine that made them. transfer.py
            refuses any decoder file but the frozen one, so a doctored copy fails here even with its manifest updated
  transfer  transfer.py --cells replays the model's own readout and the oracle at sigma = 0 for T = 24 and 48 (Soliton
            and Pure, seed 900): the time-locked readout, within one binomial standard error of 512 (at least 3);
            --full adds the whole transfer.py run of Soliton seed 900 with the frozen decoders (fresh, between, beyond,
            T = 24 and 48: 60 test cells, a few minutes): every accuracy within one binomial standard error of 512
            (at least 3), D within 1e-3 relative, the sigma cell, the edges and the calibration check exactly
  clocklaw  clocklaw.py replays Soliton seed 900 at T = 48: the overlap r(sigma) and the norm at all 25 sigmas within
            0.01, the first zero sigma0 within 0.005 (averages of 1024 float states; another CPU moves them by ~1e-5)
  --bite    the check must be able to fail, on copies with the manifest updated to match each change: one flipped
            byte (manifest left alone) must fail the hashes; a frozen result moved by one sequence must fail its
            replay; poisoned numbers (NaN, +inf, -inf, the literal 1e400 that a float parser makes inf, a large finite
            value) in exports and results must each fail while the untouched ones pass; a frozen reference emptied
            of its rows, or missing its last row, must fail (every replay checks the whole grid of its reference
            first, then that the replay returned exactly the sigmas it asked for)
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
READOUT = os.path.join(HERE, "readout-2026-10-28")
LDA = ["pooled", "low_pooled"] + [f"bin{k:02d}" for k in range(16)] + [f"low_bin{k:02d}" for k in range(16)] + \
    [f"cell{i:02d}" for i in range(25)]                                 # the LDAs in a frozen decoder file

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
# the sigma grids the producers write; a frozen reference must hold the whole grid, a replay exactly what was asked
GRID = {"probe": [round(0.05 * i, 2) for i in range(41)], "rounding": [round(0.1 * i, 1) for i in range(13)],
        "readout": [round(0.05 * i, 2) for i in range(25)]}
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


def sigmas(rows):
    return [round(r["sigma"], 2) if is_num(r.get("sigma")) else None for r in rows]


def cpu_name():
    """For the log only: never let a diagnostic block a check (there is no /proc on macOS)."""
    try:
        with open("/proc/cpuinfo") as f:
            return next((ln.split(":", 1)[1].strip() for ln in f if ln.startswith("model name")), "?")
    except OSError:
        pass
    try:
        r = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    import platform
    return platform.processor() or platform.machine() or "?"


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
    if "results/clocklaw/soliton-s900.json" in files:
        return bite_transfer()
    if "decoders/soliton-s900.npz" in files:
        return bite_decoder()

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

    # a reference with no rows, and one missing its last row (Codex review of #8: zip compared nothing and passed)
    ref, stage_ = (("results/rounding/soliton-A7-B7-stochastic.json", "rounding") if rounding
                   else ("results/probe2-soliton-4000.json", "float"))
    for name, cut in (("bite-empty-ref", lambda rows: []), ("bite-short-ref", lambda rows: rows[:-1])):
        e = doctored(name)
        poison_json(os.path.join(e, ref), lambda d, cut=cut: d.__setitem__("rows", cut(d["rows"])))
        rehash(e, [ref])
        plans.append((e, f"hashes,{stage_}", ["ok  SHA-256", "BAD "], []))

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
    # a reference with no rows, and one missing its last row (Codex review of #8: zip compared nothing and passed);
    # both are refused before the replay
    for name, cut in (("bite-empty-ref", lambda rows: []), ("bite-short-ref", lambda rows: rows[:-1])):
        e = os.path.join(TMP, name)
        shutil.rmtree(e, ignore_errors=True)
        shutil.copytree(SNAP, e)
        poison_json(os.path.join(e, target), lambda d, cut=cut: d.__setitem__("rows", cut(d["rows"])))
        rehash(e, [target])
        plans.append((e, "hashes,readout", ["ok  SHA-256", "BAD readout soliton s900: the frozen reference"]))
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


def bite_transfer():
    """A flipped byte fails the hashes; a moved overlap (T = 48, sigma 0.60, by 0.1), a moved readout accuracy
    (T = 48, sigma 0, by 30 of 512) and a moved sensor-bin edge (Pure seed 901, by 1e-3: a decoder other than the
    frozen one) fail while their neighbours pass; an emptied overlap curve and a shortened T = 24 table fail before
    any replay."""
    def copy(name):
        d = os.path.join(TMP, name)
        shutil.rmtree(d, ignore_errors=True)
        shutil.copytree(SNAP, d)
        return d
    cl, tr = "results/clocklaw/soliton-s900.json", "results/transfer/soliton-s900.json"
    a = copy("bite-byte")
    f = os.path.join(a, cl)
    raw = bytearray(open(f, "rb").read())
    raw[len(raw) // 2] ^= 1
    open(f, "wb").write(raw)
    b = copy("bite-moved")
    poison_json(os.path.join(b, cl), lambda d: d["T48"]["r"].__setitem__(12, d["T48"]["r"][12] + 0.1))
    poison_json(os.path.join(b, tr), lambda d: d["T48"][0].__setitem__("fixed", d["T48"][0]["fixed"] + 30 / 512))
    rehash(b, [cl, tr])
    poison_json(os.path.join(b, "results/transfer/pure-s901.json"),
                lambda d: d["edges"].__setitem__(7, d["edges"][7] + 1e-3))
    rehash(b, ["results/transfer/pure-s901.json"])
    e = copy("bite-empty-ref")
    poison_json(os.path.join(e, cl), lambda d: d["T48"].__setitem__("r", []))
    poison_json(os.path.join(e, tr), lambda d: d.__setitem__("T24", d["T24"][:-1]))
    rehash(e, [cl, tr])
    plans = [(a, "hashes", []),
             (b, "hashes,decoder,clocklaw,transfer", ["BAD clocklaw soliton s900 T48 sigma 0.60",
                                                       "ok  clocklaw soliton s900 T48 sigma 0.55",
                                                       "BAD transfer soliton s900 T48 sigma 0.00",
                                                       "ok  transfer soliton s900 T24 sigma 0.00",
                                                       "BAD decoder pure s901", "ok  decoder pure s900",
                                                       "ok  SHA-256"]),
             (e, "hashes,clocklaw,transfer", ["BAD clocklaw soliton s900: the frozen reference",
                                               "BAD transfer soliton s900: the frozen reference", "ok  SHA-256"])]
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


def bite_decoder():
    """A flipped byte fails the hashes. A copy with the manifest updated has one sensor-bin edge of Pure seed 901 moved
    by 1e-3 (the three-decimal calibration seed of 1dec5d4 moved them by 1.2e-3), a NaN in one LDA of Soliton seed 902
    and one array of Pure seed 902 dropped: each must fail the decoder check while Soliton and Pure seed 900 pass, and
    transfer.py must refuse to rescore with the moved Pure seed 901 file while Soliton seed 900 rescores."""
    import numpy as np

    def copy(name):
        d = os.path.join(TMP, name)
        shutil.rmtree(d, ignore_errors=True)
        shutil.copytree(SNAP, d)
        return d

    def doctor(d, rel, change):
        f = os.path.join(d, rel)
        a = dict(np.load(f, allow_pickle=False))
        change(a)
        with open(f, "wb") as fh:                                          # np.savez(path) would append ".npz"
            np.savez(fh, **a)
        rehash(d, [rel])
    a = copy("bite-byte")
    f = os.path.join(a, "decoders/pure-s900.npz")
    raw = bytearray(open(f, "rb").read())
    raw[len(raw) // 2] ^= 1
    open(f, "wb").write(raw)
    b = copy("bite-moved")
    doctor(b, "decoders/pure-s901.npz", lambda x: x["edges"].__setitem__(7, x["edges"][7] + 1e-3))
    doctor(b, "decoders/soliton-s902.npz", lambda x: x["cell12_W"].__setitem__((0, 0), float("nan")))
    doctor(b, "decoders/pure-s902.npz", lambda x: x.pop("low_bin15_b"))
    plans = [(a, "hashes", []),
             (b, "hashes,decoder,rescore", ["BAD decoder pure s901", "BAD decoder soliton s902", "BAD decoder pure s902",
                                            "ok  decoder soliton s900", "ok  decoder pure s900",
                                            "BAD rescore pure s901: transfer.py returned 0 rows, asked for 3 (it ref",
                                            "ok  rescore soliton s900 sigma 0.70", "ok  SHA-256"])]
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
    check(bool(by_path) and not wrong, f"SHA-256 and size of {len(by_path)} files"
          + (f", wrong: {wrong[:3]}" if wrong else "") + ("" if by_path else ": the manifest lists no files"))

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
        got = json.load(open(out))["acc"] if r.returncode == 0 else []
        want = json.load(open(os.path.join(SNAP, "results", ref)))["acc"]
        full = len(got) == len(want) == 13                              # intring.py scores sigma 0, 0.1, ..., 1.2
        check(full and finite(*got, *want) and got == want,
              f"integer ring {name} A={A} B={B} rnd={rnd} reproduces {ref} exactly"
              + (f" (Acc(0) {got[0]:.3f})" if full else f": {len(got)} replayed and {len(want)} frozen of 13 points"))

# float replay
if stage("float") and "results/probe2-soliton-4000.json" in by_path:
    FLOAT = [("soliton", "0.0,0.7")] + ([("soliton", "0.0,0.5,0.8"), ("pure", "0.0,0.7")] if FULL else [])
    for kind, sig in FLOAT:
        frozen_rows = json.load(open(os.path.join(SNAP, "results", f"probe2-{kind}-4000.json")))["rows"]
        if sigmas(frozen_rows) != GRID["probe"]:
            check(False, f"float {kind}: the frozen reference holds {len(frozen_rows)} rows, not the 41-point grid")
            continue
        out = os.path.join(TMP, f"probe-{kind}-{sig.replace(',', '_')}.json")
        r = run("-I", "probe.py", LAB, os.path.join(SNAP, "checkpoints", f"{kind}-T032-s900-step04000.pt"), out, "2", "4096",
                sig)
        if r.returncode:
            check(False, f"probe {kind} {sig}")
            continue
        want = {round(x["sigma"], 2): x for x in frozen_rows}
        replayed = json.load(open(out))["rows"]
        if sigmas(replayed) != [round(float(v), 2) for v in sig.split(",")]:
            check(False, f"float {kind}: the replay returned sigmas {sigmas(replayed)}, asked for {sig}")
            continue
        for row in replayed:
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
        frozen_rows = json.load(open(os.path.join(SNAP, ref)))["rows"]
        if sigmas(frozen_rows) != GRID["rounding"]:
            check(False, f"rounding {m} A={b} {mode}: the frozen reference holds {len(frozen_rows)} rows, "
                  f"not the 13-point grid")
            continue
        frozen = {round(r["sigma"], 2): r for r in frozen_rows}
        out = os.path.join(TMP, f"rounding-{m}-A{b}-{mode}.json")
        r = run("rounding.py", LAB, find("exports", f"{m}.json"), str(b), str(b), mode, out, "512", "0.0,0.2")
        if r.returncode:
            check(False, f"rounding {m} A={b} {mode}")
            continue
        replayed = json.load(open(out))["rows"]
        if sigmas(replayed) != [0.0, 0.2]:
            check(False, f"rounding {m} A={b} {mode}: the replay returned sigmas {sigmas(replayed)}, asked for 0.0, 0.2")
            continue
        for row in replayed:
            w = frozen[round(row["sigma"], 2)]
            keys = [k for k in w if k != "sigma"]
            ok = all(k in row and finite(row[k], w[k]) for k in keys)
            ok = ok and row["acc"] == w["acc"] and all(abs(row[k] - w[k]) <= 1e-9 * max(1.0, abs(w[k])) for k in keys)
            check(ok, f"rounding {m} A={b} {mode} sigma {row['sigma']:.2f}: acc {row['acc']:.4f} "
                  f"(frozen {w['acc']:.4f}), bias {row['bias_lsb']:+.4f}, frozen steps {row['frozen']:.4f}")

# honest-decoder replay (slow: every sigma of calibration and test, so only with --full or --only readout)
if (FULL or (ONLY and "readout" in ONLY)) and "results/readout/soliton-s900.json" in by_path:
    out = os.path.join(TMP, "readout-soliton-s900.json")
    frozen = json.load(open(os.path.join(SNAP, "results/readout/soliton-s900.json")))["rows"]
    if sigmas(frozen) != GRID["readout"]:   # checked before the 3-minute replay, which then has nothing to stand on
        check(False, f"readout soliton s900: the frozen reference holds {len(frozen)} rows, not the 25-point grid")
    else:
        print(f"--  readout replay on {cpu_name()}", flush=True)
        r = run("readout.py", LAB, find("checkpoints", "soliton-T032-s900-step04000.pt"), out, "2048", "2")
        replayed = json.load(open(out))["rows"] if r.returncode == 0 else []
        if r.returncode or sigmas(replayed) != GRID["readout"]:
            check(False, f"readout soliton s900: the replay returned {len(replayed)} rows, not the 25-point grid")
            replayed = []
        accs = ("fixed", "pooled", "sensor", "sigma_lda", "oracle")
        for row, w in zip(replayed, frozen):
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

# a decoder snapshot: the decoder files are readout.py's, as frozen
if stage("decoder") and "decoders/soliton-s900.npz" in by_path:
    import numpy as np
    want_keys = {"sigmas", "edges", "D_cal", "low_sigmas", "low_edges"} | {f"{n}_{x}" for n in LDA for x in "Wb"}
    for kind in ("soliton", "pure"):
        for seed in (900, 901, 902):
            what = f"decoder {kind} s{seed}"
            ref = json.load(open(os.path.join(READOUT, f"results/readout/{kind}-s{seed}.json")))
            try:
                a = dict(np.load(os.path.join(SNAP, f"decoders/{kind}-s{seed}.npz"), allow_pickle=False))
            except (OSError, ValueError) as e:
                check(False, f"{what}: does not load ({e})")
                continue
            if set(a) != want_keys:
                check(False, f"{what}: arrays missing or extra: {sorted(set(a) ^ want_keys)[:3]}")
                continue
            shapes = {(a[f"{n}_W"].shape, a[f"{n}_b"].shape) for n in LDA}
            problems = [] if len(shapes) == 1 and next(iter(shapes))[0][1:] == next(iter(shapes))[1] else ["shapes"]
            # a class with no state in an LDA's training cells has a NaN mean: a whole column of W and its entry of b
            # are NaN, and transfer.py never predicts it. That is the only nonfinite value allowed
            absent, nonfinite = [], [k for k in ("sigmas", "edges", "D_cal", "low_sigmas", "low_edges")
                                     if not np.isfinite(a[k]).all()]
            for n in LDA if not problems else []:
                W, b = a[f"{n}_W"], a[f"{n}_b"]
                gone = np.isnan(W).all(0)
                if not (np.isfinite(W[:, ~gone]).all() and np.isfinite(b[~gone]).all() and np.isnan(b[gone]).all()):
                    nonfinite.append(n)
                absent += [f"{n} class {k}" for k in np.flatnonzero(gone)]
            problems += [f"nonfinite values in {', '.join(nonfinite[:3])}"] if nonfinite else []
            low = [x for x in GRID["readout"] if x <= 0.6]
            problems += [] if a["sigmas"].tolist() == GRID["readout"] == sigmas(ref["rows"]) and \
                a["low_sigmas"].tolist() == low and a["edges"].shape == a["low_edges"].shape == (15,) else ["grid"]
            gap = float(np.max(np.abs(a["edges"] - np.asarray(ref["edges"])))) if "grid" not in problems else math.inf
            problems += [] if a["edges"].tolist() == ref["edges"] and \
                a["D_cal"].tolist() == [r["D_cal"] for r in ref["rows"]] else ["edges or D_cal differ from the readout"]
            check(not problems, f"{what}: the frozen readout.py decoder, sensor-bin edges off by {gap:.1e}"
                  + (f", an LDA without a class: {', '.join(absent)}" if absent else "")
                  + (f": {', '.join(problems)}" if problems else ""))

# the frozen decoders rescore readout.py's own test cells
if stage("rescore") and "decoders/soliton-s900.npz" in by_path:
    accs = ("fixed", "pooled", "sensor", "sigma_lda", "oracle")
    print(f"--  rescore on {cpu_name()}", flush=True)
    for kind, seed in (("soliton", 900), ("pure", 901)):
        frozen = {round(r["sigma"], 2): r for r in
                  json.load(open(os.path.join(READOUT, f"results/readout/{kind}-s{seed}.json")))["rows"]}
        out = os.path.join(TMP, f"rescore-{kind}-s{seed}.json")
        r = run("transfer.py", LAB, find("checkpoints", f"{kind}-T032-s{seed}-step04000.pt") or
                os.path.join(PR7, "checkpoints", f"{kind}-T032-s{seed}-step04000.pt"), out, "2",
                "--decoder", os.path.join(SNAP, f"decoders/{kind}-s{seed}.npz"), "--readout-cells", "0.0,0.7,0.8")
        rows = json.load(open(out))["rows"] if r.returncode == 0 else []
        if sigmas(rows) != [0.0, 0.7, 0.8]:
            check(False, f"rescore {kind} s{seed}: transfer.py returned {len(rows)} rows, asked for 3"
                  + (" (it refused the decoder)" if "not the frozen" in r.stderr else ""))
            continue
        for row in rows:
            w = frozen[round(row["sigma"], 2)]
            if not all(finite(row[k], w[k]) for k in accs + ("D",)):
                check(False, f"rescore {kind} s{seed} sigma {row['sigma']:.2f}: nonfinite value")
                continue
            gaps = {k: abs(row[k] - w[k]) * 1024 for k in accs}
            tol = {k: max(3.0, math.sqrt(w[k] * (1 - w[k]) * 1024)) for k in accs}
            rel = abs(row["D"] - w["D"]) / max(abs(w["D"]), 1e-12)
            exact = all(row[k] == w[k] for k in accs + ("D",))
            check(all(gaps[k] <= tol[k] for k in accs) and rel <= 1e-3,
                  f"rescore {kind} s{seed} sigma {row['sigma']:.2f}: sensor {row['sensor']:.4f} (readout "
                  f"{w['sensor']:.4f}), sigma_lda {row['sigma_lda']:.4f} (readout {w['sigma_lda']:.4f}), "
                  + ("exact" if exact else f"largest gap {max(gaps.values()):.0f} of 1024, D off by {rel:.1e}"))

# a transfer snapshot: every result was made with the frozen readout.py decoder
if stage("decoder") and "results/transfer/soliton-s900.json" in by_path:
    for kind in ("soliton", "pure"):
        for seed in (900, 901, 902):
            t = json.load(open(os.path.join(SNAP, f"results/transfer/{kind}-s{seed}.json")))
            ref_path = os.path.join(READOUT, f"results/readout/{kind}-s{seed}.json")
            e_t, e_r = t.get("edges"), json.load(open(ref_path)).get("edges") if os.path.exists(ref_path) else None
            ok = isinstance(e_t, list) and isinstance(e_r, list) and len(e_t) == len(e_r) == 15 and finite(*e_t, *e_r)
            gap = max(abs(a - b) for a, b in zip(e_t, e_r)) if ok else float("inf")
            cc = t.get("calibration_check", {})
            ok = ok and e_t == e_r and finite(cc.get("edges"), cc.get("D_cal")) and cc["edges"] == cc["D_cal"] == 0
            check(ok, f"decoder {kind} s{seed}: sensor-bin edges equal the frozen readout decoder's (off by {gap:.1e})")

# transfer replay: the time-locked readout at sigma = 0 for T = 24 and 48
if stage("transfer") and "results/transfer/soliton-s900.json" in by_path:
    for kind in ("soliton", "pure"):
        frozen = json.load(open(os.path.join(SNAP, f"results/transfer/{kind}-s900.json")))
        if any(sigmas(frozen.get(f"T{T}", [])) != GRID["readout"] for T in (24, 48)):
            check(False, f"transfer {kind} s900: the frozen reference lacks the 25-point grid at T = 24 or 48")
            continue
        out = os.path.join(TMP, f"transfer-{kind}-s900.json")
        r = run("transfer.py", LAB, find("checkpoints", f"{kind}-T032-s900-step04000.pt"), out, "2",
                "--cells", "24:0.0,48:0.0")
        cells = json.load(open(out))["cells"] if r.returncode == 0 else []
        if [(c_["T"], c_["sigma"]) for c_ in cells] != [(24, 0.0), (48, 0.0)]:
            check(False, f"transfer {kind} s900: the replay returned {len(cells)} cells, asked for 2")
            continue
        for c_ in cells:
            w = frozen[f"T{c_['T']}"][0]
            ok = all(finite(c_[k], w[k]) for k in ("fixed", "oracle"))
            gaps = {k: abs(c_[k] - w[k]) * 512 for k in ("fixed", "oracle")} if ok else {}
            ok = ok and all(gaps[k] <= max(3.0, math.sqrt(w[k] * (1 - w[k]) * 512)) for k in gaps)
            check(ok, f"transfer {kind} s900 T{c_['T']} sigma 0.00: own readout {c_['fixed']:.3f} (frozen {w['fixed']:.3f}),"
                  f" oracle {c_['oracle']:.3f} (frozen {w['oracle']:.3f})")

# the whole transfer run of one model with the frozen decoders (slow, so only with --full or --only transfer-full)
if (FULL or (ONLY and "transfer-full" in ONLY)) and "results/transfer/soliton-s900.json" in by_path:
    frozen = json.load(open(os.path.join(SNAP, "results/transfer/soliton-s900.json")))
    parts = {"fresh": [0.5, 0.6, 0.7, 0.8, 0.9], "between": [0.525, 0.625, 0.725, 0.825, 0.925],
             "beyond": [0.7, 0.8, 0.9], "T24": GRID["readout"], "T48": GRID["readout"]}
    if any([r_.get("sigma") for r_ in frozen.get(p_, [])] != v for p_, v in parts.items()):
        check(False, "transfer-full soliton s900: the frozen reference lacks a part or a sigma")
    else:
        print(f"--  full transfer replay on {cpu_name()}", flush=True)
        out = os.path.join(TMP, "transfer-full-soliton-s900.json")
        r = run("transfer.py", LAB, find("checkpoints", "soliton-T032-s900-step04000.pt"), out, "2")
        new = json.load(open(out)) if r.returncode == 0 else {}
        check(new.get("edges") == frozen["edges"] and new.get("calibration_check") == frozen["calibration_check"],
              "transfer-full soliton s900: the frozen decoder (edges and calibration check equal)")
        for p_, v in parts.items():
            got = new.get(p_, [])
            if [r_.get("sigma") for r_ in got] != v:
                check(False, f"transfer-full soliton s900 {p_}: {len(got)} rows replayed, {len(v)} asked for")
                continue
            gaps, worst_ = [], (0.0, "")
            for row, w in zip(got, frozen[p_]):
                ks = [k for k in w if k not in ("sigma", "sigma_cell", "D")]
                if not all(k in row and finite(row[k], w[k]) for k in ks + ["D"]) or \
                        row["sigma_cell"] != w["sigma_cell"]:
                    gaps.append(math.inf)
                    continue
                for k in ks:
                    g = abs(row[k] - w[k]) * 512 / max(3.0, math.sqrt(w[k] * (1 - w[k]) * 512))
                    gaps.append(g)
                    worst_ = max(worst_, (g, f"{k} at sigma {w['sigma']}"))
                gaps.append(abs(row["D"] - w["D"]) / max(abs(w["D"]), 1e-12) / 1e-3)
            exact = new.get(p_) == frozen[p_]
            check(max(gaps) <= 1, f"transfer-full soliton s900 {p_}: {len(v)} cells, "
                  + ("exact" if exact else f"tightest {worst_[1]} at {worst_[0]:.2f} of its bound"))

# clock-law replay: the readout-free overlap at T = 48
if stage("clocklaw") and "results/clocklaw/soliton-s900.json" in by_path:
    frozen = json.load(open(os.path.join(SNAP, "results/clocklaw/soliton-s900.json")))
    f48 = frozen.get("T48", {})
    if len(f48.get("r", [])) != 25 or len(f48.get("norm", [])) != 25 or [round(v, 2) for v in frozen.get("sigma", [])] \
            != GRID["readout"]:
        check(False, "clocklaw soliton s900: the frozen reference lacks the 25-point T = 48 curve")
    else:
        out = os.path.join(TMP, "clocklaw-soliton-s900.json")
        r = run("clocklaw.py", LAB, find("checkpoints", "soliton-T032-s900-step04000.pt"), out, "2", "48")
        new = json.load(open(out)).get("T48", {}) if r.returncode == 0 else {}
        if len(new.get("r", [])) != 25:
            check(False, "clocklaw soliton s900: the replay returned no 25-point T = 48 curve")
        else:
            for i, s_ in enumerate(GRID["readout"]):
                vals = (new["r"][i], f48["r"][i], new["norm"][i], f48["norm"][i])
                ok = finite(*vals) and abs(vals[0] - vals[1]) <= 0.01 and abs(vals[2] - vals[3]) <= 0.01
                check(ok, f"clocklaw soliton s900 T48 sigma {s_:.2f}: r {vals[0]:+.3f} (frozen {vals[1]:+.3f})")
            z0, z1 = new.get("sigma0"), f48.get("sigma0")
            check(finite(z0, z1) and abs(z0 - z1) <= 0.005, f"clocklaw soliton s900 T48: first zero {z0} (frozen {z1})")

print(f"verify {man['snapshot']}: {'FAILED: ' + '; '.join(bad) if bad else 'ok'}")
sys.exit(1 if bad else 0)
