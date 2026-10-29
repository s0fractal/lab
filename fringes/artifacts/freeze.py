"""Freeze the runs behind the fringes README into a committed snapshot (post hoc play, not a confirmatory record).
Usage: artifacts/freeze.py LAB SNAPSHOT_DIR [REGEX]   (run from fringes/; reads the ignored runs/, writes SNAPSHOT_DIR;
REGEX keeps only the runs/ paths it matches, for a snapshot of one later study)

Every file gets its SHA-256, its role, the command that made it (relative to fringes/, LAB = the lab checkout) and a
status: valid, superseded (a later run replaces it) or partly invalid (named fields wrong, the rest stands). The probe
files drop their zmeans (800 KB each, class means of z_T for dial.py): the dial outputs are kept, and the checkpoints
regenerate zmeans. artifacts/verify.py checks the snapshot and replays results from the frozen weights."""
import glob
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys

import numpy as np
import torch

LAB, OUT = sys.argv[1], sys.argv[2]
ONLY = re.compile(sys.argv[3]) if len(sys.argv) > 3 else None
RUNS = "runs"
CKPT = "{k}-T032-s{s}-step{step}.pt"
F3 = {"fields_invalid": ["phi", "coh", "acc_unwound"],
      "note": "probe.py read z as three blocks of Nd before the Codex review (F3); acc_fixed, acc_probe and zmeans "
              "never unpacked z and stand. Corrected values: probe2-*."}

# (glob under runs/, destination dir, role, made_by(match) -> str, status(match) -> dict or None)
SPECS = [
    (r"(soliton|pure)-T032-s900-step(\d+)\.pt", "checkpoints", "training run, seed 900 (saved every 1000 steps)",
     lambda m: f"train_play.py LAB {m[1]} 32 900 4000 runs", None),
    (r"(soliton|pure)-T032-s(90[12])-step(\d+)\.pt", "checkpoints", "training run, seeds 901-902",
     lambda m: f"train_play.py LAB {m[1]} 32 {m[2]} 4000 runs 1", None),
    (r"homeoarch-s(\d+)-step04000\.pt", "checkpoints", "Soliton trained with the homeostatic reflex",
     lambda m: f"homeo_arch.py LAB {m[1]} 4000 runs 1", None),
    (r"golden-q(\d+)-N34-s900-step04000\.pt", "checkpoints", "golden chain N = 34 (q = 0 control, q = 21 twist)",
     lambda m: f"golden.py LAB {m[1]} 34 900 4000 runs 2", None),
    (r"golden-qp-q21-N34-s(\d+)-step04000\.pt", "checkpoints", "quasi-periodic pinning N = 34",
     lambda m: f"golden.py LAB 21 34 {m[1]} 4000 runs 2 qp", None),
    (r"qat-(soliton|pure)-A(\d+)-s900-step04000\.pt", "checkpoints", "quantisation-aware training of the phase",
     lambda m: f"qat.py LAB {m[1]} {m[2]} 900 4000 runs 1", None),
    (r"js/(soliton|pure)\.json", "exports", "float export for the JS and integer engines",
     lambda m: f"export_js.py LAB runs/{m[1]}-T032-s900-step04000.pt runs/js/{m[1]}.json", None),
    (r"js/golden\.json", "exports", "float export, quasi-periodic pinning",
     lambda m: "export_js.py LAB runs/golden-qp-q21-N34-s900-step04000.pt runs/js/golden.json qp 21", None),
    (r"js/homeo\.json", "exports", "float export, homeostatic forward",
     lambda m: "export_js.py LAB runs/homeoarch-s900-step04000.pt runs/js/homeo.json homeo", None),
    (r"js/(qat-(soliton|pure)-A(\d+))\.json", "exports", "float export of a QAT model",
     lambda m: f"export_js.py LAB runs/qat-{m[2]}-A{m[3]}-s900-step04000.pt runs/js/{m[1]}.json  (qat_eval.sh)", None),
    (r"committed_curves\.json", "results", "Acc(sigma) of committed phs-soliton runs with Acc(0) >= 0.9",
     lambda m: "inline extraction from phs-soliton/results/manifests (tag prereg-v1)", None),
    (r"collapse\.json", "results", "fringe collapse in x = (sigma / sigma_1)^2",
     lambda m: "collapse.py runs/committed_curves.json runs/collapse.json", None),
    (r"probe2-(soliton|pure)-4000\.json", "results", "probe, corrected layout (README: Codex review, F3)",
     lambda m: f"probe.py LAB runs/{m[1]}-T032-s900-step04000.pt runs/probe2-{m[1]}-4000.json 1 4096", None),
    (r"probe-(soliton|pure)-4000\.json", "results", "probe at step 4000 (README items 1, 4)",
     lambda m: "battery.sh 4000", lambda m: F3),
    (r"probe-(soliton|pure)-1000\.json", "results", "probe at step 1000, early version (21 sigmas, no zmeans)",
     lambda m: f"probe.py LAB runs/{m[1]}-T032-s900-step01000.pt runs/probe-{m[1]}-1000.json 1 1024", lambda m: F3),
    (r"probe-(soliton|pure)-s(90[12])\.json", "results", "probe, seeds 901-902 (three-seed table)",
     lambda m: f"seeds.sh 901 902  (probe.py LAB runs/{m[1]}-T032-s{m[2]}-step04000.pt ... 1 2048)", lambda m: F3),
    (r"probez-(soliton|pure)-1000\.json", "results", "probe with zmeans at step 1000",
     lambda m: f"probe.py LAB runs/{m[1]}-T032-s900-step01000.pt runs/probez-{m[1]}-1000.json 1 1024", lambda m: F3),
    (r"probez-homeoarch-900\.json", "results", "probe of the homeostatic architecture",
     lambda m: "probe.py LAB runs/homeoarch-s900-step04000.pt runs/probez-homeoarch-900.json 1 1024", lambda m: F3),
    (r"dial-(soliton|pure)-(1000|4000)\.(json|txt)", "results", "rotation plane of the class signal (README item 1)",
     lambda m: f"dial.py runs/{'probe' if m[2] == '4000' else 'probez'}-{m[1]}-{m[2]}.json runs/dial-{m[1]}-{m[2]}.json"
               f"  ({'battery.sh 4000' if m[2] == '4000' else 'after probe.py'})", None),
    (r"dial-homeoarch-900\.(json|txt)", "results", "rotation plane, homeostatic architecture",
     lambda m: "dial.py runs/probez-homeoarch-900.json runs/dial-homeoarch-900.json", None),
    (r"(time|extra)-(soliton|pure)-(1000|4000)\.json", "results", "probe_time.py / probe_extra.py (README items 2, 3)",
     lambda m: f"battery.sh {m[3]}", None),
    (r"spring-.*-4000\.json", "results", "spring detuning (README item 5)", lambda m: "battery.sh 4000", None),
    (r"map-(soliton|pure)-(kappa|pin|lat|J)-4000\.json", "results", "interference map sigma x factor (item 6)",
     lambda m: f"probe_map.py LAB runs/{m[1]}-T032-s900-step04000.pt {m[2]} runs/map-{m[1]}-{m[2]}-4000.json 512 1", None),
    (r"comp-(soliton|pure)-(kappa|pin|lat)-4000\.json", "results", "Harrison compensation (item 7)",
     lambda m: f"compensate.py LAB runs/{m[1]}-T032-s900-step04000.pt runs/map-{m[1]}-{m[2]}-4000.json "
               f"runs/comp-{m[1]}-{m[2]}-4000.json 1024", None),
    (r"soften-(soliton|pure)-4000\.json", "results", "crystal and glass order parameters (item 8)",
     lambda m: f"probe_soften.py LAB runs/{m[1]}-T032-s900-step04000.pt runs/soften-{m[1]}-4000.json 512 2", None),
    (r"dw-(pin|all|other)-4000\.json", "results", "Debye-Waller compensation (item 9)",
     lambda m: f"dw_compensate.py LAB runs/soliton-T032-s900-step04000.pt runs/soften-soliton-4000.json {m[1]} "
               f"runs/dw-{m[1]}-4000.json 1024", None),
    (r"homeo2-soliton-(\d+)\.json", "results", "homeostatic clock, corrected timing (README: Codex review, F4)",
     lambda m: f"homeostasis.py LAB runs/soliton-T032-s{m[1]}-step04000.pt runs/homeo2-soliton-{m[1]}.json 1024 0.3 1",
     None),
    (r"homeo-soliton-(\d+)\.json", "results", "homeostatic clock, first version",
     lambda m: f"homeostasis.py LAB runs/soliton-T032-s{m[1]}-step04000.pt runs/homeo-soliton-{m[1]}.json 1024 0.3 1",
     lambda m: {"superseded_by": f"homeo2-soliton-{m[1]}.json",
                "note": "the calibration profile lagged one token (Codex review, F4)"}),
    (r"homeo-pure-(\d+)\.json", "results", "Pure baseline on the homeostasis cells (no pinning, nothing to rescale)",
     lambda m: f"homeostasis.py LAB runs/pure-T032-s{m[1]}-step04000.pt runs/homeo-pure-{m[1]}.json 1024 0.3 1 none",
     None),
    (r"homeoarch-s(\d+)\.json", "results", "evaluation written at the end of homeo_arch.py",
     lambda m: f"homeo_arch.py LAB {m[1]} 4000 runs 1", None),
    (r"(golden-q\d+-N34-s900|golden-qp-q21-N34-s\d+)\.json", "results", "evaluation written at the end of golden.py",
     lambda m: "golden.py (same command as the checkpoint)", None),
    (r"qat-(soliton|pure)-A(\d+)-s900\.json", "results", "evaluation written at the end of qat.py",
     lambda m: f"qat.py LAB {m[1]} {m[2]} 900 4000 runs 1", None),
    (r"(qat-eval|intring-sweep)\.txt", "results", "console summaries of integer-ring sweeps",
     lambda m: "qat_eval.sh soliton:6 soliton:7 pure:7 pure:8" if m[1] == "qat-eval" else "intring.py sweep, floor era",
     None),
    (r"bits-round/((?:soliton|pure|homeo|golden)-A(\d+)-B(\d+))\.json", "results/bits-round", "integer ring, round",
     lambda m: f"intring.py LAB runs/js/{m[1].split('-A')[0]}.json {m[2]} {m[3]} 256 runs/bits-round/{m[1]}.json 1",
     None),
    (r"bits-round/sweep\.txt", "results/bits-round", "console log with state hashes", lambda m: "the sweep above", None),
    (r"bits/((?:soliton|pure|homeo|golden)-A(\d+)-B(\d+))\.json", "results/bits", "integer ring, floor (PR #6)",
     lambda m: f"intring.py LAB runs/js/{m[1].split('-A')[0]}.json {m[2]} {m[3]} 256 runs/bits/{m[1]}.json  "
               "(before rnd existed; equals rnd = 0)",
     lambda m: {"note": "floor rounding of phase steps: the -1/2 LSB seam; the PR #6 tables, kept as a correction"}),
    (r"bits/qat-(soliton|pure)-A(\d+)\.json", "results/bits", "QAT model in the integer ring, floor",
     lambda m: "qat_eval.sh soliton:6 soliton:7 pure:7 pure:8  (floor era)", None),
    (r"bits/round-.*\.json", "results/bits", "integer ring, round (first round checks)",
     lambda m: "intring.py LAB runs/js/<export>.json A A 256 <out> 1", None),
    (r"bits/sweep\.txt", "results/bits", "console log with state hashes", lambda m: "the floor sweep above", None),
    (r"js/(soliton|pure)-(s90[12])\.json", "exports", "float export of seeds 901-902 for the rounding study",
     lambda m: f"export_js.py LAB artifacts/pr7-2026-10-28/checkpoints/{m[1]}-T032-{m[2]}-step04000.pt runs/js/{m[0][3:]}",
     None),
    (r"rounding/((soliton|pure)-A(\d+)-B(\d+)-(floor|nearest|nearest\+p|stochastic|feedback))\.json", "results/rounding",
     "rounding control, seed 900 (README: rounding instead of noise)",
     lambda m: f"rounding.py LAB artifacts/pr7-2026-10-28/exports/{m[2]}.json {m[3]} {m[4]} {m[5]} "
               f"runs/rounding/{m[1]}.json 512", None),
    (r"rounding/((soliton|pure)-A(\d+)-B(\d+)-dither([\d.]+))\.json", "results/rounding", "dither-width scan, seed 900",
     lambda m: f"rounding.py LAB artifacts/pr7-2026-10-28/exports/{m[2]}.json {m[3]} {m[4]} dither:{m[5]} "
               f"runs/rounding/{m[1]}.json 512", None),
    (r"rounding/(((soliton|pure)-s90[12])-A(\d+)-B(\d+)-(nearest|stochastic|feedback))\.json", "results/rounding",
     "rounding control, seeds 901-902",
     lambda m: f"rounding.py LAB runs/js/{m[2]}.json {m[4]} {m[5]} {m[6]} runs/rounding/{m[1]}.json 512", None),
    (r"rounding/(cum-(soliton|pure)-A(\d+)-(.+))\.json", "results/rounding", "rerun with the accumulated rounding error",
     lambda m: f"rounding.py LAB artifacts/pr7-2026-10-28/exports/{m[2]}.json {m[3]} {m[3]} "
               f"{m[4].replace('dither', 'dither:')} runs/rounding/{m[1]}.json 512", None),
    (r"readout/(soliton|pure)-s(\d+)\.json", "results/readout", "honest decoders (README: the honest decoder)",
     lambda m: f"readout.py LAB artifacts/pr7-2026-10-28/checkpoints/{m[1]}-T032-s{m[2]}-step04000.pt "
               f"runs/readout/{m[1]}-s{m[2]}.json 2048 1", None),
    (r"transfer/(soliton|pure)-s(\d+)\.json", "results/transfer", "frozen decoders on unseen data (README: transfer)",
     lambda m: f"transfer.py LAB artifacts/pr7-2026-10-28/checkpoints/{m[1]}-T032-s{m[2]}-step04000.pt "
               f"runs/transfer/{m[1]}-s{m[2]}.json 1",
     lambda m: {"superseded_by": "transfer-2026-10-29b", "note": "calibration seeded as 0.700, not readout.py's 0.70: "
                "another decoder than the frozen one (Codex review of #8, 1dec5d4)"}),
    (r"transfer2/(soliton|pure)-s(\d+)\.json", "results/transfer", "the frozen readout.py decoders on unseen data "
     "(README: transfer); calibration checked against artifacts/readout-2026-10-28",
     lambda m: f"transfer.py LAB artifacts/pr7-2026-10-28/checkpoints/{m[1]}-T032-s{m[2]}-step04000.pt "
               f"runs/transfer2/{m[1]}-s{m[2]}.json 1", None),
    (r"clocklaw/(soliton|pure)-s(\d+)\.json", "results/clocklaw", "clock law without a readout (README: transfer)",
     lambda m: f"clocklaw.py LAB artifacts/pr7-2026-10-28/checkpoints/{m[1]}-T032-s{m[2]}-step04000.pt "
               f"runs/clocklaw/{m[1]}-s{m[2]}.json 1", None),
    (r"(train-(?:soliton|pure)(?:-90[12])?|homeoarch-s\d+|golden-(?:q0|q21|qp-s90\d)|qat-(?:soliton|pure)-A\d)\.log",
     "logs", "training log (loss and accuracy every 100 steps)", lambda m: "the training command of the checkpoint",
     None),
]


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    if os.path.exists(OUT):
        sys.exit(f"{OUT} exists: a snapshot is frozen once; write a new one")
    files, cfgs, unmatched = [], {}, []
    for path in sorted(glob.glob(f"{RUNS}/**/*", recursive=True)):
        rel = os.path.relpath(path, RUNS)
        if os.path.isdir(path) or rel.startswith(("ci", "verify")) or "case" in rel or "smoke" in rel:
            continue
        if ONLY and not ONLY.search(rel):
            continue
        for pat, dest, role, made_by, status in SPECS:
            m = re.fullmatch(pat, rel)
            if m:
                break
        else:
            unmatched.append(rel)
            continue
        os.makedirs(os.path.join(OUT, dest), exist_ok=True)
        target = os.path.join(OUT, dest, os.path.basename(rel))
        row = {"path": os.path.relpath(target, OUT), "role": role, "made_by": made_by(m), "from": f"runs/{rel}"}
        if rel.startswith("probe") and os.path.getsize(path) > 100_000:
            d = json.load(open(path))
            for r in d["rows"]:
                r.pop("zmeans", None)
            d["zmeans_dropped"] = "regenerate with the made_by command from the frozen checkpoint"
            json.dump(d, open(target, "w"), indent=1)
            row["derived"] = {"from_sha256": sha(path), "change": "zmeans dropped from every row"}
        else:
            shutil.copyfile(path, target)
        if path.endswith(".pt"):
            c = torch.load(path)
            key = hashlib.sha256(json.dumps(c["cfg"], sort_keys=True).encode()).hexdigest()[:12]
            cfgs[key] = c["cfg"]
            row.update({k: c[k] for k in ("kind", "arch", "ema", "T", "seed", "step") if k in c}, cfg=key)
        if status and status(m):
            row["status"] = status(m)
        row.update(bytes=os.path.getsize(target), sha256=sha(target))
        files.append(row)
    if unmatched:
        print("not frozen (no spec):", *unmatched, sep="\n  ")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=LAB).stdout.strip()
    try:   # there is no /proc on macOS; the CPU name is provenance, not a reason to fail
        with open("/proc/cpuinfo") as f:
            cpu = next((ln.split(":", 1)[1].strip() for ln in f if ln.startswith("model name")), "?")
    except OSError:
        r = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
        cpu = r.stdout.strip() or platform.processor() or platform.machine() or "?"
    manifest = {
        "snapshot": os.path.basename(OUT.rstrip("/")),
        "what": "post hoc fringes runs behind fringes/README.md; not part of the phs-soliton PREREG v1 verdict",
        "source_commit": head,
        "source_note": "the code at this commit; earlier files were made by earlier versions of the same scripts "
                       "(see status for the two corrected measurements)",
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                        "platform": platform.platform(), "cpu": cpu},
        "commands_run_from": "fringes/, with LAB = the lab checkout and the phs-soliton venv (uv sync --frozen)",
        "cfgs": cfgs,
        "files": files,
    }
    json.dump(manifest, open(os.path.join(OUT, "MANIFEST.json"), "w"), indent=1)
    total = sum(f["bytes"] for f in files)
    print(f"{len(files)} files, {total / 1024:.0f} KiB, checkpoints {sum(f['path'].endswith('.pt') for f in files)}")


if __name__ == "__main__":
    main()
