"""Index of the ignored runs/ (post hoc play): what each checkpoint, export and result file is, its SHA-256 and the
command that makes it, so a fresh clone knows what to regenerate and can tell whether a regenerated file is the same.
Usage: artifact_index.py LAB > fringes/ARTIFACTS.json   (runs/ci is rebuilt by ci.sh and left out)

The checkpoints themselves stay out of git (lab AGENTS.md: run state is small JSON in the repo or ignored runs/);
publishing them is the owner's call. Producers by file name:"""
import hashlib
import json
import os
import re
import sys

import torch

PRODUCERS = [
    (r"(soliton|pure)-T032-s(\d+)-step\d+\.pt", "train_play.py LAB {1} 32 {2} 4000 runs/"),
    (r"homeoarch-s(\d+)-step\d+\.pt", "homeo_arch.py LAB {1} 4000 runs/"),
    (r"golden-qp-q(\d+)-N(\d+)-s(\d+)-step\d+\.pt", "golden.py LAB {1} {2} {3} 4000 runs/ 1 qp"),
    (r"golden-q(\d+)-N(\d+)-s(\d+)-step\d+\.pt", "golden.py LAB {1} {2} {3} 4000 runs/"),
    (r"qat-(soliton|pure)-A(\d+)-s(\d+)-step\d+\.pt", "qat.py LAB {1} {2} {3} 4000 runs/"),
    (r"js/(soliton|pure|golden|homeo)\.json", "export_js.py LAB runs/<ckpt> runs/js/{1}.json [pin_mode]"),
    (r"js/qat-.*\.json", "export_js.py LAB runs/qat-<...>.pt runs/js/<...>.json  (qat_eval.sh)"),
    (r"js/case2?-(.+)-(\d+)-(\d+)(?:-r(\d))?\.json", "intring.py --case runs/js/{1}.json {2} {3} <out> [rnd]"),
    (r"homeo2-.*\.json", "homeostasis.py LAB runs/<ckpt> <out> 1024 0.3 1  (version 2: D0 timing fixed)"),
    (r"probe2-.*\.json", "probe.py LAB runs/<ckpt> <out> 1 4096  (version 2: readout layout fixed)"),
]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(lab):
    runs = os.path.join(lab, "fringes", "runs")
    files, cfgs = [], {}
    for root, dirs, names in os.walk(runs):
        dirs[:] = sorted(d for d in dirs if os.path.relpath(os.path.join(root, d), runs) != "ci")
        for name in sorted(names):
            path = os.path.join(root, name)
            rel = os.path.relpath(path, runs)
            row = {"path": rel, "bytes": os.path.getsize(path), "sha256": sha(path)}
            for pat, cmd in PRODUCERS:
                m = re.fullmatch(pat, rel)
                if m:
                    row["made_by"] = cmd.format(*([m.group(0)] + list(m.groups())))
                    break
            if name.endswith(".pt"):
                c = torch.load(path)
                key = hashlib.sha256(json.dumps(c.get("cfg"), sort_keys=True).encode()).hexdigest()[:12]
                cfgs[key] = c.get("cfg")
                row.update({k: c[k] for k in ("kind", "arch", "T", "seed", "step", "ema") if k in c}, cfg=key)
            files.append(row)
    pt = [f for f in files if f["path"].endswith(".pt")]
    json.dump({"what": "fringes/runs (ignored): SHA-256 and producer of each file; checkpoints are not in git",
               "torch": torch.__version__, "checkpoints": len(pt), "checkpoint_bytes": sum(f["bytes"] for f in pt),
               "cfgs": cfgs, "files": files}, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main(sys.argv[1])
