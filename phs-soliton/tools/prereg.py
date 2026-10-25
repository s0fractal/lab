#!/usr/bin/env python3
"""PREREG.md as a rendering of prereg.json (S4).

    .venv/bin/python tools/prereg.py fill     # compute the test-set digests and the analyze.py hash into prereg.json
    .venv/bin/python tools/prereg.py write    # render PREREG.md
    .venv/bin/python tools/prereg.py check    # PREREG.md is the rendering; digests and hashes recompute

prereg.json is the source: never edit PREREG.md by hand.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
SRC, VIEW = ROOT / "prereg.json", ROOT / "PREREG.md"


def digests(prereg: dict) -> dict:
    from phs import tasks
    n = prereg["tasks"]["n_test"]
    return {f"{task}/{T}/{s:.2f}": tasks.digest(tasks.test_set(task, T, s, n=n))
            for task in ("A", "B") for T in prereg["grid"]["T"] for s in prereg["grid"]["sigma"]}


def frozen(prereg: dict) -> dict:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in prereg["frozen"]}


def table(rows: list, head: list) -> list:
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + \
        ["| " + " | ".join(str(c) for c in r) + " |" for r in rows] + [""]


def render(p: dict) -> str:
    o = [f"# {p['title']}", "",
         f"> Generated from `prereg.json` by `tools/prereg.py write`; `tools/prereg.py check` refuses a differing "
         f"file. Version {p['version']}, tag `{p['tag']}`, written {p['date']}.", ""]
    o += ["## Status", ""] + [f"- {x}" for x in p["status"]] + [""]
    o += ["## Hypothesis", "", p["hypothesis"], ""]
    o += ["## Sources", ""] + [f"- {k}: {v}" for k, v in p["sources"].items()] + [""]
    o += ["## Environment", ""] + [f"- {k}: `{v}`" for k, v in p["environment"].items()] + [""]
    o += ["## Models", ""]
    o += table([[k, m["name"], m["role"], m["calibration_trial"], m["params"]["A"], m["params"]["B"],
                 "`" + json.dumps(m["config"], sort_keys=True) + "`"] for k, m in p["models"].items()],
               ["kind", "model", "role", "calibration trial", "params (A)", "params (B)", "config"])
    o += ["## Optimisation (identical for every model)", "",
          "`" + json.dumps(p["optimisation"], sort_keys=True) + "`", ""]
    o += ["## Tasks", ""] + [f"- **{k}.** {v}" for k, v in p["tasks"]["text"].items()] + [""]
    o += [f"Test cells: {p['tasks']['n_test']} sequences per (task, T, sigma), SHA-256 of `phs.tasks.digest`:", ""]
    o += table([[k, f"`{v}`"] for k, v in p["tasks"]["digests"].items()], ["cell", "digest"])
    o += ["## Grid", "", f"- models: {', '.join(p['grid']['models'])}", f"- T: {p['grid']['T']}",
          f"- sigma: {p['grid']['sigma']}", f"- confirmatory seeds: {p['grid']['seeds']}",
          f"- runs: {len(p['grid']['models']) * 2 * len(p['grid']['T']) * len(p['grid']['seeds'])}", ""]
    o += ["## Seeding", ""] + [f"- {x}" for x in p["seeding"]] + [""]
    o += ["## Metrics", ""] + [f"- **{k}.** {v}" for k, v in p["metrics"].items()] + [""]
    r = p["verdict_rule"]
    o += ["## Verdict rule", "", r["about"], "",
          "Parameters: `" + json.dumps({k: v for k, v in r.items() if k not in ("about", "flags", "verdicts")},
                                       sort_keys=True) + "`", "", "Flags:", ""]
    o += [f"- `{k}`: {v}" for k, v in r["flags"].items()] + ["", "Verdict, first match wins:", ""]
    o += [f"{i}. `{v}` — {c}" for i, (v, c) in enumerate(r["verdicts"], 1)] + [""]
    o += ["## Secondary measures (not in the verdict)", ""] + [f"- {x}" for x in p["secondary"]] + [""]
    o += ["## Decisions", ""]
    o += [f"- **{d['id']}.** {d['text']} *Why:* {d['why']}" for d in p["decisions"]] + [""]
    o += ["## Calibration", ""] + [f"- {x}" for x in p["calibration"]] + [""]
    o += ["## Execution", ""] + [f"- {x}" for x in p["execution"]] + [""]
    o += ["## Review", ""] + [f"- {x}" for x in p["review"]] + [""]
    o += ["## Frozen files", ""] + [f"- `{k}`: `{v}`" for k, v in p["frozen"].items()] + [""]
    return "\n".join(o)


def main(argv: list[str]) -> int:
    p = json.loads(SRC.read_text())
    if argv[:1] == ["fill"]:
        p["tasks"]["digests"] = digests(p)
        p["frozen"] = frozen(p)
        SRC.write_text(json.dumps(p, indent=1, ensure_ascii=False) + "\n")
        print(f"filled {len(p['tasks']['digests'])} digests and {len(p['frozen'])} hashes")
        return 0
    if argv[:1] == ["write"]:
        VIEW.write_text(render(p))
        print("PREREG.md written")
        return 0
    if argv[:1] == ["check"]:
        errs = []
        if not VIEW.exists() or VIEW.read_text() != render(p):
            errs.append("PREREG.md is not the rendering of prereg.json")
        if p["tasks"]["digests"] != digests(p):
            errs.append("a test-cell digest does not recompute")
        if p["frozen"] != frozen(p):
            errs.append("a frozen file differs from its pinned SHA-256")
        if not p["tasks"]["digests"]:
            errs.append("no test cells: an empty scope proves nothing")
        for e in errs:
            print("FAIL", e)
        print("prereg: ok" if not errs else f"prereg: {len(errs)} problems")
        return 1 if errs else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
