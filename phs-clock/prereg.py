#!/usr/bin/env python3
"""PREREG.md as a rendering of prereg.json. From phs-clock/:

    ../phs-soliton/.venv/bin/python prereg.py fill     # the SHA-256 of the frozen files into prereg.json
    ../phs-soliton/.venv/bin/python prereg.py write    # render PREREG.md
    ../phs-soliton/.venv/bin/python prereg.py check    # PREREG.md is the rendering, hashes and test-cell digests
                                                       # recompute, configurations are phs-soliton's, seeds are fresh

prereg.json is the source: never edit PREREG.md by hand.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent
SRC, VIEW = ROOT / "prereg.json", ROOT / "PREREG.md"


def render(p: dict) -> str:
    o = [f"# {p['title']}", "",
         f"> Generated from `prereg.json` by `prereg.py write`; `prereg.py check` refuses a differing file. "
         f"Version {p['version']}, tag `{p['tag']}`, written {p['date']}.", ""]
    o += ["## Status", ""] + [f"- {x}" for x in p["status"]] + ["", "## Question", "", p["question"], ""]
    o += ["## Hypotheses", ""] + [f"- **{k}.** {v}" for k, v in p["hypotheses"].items()] + [""]
    o += ["## Sources", ""] + [f"- {k}: {v}" for k, v in p["sources"].items()] + [""]
    o += ["## Environment", ""] + [f"- {k}: `{v}`" for k, v in p["environment"].items()] + [""]
    r = p["reuse"]
    o += ["## Reused from phs-soliton", "", f"Tag `{r['tag']}`, commit `{r['commit']}`.", ""]
    o += [f"- {x}" for x in r["what"]] + [""]
    o += ["## Models", "", "| kind | model | role | params | config |", "|---|---|---|---|---|"]
    o += [f"| {k} | {m['name']} | {m['role']} | {m['params']} | `{json.dumps(m['config'], sort_keys=True)}` |"
          for k, m in p["models"].items()] + [""]
    d = p["design"]
    o += ["## Design", "", f"- task {d['task']}; models {', '.join(d['models'])}; T {d['T']}",
          f"- seeds {d['seeds']} (never used before: {json.dumps(d['used_seeds'], ensure_ascii=False)})",
          f"- sigma {d['sigma']}", f"- {d['n_test']} test sequences per (T, sigma); {d['runs']} runs", ""]
    o += ["## Test cells", "", p["test_cells"]["text"], "", "| cell | digest |", "|---|---|"]
    o += [f"| {k} | `{v}` |" for k, v in p["test_cells"]["digests"].items()] + [""]
    o += ["## Decoders", ""] + [f"- **{k}.** {v}" for k, v in p["decoders"].items()] + [""]
    o += ["## Measures", ""] + [f"- **{k}.** {v}" for k, v in p["measures"].items()] + [""]
    rule = p["rule"]
    o += ["## Verdict rules", "", "Parameters: `" + json.dumps({k: v for k, v in rule.items() if not isinstance(v, str)},
                                                                sort_keys=True) + "`", ""]
    o += [f"- **{k}.** {v}" for k, v in rule.items() if isinstance(v, str)] + [""]
    o += ["## Reported, not in a verdict", ""] + [f"- {x}" for x in p["reported"]] + [""]
    o += ["## Decisions", ""]
    for x in p["decisions"]:
        o += [f"- **{x['id']}.** {x['text']} *Why:* {x['why']}"]
    o += ["", "## Cost", ""] + [f"- {x}" for x in p["cost"]] + [""]
    o += ["## Execution", ""] + [f"- {x}" for x in p["execution"]] + [""]
    o += ["## Review", ""] + [f"- {x}" for x in p["review"]] + [""]
    o += ["## Frozen files (SHA-256)", "", "| file | SHA-256 |", "|---|---|"]
    o += [f"| `{k}` | `{v}` |" for k, v in p["frozen"].items()] + [""]
    return "\n".join(o)


def problems(p: dict) -> list:
    out = []
    if VIEW.read_text() != render(p):
        out.append("PREREG.md is not the rendering of prereg.json")
    out += [f"{f} differs from its pinned SHA-256" for f, h in p["frozen"].items()
            if hashlib.sha256((LAB / f).read_bytes()).hexdigest() != h]
    v1 = json.loads((LAB / "phs-soliton" / "prereg.json").read_text())
    for k, m in p["models"].items():
        cfg = dict(v1["optimisation"])
        cfg.update(v1["models"][k]["config"])
        if m["config"] != cfg or m["params"] != v1["models"][k]["params"]["A"]:
            out.append(f"{k}: configuration or parameter count is not phs-soliton's prereg-v1 one")
    used = {s for v in p["design"]["used_seeds"].values() for s in v}
    if used & set(p["design"]["seeds"]):
        out.append(f"seeds used before: {sorted(used & set(p['design']['seeds']))}")
    sys.path.insert(0, str(LAB / "phs-soliton"))
    from phs import tasks
    d = p["design"]
    want = {f"A/{T}/{s:.2f}": tasks.digest(tasks.test_set("A", T, s, n=d["n_test"])) for T in d["T"] for s in d["sigma"]}
    if want != p["test_cells"]["digests"]:
        out.append("test-cell digests do not recompute")
    if d["runs"] != len(d["models"]) * len(d["T"]) * len(d["seeds"]):
        out.append("design.runs is not models x T x seeds")
    return out


def main(argv: list) -> int:
    p = json.loads(SRC.read_text())
    if argv[:1] == ["fill"]:
        p["frozen"] = {f: hashlib.sha256((LAB / f).read_bytes()).hexdigest() for f in p["frozen"]}
        SRC.write_text(json.dumps(p, indent=1, ensure_ascii=False) + "\n")
        return 0
    if argv[:1] == ["write"]:
        VIEW.write_text(render(p))
        return 0
    if argv[:1] == ["check"]:
        found = problems(p)
        for x in found:
            print("problem:", x)
        print("prereg check:", "FAILED" if found else f"ok ({len(p['test_cells']['digests'])} test-cell digests, "
                                                    f"{len(p['frozen'])} frozen files)")
        return 1 if found else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
