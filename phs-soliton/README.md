# phs-soliton

Pure-PHS / Soliton-PHS / Control-GRU: the controlled experiment of [`GOAL.md`](GOAL.md) on the starter in
[`starter/`](starter/port-hamiltonian-soliton-reservoirs.md).

| Path | What it is |
|---|---|
| [`starter/`](starter/) | the PDF (unchanged), its transcription, the numerical checks of its claims |
| [`phs/`](phs/) | `core.py` (PortHamiltonianCore, Cayley gyroscopic step), `models.py`, `tasks.py`, `train.py` |
| [`tests/`](tests/), [`mutants.json`](mutants.json) | the five kernel tests of decision 3, task and verdict-rule tests; every test burned by a named mutant |
| [`calibration/`](calibration/) | S3: every trial (`trials.jsonl`), probes (`probes.jsonl`), [`SUMMARY.md`](calibration/SUMMARY.md) |
| [`prereg.json`](prereg.json) → [`PREREG.md`](PREREG.md) | S4: the frozen design, tag `prereg-v1` |
| [`analyze.py`](analyze.py) | S6: the frozen verdict computation (SHA-256 pinned in `prereg.json`) |
| `results/` | S5-S6: `gate.json`, one manifest per run, `results.json` |
| `RESULTS.md`, `figures/`, `REPORT.md` | generated views and the report |

## Reproduce

From `phs-soliton/`, with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --frozen
```

```bash
make check
```

`make check` runs the starter checks (12/12), the tests and every mutant (each must fail a named test on an
assertion), and `tools/prereg.py check` (PREREG.md is the rendering of prereg.json; the 328 test-cell digests and
the analyze.py hash recompute). `make starter-bytes` compares the starter output byte for byte (arm64 macOS only).

One confirmatory run (here task A, Soliton-PHS, T = 16, seed 0), from a clean checkout of the tag:

```bash
.venv/bin/python -c "import json,sys; sys.path.insert(0,'.'); from tools.confirm import model_config; from phs.train import run, setup_determinism; p=json.load(open('prereg.json')); setup_determinism(1); r=run('soliton','A',16,0,model_config(p,'soliton','A'),sigmas=tuple(p['grid']['sigma'])); print(r['cells']['0.00'])"
```

The whole grid and the verdict:

```bash
.venv/bin/python tools/confirm.py gate && .venv/bin/python tools/confirm.py run --workers 12 && make analyze
```

`tools/confirm.py` refuses unless the experiment files equal the tag and nothing is uncommitted; each run writes
`results/manifests/<task>-<kind>-T<T>-s<seed>.json`, and a stopped grid resumes where it stopped. `make analyze` runs
the frozen `analyze.py` (→ `results/results.json`) and renders `RESULTS.md` and `figures/` from it.
