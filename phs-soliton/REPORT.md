# REPORT: Pure-PHS vs Soliton-PHS vs Control-GRU (PREREG v1)

Experiment of [`GOAL.md`](GOAL.md) on the starter [`starter/port-hamiltonian-soliton-reservoirs.md`](starter/port-hamiltonian-soliton-reservoirs.md), run under [`PREREG.md`](PREREG.md) (tag `prereg-v1`). The numbers below come from [`results/results.json`](results/results.json), written by the frozen [`analyze.py`](analyze.py). Its generated view is [`RESULTS.md`](RESULTS.md). Statements marked *post hoc* were computed after the verdict and are not part of it.

## Verdict: `INCONCLUSIVE`

The frozen rule ends in its last branch: "neither confirmation nor refutation; a new PREREG with more seeds is the remedy".

- **Not CONFIRMED (any form).** A1 fails. Soliton-PHS does not have a higher noise threshold than Pure-PHS on task A: its mean σ_crit is 0.478 against Pure-PHS's 0.820 (Δ = −0.343, one-sided p = 1.0). A2 also fails: Δλ = +0.041 with p = 0.50.
- **Not REFUTED.** REFUTED needs both point differences, Soliton − Pure, below their minimum effects. The σ_crit difference is (−0.343 < 0.05). The λ difference is not: +0.041 is above the 0.02 minimum, although it is far from significant. Room held on both metrics, so the ceiling clause played no part.

What the data do show, outside the frozen verdict: on the noise threshold the comparison points clearly away from the hypothesis. All five Soliton-PHS seeds lie below all five Pure-PHS seeds (Soliton 0.41–0.55, Pure 0.74–0.90; *post hoc* exact one-sided p = 1/252 ≈ 0.004 for Pure > Soliton). The test the PREREG registered is one-sided in the hypothesis's direction, so this evidence cannot produce REFUTED under v1. λ cannot settle the hypothesis's other conjunct, for the reason given under T = 128 below.

## Flags

| flag | what | value | holds |
|---|---|---|---|
| A1 | Soliton > Pure, mean σ_crit (task A) | 0.478 vs 0.820, Δ −0.343, p 1.000 | no |
| A2 | Soliton > Pure, λ (task A, σ 0.10) | −0.155 vs −0.197, Δ +0.041, p 0.500 (λ floor met) | no |
| G1 | Soliton > Control-GRU, mean σ_crit | 0.478 vs 0.388, Δ +0.090, p 0.012 (Holm: reject) | **yes** |
| G2 | Soliton > Control-GRU, λ | −0.155 vs −0.170, Δ +0.015, p 0.345 | no |
| B | Soliton not worse than Pure on task B by > 0.05 | 0.508 vs 0.536, Δ −0.029 | **yes** |
| V | gate (44 tests, 35/35 mutants, PREREG check) and Acc ≥ 0.95 at T 16, σ 0 | Pure 5/5, Soliton 5/5, GRU 5/5 | **yes** |
| room | Pure leaves room (σ_crit ≤ 1.95 or λ ≤ −0.02) | 0.820, −0.197 | yes (both) |
| complete | 120 runs of the verdict models OK | 160/160 OK, 0 FAILED | yes |

On task B, Pure-PHS is above chance (q = 5, chance 0.20). Condition B therefore carries information here: the two PHS models are within 0.03 of each other, and both are far below Control-GRU (0.80).

## What was run

- PREREG: commit `ebaec28` = tag `prereg-v1`. `prereg.json` SHA-256 `8cdffc90c408d047d142684a49ea1ad42bd800dd333f40c6959106bac4910a7d`; `analyze.py` SHA-256 `227f952d175fbee3d1c784f5d8cfd3bf37f6c05a9d3b79443143a1e9734be98b`.
- Environment: Python 3.13.15, torch 2.14.1, numpy 2.5.3 (uv.lock). Hardware: Apple M4 Pro (14 cores), 48 GB, macOS. CPU, float32, deterministic, 1 thread per run, 12 runs in parallel.
- Models:
  - Pure-PHS and Soliton-PHS: (N, d) = (32, 2), dt 1, S 4, γ 0.5, u_max 2, lr 1e-2. Soliton adds m = 2. They differ only in the lattice. Parameters: A 3560/3562, B 2534/2536.
  - Control-GRU: hidden 128, lr 7e-3, 54279/52101 parameters.
  - Control-GRU-pm: hidden 28/25, lr 3e-2, 3479/2455 parameters.
  - Shared settings: 24000 steps, batch 64, AdamW (wd 0.01), clip 1.0.
- Grid: 4 models × 2 tasks × T ∈ {16, 32, 64, 128} × seeds 0–4 = 160 runs. All 160 have an OK manifest (`results/manifests/`), all from commit `ebaec28`. Wall time 16.3 h; peak RSS of one run 1.32 GB; longest run 7.6 h.
- Test cells: 4096 sequences for each (task, T, σ), with σ ∈ {0, 0.05, …, 2.00}. That makes 328 cells, every one pinned by SHA-256 in PREREG and re-checked at admission.

### Reproduce

From `phs-soliton/` at the tag:

```bash
uv sync --frozen && make check
```

One run (task A, Soliton-PHS, T = 16, seed 0):

```bash
.venv/bin/python -c "import json,sys; sys.path.insert(0,'.'); from tools.confirm import model_config; from phs.train import run, setup_determinism; p=json.load(open('prereg.json')); setup_determinism(1); r=run('soliton','A',16,0,model_config(p,'soliton','A'),sigmas=tuple(p['grid']['sigma'])); print(r['cells']['0.00'])"
```

The whole grid and the analysis:

```bash
.venv/bin/python tools/confirm.py gate && .venv/bin/python tools/confirm.py run --workers 12 && make analyze
```

`make analyze` on the committed manifests reproduces `results/results.json`, `RESULTS.md` and `figures/`. `tools/render_results.py --check` refuses a RESULTS.md that is not the rendering of the results.

## Deviations from the PDF

Every item is registered in PREREG with its reason.

1. **Equal state dimension.** D_state = 2Nd for the three verdict models, plus the auxiliary Control-GRU-pm (decision 1).
2. **Integrator.** The gyroscopic term is an exact Cayley rotation, B → G → A → O → A → G → B. The PDF's explicit kick in B diverges (to 1e58 in the starter checks) and biases the Gibbs variance by 4–17%. Unit test (1) at a fixed horizon kills that kick; the PDF's own fixed-step test does not see it (decision 2, starter note 4).
3. **Unit tests.** The PDF's three tests are replaced by five, each burned by named mutants (decision 3).
4. **Noise channels.** PHS gets noise through the O-step; GRU gets σξ once per token. These channels are not commensurable (decision 4).
5. **Task details.** The PDF does not fix several details of the tasks (decision 5). The query token is an extra input token (vocabulary 9, 7 classes). The automaton modulus is q = 5.
6. **Fairness.** Equal optimiser, steps, batch and data stream for every model, and validation-selected checkpoints (decision 6).
7. **Initialisation.** Θ₀ = P₀ = 0. κ, g_lat, g_pin and M are learned through softplus or exp (decision 7).
8. **γ is calibrated, not learned (D8).** A learned γ → 0 switches the thermostat off.
9. **Evaluation grid σ ∈ [0, 2.0] (D14).** The PDF's σ ∈ [0, 0.35] remains the training grid. On the PDF grid alone the mean σ_crit would be Pure 0.270, Soliton 0.290, GRU 0.263: censored, with the order of Pure and Soliton reversed. The −0.34 difference is visible only on the wider grid (*post hoc* comparison; the registered reason was the calibration probes).
10. **Tightenings of the reconstructed rule.** Room (D11), the λ floor (D19), admission and source binding (D20), and the non-finite policy (D21).

## Deviations from PREREG

- **None** in the runs, the manifests, `analyze.py` or the verdict. The runner checked in every worker that the experiment files equal the tag and imported `phs/` from verified bytes.
- **Two view-only fixes** in `tools/render_results.py`, made after the tag. That file is not pinned and computes nothing that enters the verdict.
  1. Heat-map labels: an accuracy of 0.995–0.999 was printed as ".00". It now prints "1".
  2. The room line of RESULTS.md said "and". It now says "or", as the rule (D11) and `analyze.py` compute.

## Observations

These are descriptive and *post hoc*; they are not part of the verdict.

1. **T = 128 measures trainability, not memory decay.** For both PHS models at T = 128, every run whose training diverged stayed at chance on task A, and every run that did not diverge learned the task.
   - Pure-PHS: seed 1 learned (Acc 1.00); seeds 0, 2, 3 and 4 diverged at steps 6454, 12169, 9517 and 5055 and stayed at 0.17–0.19.
   - Soliton-PHS: seeds 0 and 2 learned (0.97, 1.00); seeds 1, 3 and 4 diverged and stayed at 0.14–0.15.
   - Control-GRU never diverged and learned T = 128 in 2 of 5 seeds.

   λ is a least-squares slope over four T. It is therefore bimodal per seed: about 0 when T = 128 was learned, about −0.25 when it was not. λ differences between models mostly count learned T = 128 runs (Pure 1, Soliton 2, GRU 2). That is why A2 and G2 cannot be resolved by this design. More seeds alone would sharpen a learnability rate, not a decay rate. In all, 29 PHS runs diverged (D18 kept their best checkpoints): 7 on task A, all at T = 128, and 22 on task B. One Control-GRU-pm run diverged too.

2. **Soliton-PHS anti-recall.** Above its threshold, Soliton-PHS does not fall to chance (1/7). It falls far below chance: mean accuracy 0.009 at T = 16, σ 1.35; 0.002 at T = 32, σ 0.90; 0.039 at T = 64, σ 0.70. Only at larger σ does it return to chance. Noise in that range maps the stored marker systematically to a wrong class. Pure-PHS and both GRUs decay monotonically to chance.

3. **Memory is lost without phase slips.** In those anti-recall cells R_slip is 0.00–0.04 slips per sequence. Slips accumulate only after accuracy has reached chance (`figures/rslip_vs_acc.png`). The Spearman correlation between R_slip and Acc over the Soliton task-A cells is −0.52, the sign the starter predicted. But the starter's mechanism, "phase slips destroy the soliton memory", is not what fails first here. The barrier ΔE_PN ≈ 2 g_lat was not measured (starter note 8).

4. **Learned physics (medians over runs).**
   - Soliton-PHS kept its lattice: g_lat 1.26, g_pin 0.79. LATTICE_UNUSED in 2 of 40 runs, both on task B.
   - Soliton-PHS has larger κ than Pure-PHS (0.95 vs 0.37) and a narrower mass range (0.39–1.30 vs 0.28–2.47).
   - The gyroscopic coupling ‖J‖₂ grew to about 10.7 in both models (initial ≈ 1.45).

5. **Cost.**
   - Analytic FLOPs per token are similar: PHS 1.34–1.37·10⁵, GRU 1.09·10⁵.
   - Measured C_step differs by about 40×: PHS 3.0–3.2 ms/token against GRU 0.07 ms/token.
   - The difference is the sequential, Python-level sub-step loop (S = 4 BAOAB sub-steps per token).
   - C_step was measured under the 12-process load: it is a relative measure.

6. **Against the parameter-matched GRU.** Soliton-PHS beats Control-GRU-pm on mean σ_crit (0.478 vs 0.245, p 0.004). It does not beat it on λ (p 0.37).

## What is not established

- **The hardware, energy and Landauer claims** of the starter's framing. Nothing here touches physical hardware; the PHS models are simulated in float32 on a CPU.
- **Behaviour outside these two synthetic tasks** (selective recall with distractors, a mod-5 automaton) and outside T ≤ 128, N·d = 64.
- **Power.** Five seeds per model give a smallest one-sided p of 1/252. The λ comparison is underpowered and, by observation 1, mis-specified as a decay measure.
- **Commensurable noise between families.** σ enters PHS through the thermostat and GRU additively per token. G1 compares thresholds in two different noise units. The clean noise comparison is Soliton vs Pure, and there Soliton is worse.
- **Robustness of the calibration choice.**
  - Configurations were selected on one seed (100).
  - Soliton s07 beat its neighbours s10 and s11 on task B alone (validation 0.51 vs 0.21–0.22).
  - Control-GRU-pm learned task A only at lr ≥ 3e-2.
  - Other settings could change the ranking; the same budget of at most 12 configurations was spent per model.
- **Long-T trainability under the equal budget.** No model received a curriculum, more steps or a lower lr at long T. A different optimiser policy could change every T = 128 cell.
- **The Peierls–Nabarro barrier estimate** ΔE_PN ≈ 2 g_lat (not measured; no path named).

## Residual and possible next steps

Each item would be a separate decision of the owner. None is a new front in this goal.

1. **PREREG v2 for the λ conjunct.** Separate trainability from decay before adding seeds: for example, read λ only over runs that learned (a per-run learnability gate), or over T ≤ 64, and register the T = 128 learn rate as its own measure. The frozen rule's own remedy, "more seeds", would mostly sharpen the learn rate.
2. **A two-sided or reverse-direction σ_crit test,** if the owner wants the noise-threshold result to be able to refute. v1 could not issue REFUTED on it despite a 5-of-5 separation.
3. **Stable long-T training for PHS.** Every diverged T = 128 run stayed at chance. Candidates: a lower lr or per-sub-step gradient scaling at long T, or a smaller h. Calibration found S = 2 diverging at (32, 2).
4. **Mechanism of anti-recall.** Which wrong class the noise selects, and how Q_top and the phase configuration move in those cells, given that R_slip stays near 0.
5. **Measure the slip barrier along a named path** (starter note 8). The barrier is untested.

## Package

- **PREREG**:
  - [`prereg.json`](prereg.json) → [`PREREG.md`](PREREG.md), tag `prereg-v1` (`ebaec28`).
  - Review: Codex r1 AMEND, r2 AMEND, r3 ACCEPT ([`reviews/`](reviews/)).
- **Calibration**: every trial in [`calibration/trials.jsonl`](calibration/trials.jsonl) (92 jobs, 5 rounds), probes in [`calibration/probes.jsonl`](calibration/probes.jsonl) (18), summary in [`calibration/SUMMARY.md`](calibration/SUMMARY.md).
- **Mutants**: 35/35 killed on an assertion at the tag ([`results/gate.json`](results/gate.json), [`results/mutants-status.json`](results/mutants-status.json)). Tests: 44.
- **Results**: [`results/results.json`](results/results.json), [`RESULTS.md`](RESULTS.md), [`figures/`](figures/), 160 manifests in [`results/manifests/`](results/manifests/).
- **Pull requests**:
  - s0fractal/lab#1: kernel and tests. Open; self-merge was refused by the auto-mode classifier, so it is left to the owner.
  - s0fractal/lab#2: calibration and PREREG. Stacked on #1.
  - Results: stacked on #2.
  - s0fractal/primitives#18: the mutant-runner instance and lesson. Open.
- **CI**: GitHub Actions runs the starter checks, the tests, the mutants and the PREREG check on Linux.
  - lab#1: green.
  - lab#2: green at `b2889e6`. That run includes the PREREG check, so the 328 test-cell digests recompute identically on x86-64 Linux.
