# REPORT: phs-clock, the confirmatory test of the Soliton clock (PREREG v1)

This is the test of [`GOAL.md`](GOAL.md), built on the findings listed in [`starter/README.md`](starter/README.md) and run under [`PREREG.md`](PREREG.md) (tag `phs-clock-prereg-v1`). Every number comes from [`results/results.json`](results/results.json), which the frozen [`analyze.py`](analyze.py) wrote from the 60 manifests in `results/manifests/`. Statements marked *post hoc* were computed after the verdicts and are not part of them.

## The line: close it

**The line is not worth continuing. I recommend closing it: phs-soliton, fringes and phs-clock.** The continuation rule reads H1 alone (D10), and H1 is CONTRADICTED.

The question was whether Soliton-PHS keeps its memory under noise and only rotates it, so that a decoder blind to σ reads it back at least as well as Pure-PHS's. It does not:

- Under the frozen honest decoder, Soliton's σ_crit is 0.32 below Pure's. The interval of differences the test does not reject is [−0.385, −0.25], so the difference is far past the 0.05 margin.
- At T = 32, no Soliton run reaches the weakest Pure run: 0 of 100 pairs. Soliton's best is 0.95; Pure's weakest is 1.05.
- Even the diagnostic oracle leaves Soliton below Pure, by 0.185 [−0.25, −0.12]. The oracle is an LDA fit on half of each test cell, so it knows σ and the test labels. The tested linear oracle therefore does not close the gap. This is evidence about that fitted LDA, not an upper bound over all linear or nonlinear readers or on the information present in the state.

The registered revival-ratio signature replicates. H3 is SUPPORTED on fresh seeds: the revival sits at the median x = (σ₂/σ₁)² = 1.93, and every Soliton run shows it while no Pure or GRU run does. This does not meet the registered H1 continuation criterion. The honest decoder wins back more for Soliton than for Pure (lift +0.34 against +0.21 at T = 32). Yet Pure's memory, read by the same decoder, stays clearly ahead. H2 replicates v1 under a two-sided test: Soliton's own readout is worse than Pure's by 0.41.

## Verdicts

| | claim | measure | Δ (Soliton − Pure) | p at the margin / p at 0 | not rejected | verdict |
|---|---|---|---|---|---|---|
| H1 | Soliton ≥ Pure, honest decoder | honest σ_crit, gated runs, stratified by T | **−0.3175** | 3.5e-8 / 2.9e-9 | [−0.385, −0.25] | **CONTRADICTED** |
| H2 | Soliton < Pure, own readout (v1 replicated) | native σ_crit | **−0.4075** | 5.9e-11 / 5.9e-11 | [−0.455, −0.36] | **SUPPORTED** |
| H3 | median x in [1.75, 2.25] | x = (σ₂/σ₁)² over Soliton runs with a visible revival | median 1.929, n 20, 18 inside | 4.0e-4 at 1.75 / 3.8e-6 at 2.25 | | **SUPPORTED** |

- **Tests.** For H1 and H2: an exact two-sided permutation test stratified by T, over C(20, 10)² = 34 134 779 536 relabellings, against H0: Δ = −0.05. For H3: exact two-sided sign tests at both ends. Each test runs at α = 0.05 on its own.
- **H1 adequacy holds** (D12). The honest decoder reads Pure better than Pure's own readout does: 1.23 against 1.02 at T = 32, and 0.89 against 0.785 at T = 64. So the CONTRADICTED verdict is not a decoder that reads nothing. The decoder reads well, and it reads Soliton worse.
- **Per T.** H1: −0.36 at T = 32 and −0.275 at T = 64. H2: −0.49 and −0.325.
- **x per Soliton run.** 1.71–2.25. Both values outside the interval (1.71 and 1.72) lie just below 1.75.

## Gate and admission

| | T = 32 | T = 64 |
|---|---|---|
| Soliton-PHS | 10/10 | 10/10 |
| Pure-PHS | 10/10 | 10/10 |
| Control-GRU | 10/10 | 10/10 |

- **Gate.** Every run passes the learnability gate. The lowest Acc(σ = 0) is 0.9934 (Soliton, T = 64, seed 301), so every T stratum enters every comparison with ten runs per arm.
- **Runs.** 60 of 60 runs are OK: 0 FAILED, 0 interrupted attempts, 0 non-finite states.
- **Play check.** It was exact in all 40 PHS runs: `score()` on readout.py's play cells reproduced readout.py's numbers bit for bit.
- **Admission.** It checked the prereg.json hash, the analyze.py hash, the fringe_scan.py hash, the tag commit `37ddbd5` of every manifest, the registered decoder lists and every accuracy.

## Means per model and T (σ_crit)

| model | T | native (own readout) | honest (run's decoder) | pooled only | oracle (diagnostic) | lift honest − native |
|---|---|---|---|---|---|---|
| Soliton-PHS | 32 | 0.53 | 0.87 | 0.64 | 0.995 | +0.34 (0.20–0.40) |
| Soliton-PHS | 64 | 0.46 | 0.615 | 0.515 | 0.815 | +0.155 (0.10–0.30) |
| Pure-PHS | 32 | 1.02 | 1.23 | 1.24 | 1.27 | +0.21 (0.15–0.30) |
| Pure-PHS | 64 | 0.785 | 0.89 | 0.895 | 0.91 | +0.105 (0.05–0.15) |
| Control-GRU | 32 | 0.53 | 0.555 | 0.555 | 0.54 | +0.025 (0–0.05) |
| Control-GRU | 64 | 0.385 | 0.385 | 0.385 | 0.375 | 0.00 (−0.05–0.05) |

- **Decoder choices.** Soliton chose the sensor in 20 of 20 runs. Pure chose the sensor in 18 runs and pooled in 2 (T = 64, seeds 300 and 308). GRU uses pooled only (D9).
- **Pure.** Its honest decoder is close to the oracle: within 0.04 at T = 32 and 0.02 at T = 64.
- **Soliton.** Its honest decoder leaves 0.125 at T = 32 and 0.20 at T = 64 to the oracle. Even with all of that recovered, Soliton would stay below Pure.

## Control-GRU (reported, no verdict)

| comparison | metric | Δ | p at 0 | not rejected |
|---|---|---|---|---|
| Soliton − GRU | honest | +0.2725 | 5.9e-11 | [0.225, 0.325] |
| Soliton − GRU | native | +0.0375 | 0.010 | [0.01, 0.065] |
| Pure − GRU | honest | +0.59 | 5.9e-11 | [0.535, 0.64] |
| Pure − GRU | native | +0.445 | 5.9e-11 | [0.41, 0.48] |

Noise reaches PHS through the thermostat and reaches GRU additively per token, so these numbers do not say which memory is better (D9). They say that re-calibrated linear decoding lifts GRU by at most 0.05, Pure by 0.10–0.21 and Soliton by 0.15–0.34. Under its own readout, Soliton is barely above GRU, as in v1 (G1 there: Δ +0.090).

## Sensitivity of H1 (reported, not verdicts)

| decoder for both arms | Δ | not rejected | margin test |
|---|---|---|---|
| sensor only | −0.315 | [−0.38, −0.245] | p 4.7e-8, the CONTRADICTED side |
| pooled only | −0.49 | [−0.555, −0.425] | p 1.2e-10, the CONTRADICTED side |
| oracle (knows σ and the test labels) | −0.185 | [−0.25, −0.12] | p at 0 4.6e-6 |

No decoder choice changes the direction. The sensor decoder favours Soliton and is the one Soliton chose; with it for both arms, Δ is within 0.003 of H1's.

## Calibration edge

- **The count.** Six honest σ_crit values reach 1.2, the top of the calibration range, or go past it (counted, not trusted, D3). All six are Pure at T = 32: seeds 300, 303, 304, 305, 306 and 308, with values 1.2–1.4. No Soliton run comes near: its maximum is 0.95.
- **Capped check** (*post hoc*, arithmetic only). Cap those six at 1.15, the last point below the edge. Pure's T = 32 mean drops to 1.135 and Δ becomes −0.27, still 0.22 past the −0.05 margin. The verdict does not rest on the values above the calibration range.

## Against v1 and fringes

**v1** (`phs-soliton`, 5 seeds, mean over T ∈ {16, 32, 64, 128}):
- Soliton's native σ_crit was 0.478 against Pure's 0.820.
- v1's one-sided test could not call that a refutation.
- Here, at T = 32 and 64 on fresh seeds, the native gap is 0.49 and 0.325, and H2 is SUPPORTED by a two-sided test. Soliton's own readout is worse than Pure's; that is now a confirmed result, not a suspicion.

**fringes** (post hoc: models trained for 4000 steps, T = 32, seeds 900–902):
- The sensor decoder brought Soliton from 0.43 to 0.58.
- The gap to Pure's best honest decoder was 0.17.
- The decoder that knows σ nearly closed the gap (0.73 against 0.75).
- That near-closure was the hope of this line: the memory is kept and only rotated.

At the verdict budget (24 000 steps), the picture is worse for Soliton, not better:
- Both models learn more. Pure gains the most.
- The honest gap at T = 32 is 0.36.
- Even the oracle leaves 0.275 at T = 32.
- The fringes reading "the memory is there, the readout is wrong" holds only in part. The trained readout leaves a gap (lift +0.34), but the tested readers still recover less than in Pure.

**H3 supports the registered revival-ratio criterion.** On fresh seeds and with the frozen revival scan, x = 1.93 sits in the interval and close to 2, which is the value for a phase ∝ σ². fringes' post hoc v1 data gave 1.83–2.04. The revival is visible in 20 of 20 Soliton runs and in 0 of 40 Pure and GRU runs.

## What this establishes and what it does not

Within this design (task A, T ∈ {32, 64}, the v1 configurations and budget, ten fresh seeds per arm and T), the test establishes four things:
1. Under the tested readers, Soliton-PHS has a lower noise threshold than Pure-PHS: its own readout, the frozen σ-blind decoder, and the fitted linear oracle that knows σ and the test labels.
2. The median registered revival ratio is consistent with quadratic phase scaling (H3). This criterion does not uniquely identify the dynamics or a physical mode.
3. A σ-blind sensor decoder recovers part of the gap left by the native readout, but not enough to reach Pure.
4. Both PHS models are under-read by their own trained readouts. Pure is under-read by 0.10–0.21.

It does not establish:
- **Other decoders or total stored information.** The diagnostic oracle is one fitted linear LDA; it is not a proven upper bound over all linear readers, and nonlinear readers were not tested.
- **Other conditions.** Other tasks, other T, other budgets, or another lattice (m_pin, couplings).
- **Physics or energy.** Anything about hardware, energy or the Landauer limit.
- **Clock as a sensor.** Any use of the clock as a sensor of σ. That would be a different question with its own PREREG, and the fringes D = ⟨cos 2θ⟩ thermometer is post hoc.

## What was run

- **PREREG.** Tag `phs-clock-prereg-v1` = commit `37ddbd5`.
  - prereg.json SHA-256 `be0a6c41f298b93e2c9cf26b06bb3171634728b4678f56147829c22013282714`.
  - analyze.py SHA-256 `d1dd84c2251daefe0844257179d01fca17ffe3ad98f3bf365577ed917e5c128c`.
  - Merged to main in s0fractal/lab#11, after one Codex review round (AMEND, seven findings fixed as named residuals R1–R7).
  - Gate at the tag: 21 tests OK, 28/28 mutants killed, prereg check ok ([`results/gate.log`](results/gate.log)).
- **Environment.**
  - Software: Python 3.13.15, torch 2.14.1, numpy 2.5.3 (phs-soliton's uv.lock).
  - Hardware: Apple M4 Pro (14 cores), 48 GB, macOS.
  - Execution: CPU, float32, deterministic, one thread per run, 12 runs in parallel, the longest first.
- **Grid.** 3 models × task A × T ∈ {32, 64} × seeds 300–309 = 60 runs, all from commit `37ddbd5`.
  - Wall time: 2026-10-30 13:34Z to 23:27Z (9.9 h, against the 10.5 h estimate).
  - Worker time: 118 worker-hours in all.
  - Longest run: 4.06 h (Soliton, T = 64, seed 300).
  - Peak RSS of a worker: 0.97 GB.
- **Results committed.** The 60 manifests (each with its honest-decoder rows and play check), the 60 attempt records and the 60 checkpoints (5 MB). The checkpoints are enough to re-decode without retraining.

### Reproduce

From `phs-clock/` at the tag, with phs-soliton's environment (`uv sync --frozen` in `phs-soliton/`):

```bash
../phs-soliton/.venv/bin/python -m unittest discover -s tests && ../phs-soliton/.venv/bin/python mutants.py && ../phs-soliton/.venv/bin/python prereg.py check
```

```bash
../phs-soliton/.venv/bin/python run.py run --workers 12 && ../phs-soliton/.venv/bin/python analyze.py
```

Only the second command is needed to recompute the verdicts from the committed manifests:

```bash
../phs-soliton/.venv/bin/python analyze.py
```

## Residuals

From the review (PREREG §Review, fixed but not re-reviewed):

- **R1.** H3 as a claim about the median, by sign tests (Codex F1).
- **R2.** The whole-repository source guard (F2).
- **R3.** H1 adequacy (F3).
- **R4.** Decoder-list and accuracy admission (F4).
- **R5.** Attempt records and redo (F5).
- **R6.** Interval bounds (F6).
- **R7.** A FAILED run needs a reason (F7).

From the run:

- **R8.** No run was interrupted. The redo path of R5 therefore never ran in production. It is covered only by the unit tests, mutants M23 and M27 and the one simulated redo before the tag.

## Recommendation

Close the line. phs-soliton keeps its v1 verdict (INCONCLUSIVE). fringes stays a sandbox without a verdict. phs-clock ends here with H1 CONTRADICTED. No further PREREG on "Soliton memory is kept and only rotated" is warranted by these data. Reopening this or a related question (for example the clock as a σ-sensor) needs a new owner decision and a new question; it is not a continuation of this one.
