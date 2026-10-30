# PREREG v1: is Soliton-PHS memory kept and only rotated by noise? One confirmatory test of the clock

> Generated from `prereg.json` by `prereg.py write`; `prereg.py check` refuses a differing file. Version 1, tag `phs-clock-prereg-v1`, written 2026-10-30.

## Status

- Frozen at tag phs-clock-prereg-v1. A change after the tag is a new version with its reason; results of this version stay in the report.
- One bounded confirmatory test, then a close-out: REPORT.md says whether the line is worth continuing, by the continuation rule below.

## Question

phs-soliton PREREG v1 (tag prereg-v1) found the own readout of Soliton-PHS clearly worse than Pure-PHS under noise on task A (mean sigma_crit 0.478 vs 0.820, all 5 Soliton seeds below all 5 Pure seeds), but its one-sided test could not issue REFUTED, and its decay measure lambda confounded trainability at T = 128. The post hoc sandbox fringes/ then found, on exploratory models (seeds 900-902, 4000 steps, T = 32), that noise rotates the Soliton memory rather than erasing it (a clock whose angle grows as sigma^2; accuracy fringes with sigma_2/sigma_1 close to sqrt 2) and that a decoder blind to sigma and to the test labels reads much of it back. This PREREG turns that finding into one confirmatory test on fresh seeds at the v1 budget, with the decoders as frozen in fringes: is the memory kept (H1), does the own readout still lose (H2), and is the clock quadratic (H3)?

## Hypotheses

- **H1.** Kept, only rotated. Read by the honest decoder, the noise threshold of Soliton-PHS is not lower than that of Pure-PHS by the minimal effect or more: Delta_honest = sigma_crit(Soliton) - sigma_crit(Pure) > -0.05.
- **H2.** The own readout loses (replicates v1). Read by each model's own readout, the noise threshold of Soliton-PHS is lower than that of Pure-PHS by more than the minimal effect: Delta_native < -0.05.
- **H3.** The clock is quadratic. On Soliton-PHS runs with a visible revival, the revival lies at x = (sigma_2 / sigma_1)^2 in the interval [1.75, 2.25] around 2 (a phase growing as sigma^2 puts the first minimum at pi and the revival at 2 pi, x = 2; a linear phase gives x = 4).

## Sources

- starter: starter/README.md: the exact commit (80c5192) and paths of the findings this test rests on; no payload is copied.
- v1: phs-soliton/ at tag prereg-v1 (ebaec28): code, PREREG, REPORT (verdict INCONCLUSIVE; residual items 1-2).
- fringes: fringes/README.md at 80c5192, sections 'Спостереження в закомічених даних', 'Механізм на свіжих моделях', 'Годинник можна підвести', 'Чесний декодер', 'Чи переноситься чесний декодер', 'Що не встановлено'; readout.py, transfer.py, fringe_scan.py.
- goal: GOAL.md (in Ukrainian): the task, stages and the done criterion.

## Environment

- python: `3.13.15`
- torch: `2.14.1`
- numpy: `2.5.3`
- lock: `uv.lock (Linux resolves torch from the PyTorch CPU index)`
- machine: `Apple M4 Pro, 14 cores (10 performance + 4 efficiency), 48 GB, macOS (Darwin 27.0.0)`
- device: `cpu, float32, torch.use_deterministic_algorithms(True), 1 thread per run`

## Reused from phs-soliton

Tag `prereg-v1`, commit `ebaec28b2c62a1c436d196aab598f4b99bf3325f`.

- phs-soliton/phs/ (kernel, models, tasks, train.run) unchanged: a run imports it with phs-soliton's own loader (tools/confirm.py load_verified), from bytes checked against the blobs of prereg-v1.
- The model configurations and the optimisation of phs-soliton/prereg.json (pinned by SHA-256), task A: no recalibration. 24000 steps, batch 64, AdamW (weight decay 0.01), clip 1.0, validation every 100 steps on 1024 sequences, checkpoint chosen on validation; training noise sigma drawn from {0, 0.05, ..., 0.35}.
- The test cells of v1 for task A at T = 32 and 64 (4096 sequences per sigma, digests below) and its evaluation of the own readout (noise seed ('testnoise', A, T, sigma, seed)).
- The definition of sigma_crit (threshold 0.8, contiguous from sigma = 0; -0.05 below the grid).

## Models

| kind | model | role | params | config |
|---|---|---|---|---|
| soliton | Soliton-PHS | verdict (H1-H3) | 3562 | `{"N": 32, "S": 4, "batch": 64, "clip": 1.0, "d": 2, "dt": 1.0, "eval_every": 100, "g_lat0": 1.0, "g_pin0": 0.5, "gamma": 0.5, "j_scale": 0.1, "kappa0": 1.0, "lr": 0.01, "m_pin": 2, "n_val": 1024, "steps": 24000, "u_max": 2.0, "weight_decay": 0.01}` |
| pure | Pure-PHS | verdict (H1, H2) | 3560 | `{"N": 32, "S": 4, "batch": 64, "clip": 1.0, "d": 2, "dt": 1.0, "eval_every": 100, "gamma": 0.5, "j_scale": 0.1, "kappa0": 1.0, "lr": 0.01, "n_val": 1024, "steps": 24000, "u_max": 2.0, "weight_decay": 0.01}` |
| gru | Control-GRU | control: reported under the same decoder family, no verdict | 54279 | `{"N": 32, "batch": 64, "clip": 1.0, "d": 2, "eval_every": 100, "lr": 0.007, "n_val": 1024, "steps": 24000, "weight_decay": 0.01}` |

## Design

- task A; models soliton, pure, gru; T [32, 64]
- seeds [300, 301, 302, 303, 304, 305, 306, 307, 308, 309] (never used before: {"phs-soliton confirmatory": [0, 1, 2, 3, 4], "phs-soliton calibration": [100], "fringes": [900, 901, 902], "phs-clock smoke runs": [990]})
- sigma [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0]
- 4096 test sequences per (T, sigma); 60 runs

## Test cells

Task A (selective copy-recall, 7 classes, chance 1/7), 4096 sequences per (T, sigma), sigma 0 .. 2.0 step 0.05: the cells of phs-soliton PREREG v1 (namespace 'test'), SHA-256 of phs.tasks.digest.

| cell | digest |
|---|---|
| A/32/0.00 | `4cf3bada71e3b35aff53b72fcd2f0736e4b4d2f0790165f31904e22a83857ab3` |
| A/32/0.05 | `0414c9b87e892783514a8cd51dbb2c21f272ff0dc501aad840e6e1ee80a4a0ef` |
| A/32/0.10 | `bb846908e910dc492299dcc38969bf77c6aba5b9dd1b9e78d4c37a1bcb93f773` |
| A/32/0.15 | `99b1d7a379a5748e8ef18b117d0ead16992a8dbc06bf5ef625b0e95ca34b32ac` |
| A/32/0.20 | `e3769c6e1bf814c568d1f4fd7612125747764dcbabf4ab29c1a8e69ff6152d96` |
| A/32/0.25 | `0cde5779255a6656c31abb949818e072950509ae129a107feae9c96c2e9e0829` |
| A/32/0.30 | `db6d3f197815b6d96121b204c9820a0a06d066bc4d99f7cb465ac9af76aee8e5` |
| A/32/0.35 | `fa0b96f39e46a862e1b46bc3480609a681243298f687b437f7547e0ee001b30a` |
| A/32/0.40 | `318a8411b4f2a3940c2e3e44273517bdc5604fa01d4a17c07d442ea44510d0f6` |
| A/32/0.45 | `4d025c3b85379b8b49f22c706ed3fef994db131730f38a69b5e7183c2d67d2d4` |
| A/32/0.50 | `6373b8381f47089d5c59beeac4c0331c339006bdbeaff8ea9f4735eb73a37e69` |
| A/32/0.55 | `214c25a243e000ac0fd096bacaa714ef66d3bbf9480c7be02cc7f280c53616b2` |
| A/32/0.60 | `c9452c75598b9be3018989b91807b3ba8cc63076b12e0c3330adde7d08424752` |
| A/32/0.65 | `8c61207a37b97a98c397d8ae787a1d85553996eb55681728d089ad60e5cc6c57` |
| A/32/0.70 | `d726f26008bbf67b68b53bad38b7151f71c72ca20a2877b7e3b68db69f20d59c` |
| A/32/0.75 | `5ca19df0f964928bbf9f096bdb1d359af7c189e74d585360fa6313aa660e3e78` |
| A/32/0.80 | `62f6aae5175e8dc254bb9616d6f8cc9ab909b01f72b6a738c27e0b2c62c98cc8` |
| A/32/0.85 | `2afa58ce25d2b820260296456eb7f8f0a4f6ffd312d6e0671fb41c8534fc9a97` |
| A/32/0.90 | `f3e198a8efea1cdeb0f4d5b81c936d2df60d5f7448ea52b2fbaeaa42e4350779` |
| A/32/0.95 | `006bd17debbc29b6414c23d5db31f587b9f8d13d55e30debe7e48da2b25c4f5a` |
| A/32/1.00 | `a7a27c000622248e976fa4ad24841b836214d166913772d0d6fce518ef73da05` |
| A/32/1.05 | `55c77f0975f4982a38b8135ce684f136cc97af001d4af6e914008941864911d0` |
| A/32/1.10 | `34110487ab11b115281b412e751faec4712f013d5a7a2d4e5002a12686c1e57f` |
| A/32/1.15 | `549f95979841348a0c1154ca8c7f1ebd7bf94c772c4148e6dfdbba936ea4803b` |
| A/32/1.20 | `08712f8722bb1f1af53462e8c377cdfbc0a5c29f4e6f07b084c2576b987252a5` |
| A/32/1.25 | `3f401c800cc8d94f28a525445c4c4e6cc76f65f9e3b22f0e8bc0cbcddc029e70` |
| A/32/1.30 | `534497e4327d4ccce2cf883750b21c2301dce30b22294d7502793d795862d68d` |
| A/32/1.35 | `51a02017aa93e931fff99d7a55b4941139f635393907b524a15344922519f806` |
| A/32/1.40 | `86d49c91756db90b2990852526c01e8efcfba19714ed73b7867a9d2232dc5eec` |
| A/32/1.45 | `daca7557ce46128e3686ff9949ebae816d672895ab3cddd568c983de8d8a1e59` |
| A/32/1.50 | `8e9c94ff580c97c36d8c879f4c366617e23dc705f39992a2df5e57247fba2af1` |
| A/32/1.55 | `30a4d1792e88dd825a17cf2f868b8dd36e757a39567fc8aceba4a3bd53af68cc` |
| A/32/1.60 | `dd4c20fda402b1e9ade3ada515609278294c8c3e39eed59f2fb33c6899ef6d51` |
| A/32/1.65 | `9670eed5323a7007a030acd2e8f3d976cee38e25d32511eadbb12d7ba04ac98c` |
| A/32/1.70 | `9746ceffcd921b35e7a5a8a9cd41586d28406d164e9ebfc7f36460b083a0b1df` |
| A/32/1.75 | `c144df835900eb311dbb80450d30239d3d0ae08bbe8cebc09bf99a2452dc0981` |
| A/32/1.80 | `fb5cbc1b9b4d29cbcae3019cc575e097a7b03be74da9bf295979925d3214af38` |
| A/32/1.85 | `d4578eaec881d272d926e87b3de1a9e6aa7752b67b2dc1d53b60ac91f1a30d3b` |
| A/32/1.90 | `f3bddcd86a8637761f5407ae5e2eba3db918db21b05ac503cd7800d2db8ed0b4` |
| A/32/1.95 | `5ae772277c61edb78017fe0090230a41d9176cea03a5364fd6c0f2c9e5a16902` |
| A/32/2.00 | `10c9070a9a42233728fe85bac23356f7fa444fb57fb37d033224c18d342b79a3` |
| A/64/0.00 | `37c08de52c75d504d111b6dbd994ca9fd9a75ed557102695feec02d57c4a1b21` |
| A/64/0.05 | `a5909a5e498bf277a2b1eaddf792a8d572ef275bed8fd6e6d3ff1a5adf5ba84a` |
| A/64/0.10 | `ae251ce7385d62572496df6437c171371d6097de462dfb5aaa963632b8861fb6` |
| A/64/0.15 | `a5686c0349927e1d050186f9b2617122ba09895d75d2b04a0662e6c6753aa804` |
| A/64/0.20 | `f8eb91d9b7958b88aa174edb5ad40e58943560bf31022b1e053452170e518b2f` |
| A/64/0.25 | `03471064de9c493003cef97de07556a53ad191c66eac42834c9beed2a9f60fa6` |
| A/64/0.30 | `e4459057ae3f89c4e2123af0b907970ce8d697f48b51fb63cfb17b00a7654cd1` |
| A/64/0.35 | `47846198834c15a6fefea5ef9f12caf6b0718cd7479cd47f607b3bd648c150b4` |
| A/64/0.40 | `cad7155b80d6617deaff7db963407aabe9edf7e9b929b7326f902b900d81f244` |
| A/64/0.45 | `53efeedfc9670d8421e4cbdd5ab6fe0abccbed0b1b791088c63031253a7f5099` |
| A/64/0.50 | `809fc2b619fd6ebc3b59bd5113f03e9638af1e066b232d715aea151c8da25725` |
| A/64/0.55 | `2d1b64117670c8b07397d30773458c387f1f0ac9cefe06316813198c2bc8942d` |
| A/64/0.60 | `fc80179c2f2be1ffe2e61e0f221a45d553da14d5e34f6535548e6c4f9e077bc8` |
| A/64/0.65 | `58ded78838b345ad454b3876876ca73497b9d5c403d5bc71264104291c4bbcbb` |
| A/64/0.70 | `f075192f33efc352b738db86fe4d1a9ff47e4860bcf6db1df098459708ce3d95` |
| A/64/0.75 | `74af2edddbb2c283c9ecb97c44964dffe3b5e12a7289d3513414e9535dd97195` |
| A/64/0.80 | `a923f642d87d25ae7ca4d70f280fcf127ccd4d1028e4a2e203878c77b8a908fb` |
| A/64/0.85 | `fda95626dfa418c7c19cfecef9f2c6eb647a80c85faf7012bee81eff9ac4f876` |
| A/64/0.90 | `8945aa1e769470d8fd95ed4cb5c070ed5c4966bdde7b7c1c071f292945e85139` |
| A/64/0.95 | `875f9c45ae6aebaa78ca65794b99f3aff5d0ef3d4e353617507d47cfb37ca318` |
| A/64/1.00 | `8a134f4566efe14f26b0625581872cf732a3c6de268d74357bb67d45468336b8` |
| A/64/1.05 | `919d47c96f78ebd7f32d8301770981d70a9c1d2dbed83dbf6271f53701718d84` |
| A/64/1.10 | `fe779bbe3acbaa0310f248a392ae6e1965345ab44d73c831bc9bd5c82b2d4dc1` |
| A/64/1.15 | `7bcac3f978b274cc9c70c50ab8270395a87dcafe302816e9a5ffb9b81946f7e3` |
| A/64/1.20 | `4be84cc2b57c2459b1a1e1574060d5ed6e61da68c3c93efd5d545976b356ae4e` |
| A/64/1.25 | `04d11e1ece2ccbae3428628ac8246a5479bf8a98b57c584185f1f17675e5c955` |
| A/64/1.30 | `2abbd50509e928092caa9e56f297ea309572b9b62fd60feb3d64dfafc306a306` |
| A/64/1.35 | `3a2ed1011d4f736221beded21b8dc33469fb0956a7d6fe403a3540eddbe26ce2` |
| A/64/1.40 | `313b0328085ce264cb6aa0914916e4162f6813ef5d00f2ed0d7438b09e559987` |
| A/64/1.45 | `19f17c4135f74c457fca5b0d2fcece2335c5c8671fd0f3a46f913741f11bf3dc` |
| A/64/1.50 | `2bee9a4e11c91cdd5655bab89e2c88744283a47a4da32fac7923846d0aa6dee8` |
| A/64/1.55 | `44da98c151d23f52d018912a574efab328327d08fb93ee8e78ad2d666a088f5e` |
| A/64/1.60 | `d5e946c3e3f7aac3ce6db8ab46df806bd03deaadf2548e89d58446566f0cd02b` |
| A/64/1.65 | `67e948c31937ffad76df593caf3d0595e097d74d4deb1e77a3c9bdde33cb1cd3` |
| A/64/1.70 | `bd724eaecbed37187c83164924856fcc4aedd0c1ee61b2594e615c8b6711fc94` |
| A/64/1.75 | `3bfe96a888ef2b3c997c68eaa1ba077f7003926ff841349572c51052f47f63fb` |
| A/64/1.80 | `7dd23454e2a7ba712a3f28ab18be2139ca55d4f4d42b2986f4a1b9fe1cda788f` |
| A/64/1.85 | `3d6ee3804867d68fb23e58ca00217f39a6304e04a8e4f8bf369909ceb0f18aee` |
| A/64/1.90 | `0e871f18a0cb69ce36a5e7ef3b8480bd9dfc8e19adb352f1d942aaa5a0994843` |
| A/64/1.95 | `5833e511a4adf4da2d18920f6972fa92d3892db074230f147d91fd02ed484303` |
| A/64/2.00 | `0e2c974c746db010d6a9bd8eb1352c3b251a59b6515d65a02eb8b0dcf6230a4e` |

## Decoders

- **own readout (native).** The model's trained linear readout on the v1 test cells, all 4096 sequences, as train.run evaluates them (v1's metric).
- **honest, PHS.** fringes/readout.py, run unchanged on the run's checkpoint: calibration cells of namespace 'calib-readout', sigma 0, 0.05, ..., 1.2, 2048 sequences each; the pooled LDA (one LDA on all calibration states, blind to sigma) and the sensor decoder (16 quantile bins of the label-free sensor D = <cos 2 theta> read from the state, one LDA per bin, blind to sigma and to test labels). transfer.py's states() and score(), unchanged, score them on the v1 test inputs (namespace 'test', noise seed ('test-noise', T, sigma) with three decimals), 4096 sequences, on the second half (2048).
- **honest, GRU.** No phases, so no sensor. The pooled LDA only: readout.py's lda_fit on the same calibration cells (seeded as readout.py seeds them), lda_predict on the test cells seeded as transfer.py seeds them, second half.
- **decoder of a run (H1).** PHS: the better of pooled and sensor by sigma_crit on readout.py's own held-out play cells (sigma <= 1.2, scored on one half); a tie takes the sensor. Neither the test cells nor their labels enter the choice. GRU: pooled.
- **oracle.** LDA fit on the first half of each test cell (knows sigma and the test labels): a diagnostic bound, never a decoder of a verdict.
- **check.** In every PHS run, score() on readout.py's play cells must reproduce readout.py's numbers exactly (the decoders scored are the ones readout.py fit), or the run is FAILED.

## Measures

- **gate.** Learnability gate, its own reported measure: own-readout Acc on the test cell sigma = 0 of the run's T >= 0.95. A run that fails it, or FAILED, enters none of the comparisons; pass counts are reported per model and T.
- **sigma_crit.** max{sigma_g : Acc(sigma') >= 0.8 for every grid sigma' <= sigma_g}; -0.05 if Acc(0) < 0.8. Per run: native (own readout) and honest (the run's decoder), plus per decoder.
- **revival.** fringe_scan.py's fringes(), unchanged, on the own-readout curve of a run: after Acc first falls under 1/7, the extrema alternate minimum, peak, ...; sigma_1 = the first minimum; sigma_2 = the first later peak with Acc > 1/7 + 0.03, a peak at the grid edge censored. Visible revival: both exist. x = (sigma_2 / sigma_1)^2.

## Verdict rules

Parameters: `{"acc_threshold": 0.8, "alpha": 0.05, "gate_acc": 0.95, "h3": {"interval": [1.75, 2.25], "min_runs": 6, "revival_margin": 0.03}, "interval_step": 0.005, "margin": 0.05, "min_runs_per_arm_per_T": 4, "sigma_crit_below_grid": -0.05}`

- **test.** Delta = the mean over the entering T of (mean Soliton - mean Pure) over gated runs. A T enters only with at least 4 gated runs of each arm. Exact two-sided permutation test, stratified by T: every relabelling of the arms within each T (C(20, 10)^2 for ten runs per arm and T). The minimal effect is the margin 0.05 (one grid step): the test is of H0: Delta = -0.05, by shifting every Soliton value up by 0.05. The test is exact when, within each T, the two arms differ by a location shift; the interval of shifts it does not reject is reported, not used.
- **H1.** SUPPORTED if p < 0.05 and Delta_honest > -0.05; CONTRADICTED if p < 0.05 and Delta_honest < -0.05; INCONCLUSIVE otherwise (also with no T entering).
- **H2.** SUPPORTED if p < 0.05 and Delta_native < -0.05; CONTRADICTED if p < 0.05 and Delta_native > -0.05; INCONCLUSIVE otherwise.
- **H3.** Over gated Soliton-PHS runs (both T) with a visible revival: exact two-sided sign-flip tests of H0: the x are symmetric about 1.75, and about 2.25 (all 2^n sign vectors). SUPPORTED if both reject (p < 0.05) and 1.75 < mean x < 2.25; CONTRADICTED if the mean lies outside the interval and the test at that end rejects; INCONCLUSIVE otherwise, or with fewer than 6 such runs.
- **multiplicity.** None: each hypothesis has its own verdict, and the continuation rule reads H1 alone.
- **continuation.** H1 SUPPORTED: the line has a usable property (the memory is kept and can be read without sigma at Pure's level or better); REPORT names the next step. H1 CONTRADICTED or INCONCLUSIVE: REPORT recommends closing the line.

## Reported, not in a verdict

- Gate pass counts and Acc(sigma = 0) per model and T; FAILED runs with reasons.
- Per T means of native and honest sigma_crit for every model; the lift honest - native per run.
- Control-GRU: honest (pooled) and native sigma_crit; Soliton vs GRU and Pure vs GRU on both, with the stratified two-sided p at 0 and the non-rejected interval; no verdict (noise channels are not commensurable: PHS gets noise through the thermostat, GRU additively per token).
- Sensitivity of H1: the same margin test with the sensor decoder only and the pooled decoder only, and the oracle bound; not verdicts.
- For H1 and H2: p at 0 and the shifts not rejected by the two-sided test (step 0.005 in [-1, 1]).
- Revival visibility per model and T; x per Soliton run.
- The decoder choices; honest sigma_crit at or above 1.2, the top of the calibration range (counted, not trusted); non-finite states.

## Decisions

- **D1.** Reuse without recalibration: phs/ at prereg-v1, the v1 configurations and budget, the v1 test cells. Only task A, T in {32, 64}, Pure-PHS, Soliton-PHS and Control-GRU; task B, T in {16, 128} and Control-GRU-pm are not run. *Why:* The finding to confirm is on task A at T = 32 (fringes) and was seen in the v1 data at T >= 32; T = 16 censors the revival beyond sigma 2.0 and T = 128 measured trainability, not memory (v1 REPORT, observation 1).
- **D2.** Fresh seeds 300-309: ten per model and T, sixty runs. None was used before (design.used_seeds). *Why:* Ten rather than the minimum eight: for H1 at the 0.05 margin with a per-run spread near 0.05, ten per arm and T give about 0.9 probability of SUPPORTED when the true difference is 0, eight about 0.8; the cost of two more seeds is about two hours.
- **D3.** The honest decoders are readout.py and transfer.py as frozen in fringes, pinned by SHA-256 and executed from those bytes: readout.py whole, transfer.py's states(), lda_fit(), lda_predict() and score(). The calibration range stays readout.py's sigma <= 1.2; the test grid is 0 .. 2.0. *Why:* Extending the calibration would change the decoder. Above 1.2 a state falls in the sensor bins of the lowest calibrated D; this matters only for an honest sigma_crit at 1.2 or above, which is counted.
- **D4.** The decoder of a PHS run for H1 is the better of pooled and sensor on readout.py's own held-out play cells; a tie takes the sensor. *Why:* In fringes the sensor read the crystal (Soliton) and the pooled LDA the glass (Pure: sensor 0.70-0.75 against pooled 0.75): one decoder for both would handicap one arm. The play cells are not the test cells, so the choice uses no test label.
- **D5.** The learnability gate (Acc >= 0.95 at sigma = 0) is its own reported measure; a run that fails it, or FAILED, enters no comparison. A T enters a comparison only with at least four gated runs of each arm. *Why:* A model that did not learn the task has no memory to keep or lose; v1 showed that such runs turn a retention measure into a learnability measure.
- **D6.** Two-sided exact tests at the minimal effect, stratified by T, alpha 0.05 each, no multiplicity correction. *Why:* v1's one-sided test could not refute despite a 5-of-5 separation (v1 REPORT, residual 2). Testing at the margin lets each hypothesis be supported or contradicted. Each verdict stands alone, and the continuation reads H1 only.
- **D7.** H3's interval is [1.75, 2.25] around 2, tested on the mean x by exact sign flips, with at least six runs. *Why:* x = 2^(2/k) for a phase growing as sigma^k: the interval holds k in [1.71, 2.48] and excludes linear (k = 1, x = 4) and cubic (x = 1.59) phases. One grid step in sigma_1 near 0.8 moves x by about 12%, so a single run is not decisive; the post hoc v1 data gave x 1.83-2.04 (twelve runs, T >= 32).
- **D8.** H2 and H3 use the own readout exactly as v1 evaluated it (all 4096 sequences, v1's noise seeds); H1 uses transfer.py's test states (the same inputs, noise seeded by (T, sigma) and common to all models), scored on the second half. *Why:* H2 replicates v1's measure; transfer.py's states() is the frozen honest path, and its score() fits the oracle on the first half.
- **D9.** Control-GRU is decoded by the pooled LDA only and reported without a verdict. *Why:* The sensor needs phases; GRU noise enters in other units, so a GRU comparison says how much any re-calibrated linear decoder lifts a model without a clock, not which memory is better.
- **D10.** The continuation rule reads H1 alone: not SUPPORTED, the line is recommended closed. *Why:* The owner's brief: one test that decides whether the line has a usable property, then a close-out.
- **D11.** No new fringes probes; no change to phs-soliton/ or fringes/. *Why:* The brief, and lab AGENTS.md (a new experiment does not change other directories).

## Cost

- Basis: the v1 manifests of task A under the same 12-process load (Pure-PHS and Soliton-PHS wall 6650-6990 s at T = 32 and 11940-14460 s at T = 64, of which 260-290 s and 490-570 s test evaluation; Control-GRU 283-285 s and 552-556 s), and smoke runs of this runner (seed 990, short training, no load): the honest decoding of a PHS run takes 1.9 times its own test evaluation (365 s against 192 s at T = 32, 700 s against 368 s at T = 64); GRU decoding 12 s at T = 32.
- Per run under load: PHS at T = 32 about 7500 s; Pure-PHS at T = 64 about 13000 s; Soliton-PHS at T = 64 about 15300 s; Control-GRU about 320 s at T = 32 and 620 s at T = 64. Sixty runs: about 4.4e5 worker-seconds (123 worker-hours); over 12 workers, longest first, about 10.5 h wall.
- Memory: peak RSS of one worker with decoding 1.01 GB at T = 32 and 1.01 GB at T = 64 (smoke); v1 training peaked at 0.86 GB at T = 64 and 1.32 GB at T = 128. Twelve workers stay under 20 GB of the 48 GB.

## Execution

- From phs-clock/ at the tag: `../phs-soliton/.venv/bin/python -m unittest discover -s tests -t .`, `../phs-soliton/.venv/bin/python mutants.py`, `../phs-soliton/.venv/bin/python prereg.py check`, then `caffeinate -i ../phs-soliton/.venv/bin/python run.py run --workers 12`, then `../phs-soliton/.venv/bin/python analyze.py`.
- run.py refuses unless the tag exists, prereg.json equals the tagged one, the pinned SHA-256 hold, every file a run reads (phs-clock/ outside results/ and REPORT.md, phs-soliton/phs/, phs-soliton/prereg.json, phs-soliton/tools/confirm.py, the three fringes files) equals the tag with nothing uncommitted, and the configurations equal phs-soliton's. Each worker checks again before it starts.
- Each run writes results/manifests/A-<model>-T<T>-s<seed>.json (status OK or FAILED with its reason, never overwritten) and its validation-selected checkpoint results/checkpoints/A-<model>-T<T>-s<seed>.pt (SHA-256 in the manifest). The fitted decoders go to the ignored runs/decoders/ (readout.py refits them from the checkpoint).
- analyze.py refuses unless its SHA-256, prereg.json and fringe_scan.py match, and every one of the sixty runs has an admissible manifest (one per run, from the tag, registered configuration, environment and test-cell digests, checkpoint hash, complete decoder rows, the play check exact).
- A crashed run is a FAILED manifest, not a gap; a rerun is a new version of this PREREG with its reason.

## Review


## Frozen files (SHA-256)

| file | SHA-256 |
|---|---|
| `phs-clock/analyze.py` | `b1416090c4bf13f3bb1082e8116f31cd39ea35fab5441f874f83215ffbdd9a0d` |
| `phs-clock/decode.py` | `0903337461257b9f3ffc928bc7d03a3331a6d9b60564fe0691c831bae5715f6a` |
| `fringes/readout.py` | `d065e98428fbdc9852d0beb1547cf0b1b4f2613472e8154f67e63a874c77e314` |
| `fringes/transfer.py` | `da01856bbb4751b7b7c0c49bb921fb6ecfb30f92e73fc2ad6bebebdfda5598c7` |
| `fringes/fringe_scan.py` | `a05ab2d1e7c630df8ce96e0ad8c4643249faaeec62fa1fe96641ac0bb920c89f` |
| `phs-soliton/prereg.json` | `8cdffc90c408d047d142684a49ea1ad42bd800dd333f40c6959106bac4910a7d` |
