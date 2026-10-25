# PREREG v1 review, round 1 (Codex, read-only)

Commit reviewed: cbc355f25bd0cef72f70352d9ecfd23d5e7573ac. Prompt: [prereg-v1-codex-prompt.txt](prereg-v1-codex-prompt.txt). Reviewer's final message, verbatim except that absolute paths are shortened to repository paths. Resolutions are recorded in prereg.json (review, decisions D11, D19-D21).

Reviewed `cbc355f25bd0cef72f70352d9ecfd23d5e7573ac`. No files modified. All 30 unit tests and `prereg.py check` passed; counterexamples below used synthetic inputs. The mutating runner was not executed.

1. **P1 — `room` has contradictory definitions.**  
   [prereg.json:570](prereg.json:570) requires sigma room **and** lambda room; D11 at line 670 and [analyze.py:182](analyze.py:182) use **or**. For identical Soliton/Pure results with mean sigma threshold `0.10` and lambda `0`, the implementation returns `REFUTED`; the flags definition requires `INCONCLUSIVE`. This is reproduced. Resolve the normative rule before freezing.

2. **P1 — Manifest admission does not establish correspondence to PREREG.**  
   [analyze.py:269](analyze.py:269) checks only the claimed PREREG hash. It never validates `tag_commit`, configuration, environment, `cell_namespace`, cell digests or sample counts. [confirm.py:166](tools/confirm.py:166) similarly accepts existing manifests based on that hash and skips their filenames. Synthetic manifests naming foreign code, `calib` cells, wrong digests and `n=1` still produced `CONFIRMED`. Validate these fields against the frozen specification before admitting results.

3. **P1 — A failed mutant run can reuse a stale successful matrix.**  
   [confirm.py:147](tools/confirm.py:147) reads an existing `mutants-status.json` even when the current subprocess fails before replacing it. The gate return at line 158 ignores `mut.returncode`; [analyze.py:169](analyze.py:169) ignores both `mutants_exit` and `prereg_check_ok`, and accepts any positive matching mutant counts. Thus an interrupted gate can inherit an earlier or partial matrix and satisfy `V`. Synthetic gates with `mutants_exit=1`, failed prereg-check, or only `1/1` mutants each permitted `CONFIRMED`. Require fresh, source-bound evidence for the complete registered mutant set.

4. **P1 — Duplicate manifests permit selection by filename order.**  
   [analyze.py:95](analyze.py:95) silently overwrites duplicate `(kind, task, T, seed)` entries. Since main reads filenames in sorted order, the lexically last file wins. Appending duplicate Soliton results changed the synthetic verdict from `CONFIRMED` to `REFUTED`; reversing their order restored `CONFIRMED`. An accidentally retained alternate result therefore becomes an undeclared selection mechanism. Reject duplicates and unexpected run identities.

5. **P2 — Execution is not bound to the source checked at startup.**  
   [confirm.py:52](tools/confirm.py:52) checks source once, before creating the pool. Fresh workers subsequently import the live checkout at [confirm.py:108](tools/confirm.py:108), while line 117 stamps the original commit context without another source check. For example, a concurrent gate temporarily mutates `phs/core.py`; a newly spawned worker can import that mutant and emit an apparently frozen manifest. Execute from an immutable source snapshot and prevent gate/run interference.

6. **P2 — The lambda rule rewards performance at the chance floor.**  
   [analyze.py:123](analyze.py:123) measures slope at `sigma=0.10`, but validity at line 167 requires learning only at `sigma=0`. A reproduced counterexample has Soliton accuracy approximately `0.99` through sigma `0.05` at every T, then approximately `1/7` at sigma `0.10` at every T. Pure/GRU reach `0.99` through sigma `0.10` at T16 and chance at longer T. Soliton receives `Δsigma_crit=0.0625`, `Δlambda≈0.254`, both `p=1/252`, and the overall verdict is `CONFIRMED`. Its zero slope represents uniformly failed recall at the reference noise. Register a reference-noise learning requirement or another floor-aware rule before freezing.

7. **P2 — Non-finite evaluation can become ordinary accuracy.**  
   [train.py:72](phs/train.py:72) does not check finite logits or loss before `argmax`. Injected all-NaN logits returned `acc=0.14306640625`, `loss=NaN`, `n=4096` without refusal. At [train.py:153](phs/train.py:153), higher validation accuracy can also select a checkpoint despite NaN loss. Such evaluations can enter an `OK` manifest and influence thresholds or slopes as apparent chance performance. Define and enforce a non-finite validation/test failure policy.

The logged calibration contains 92 trials and 18 probes, all using seed 100 and matching `calib` digests. I found no test-cell leakage in those records or checkpoint selection.

AMEND
