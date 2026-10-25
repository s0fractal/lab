# PREREG v1 review, round 2 (Codex, read-only, scoped)

Commit reviewed: 8d70cfd. Prompt: [prereg-v1-codex-r2-prompt.txt](prereg-v1-codex-r2-prompt.txt). Reviewer's final message, verbatim except that absolute paths are shortened to repository paths.

Reviewed HEAD `8d70cfd7cbda3cbf4e54b479ee8a9a7d387657e5`. No files modified.

1. **CLOSED — contradictory `room` definitions.** PREREG now consistently specifies OR, matching implementation and regression test. No new defect found.

2. **OPEN — P1: tag correspondence remains unverified.** [analyze.py:126](analyze.py:126) and [analyze.py:148](analyze.py:148) check agreement between claims, without resolving the registered tag. **Reproduced:** remove `tag_commit` from every otherwise admissible manifest and the gate; admission returns `[]`, and analysis using the current PREREG and 33 registered mutants returns `CONFIRMED`. Giving everything the same foreign commit also passes.

3. **CLOSED — stale/partial mutant matrix.** The gate deletes the previous matrix and requires successful subprocess exit, successful PREREG check, and the complete registered count. Analysis rejects the original failed/partial counterexamples. No new defect found; the separate tag-binding gap is covered by finding 2.

4. **CLOSED — duplicate selection by filename order.** Admission rejects duplicates, unexpected identities, and filenames inconsistent with identity. No new defect found.

5. **OPEN — P2: execution still imports mutable source.** [confirm.py:147](tools/confirm.py:147) checks source before [line 150](tools/confirm.py:150) imports it. **Concrete schedule:** pause after the check succeeds, replace `phs/train.py`, resume its import, then restore the file. The worker executes the replacement and stamps the original tag at [line 159](tools/confirm.py:159). The cooperative lock does not prevent this mutation. This counterexample was assessed statically to preserve read-only operation.

6. **CLOSED — chance-floor lambda advantage.** The registered reference-noise learning floor prevents the original uniformly chance-level Soliton counterexample from confirming. No new defect found.

7. **OPEN — P2: non-finite validation loss remains eligible.** [train.py:162](phs/train.py:162) checks only the non-finite-logit count. **Reproduced:** finite float32 logits with class scores `(+MAX, -MAX, 0, …)` and half the targets in each of the first two classes produce `acc=0.5`, `loss=inf`, `nonfinite=0`. This checkpoint passes the selection condition against the initial best checkpoint.

   **New defect — P2: all-non-finite cells crash aggregation.** [train.py:89](phs/train.py:89) omits diagnostics when no sequences are finite, while [analyze.py:296](analyze.py:296) assumes their presence across seeds. **Reproduced:** seed 0 has ordinary PHS diagnostics; seed 1 has an all-NaN cell reporting zero accuracy and no diagnostics. Admission succeeds, then analysis raises `KeyError: 'r_slip'`.

All 26 focused analysis/evaluation tests passed. Full discovery encountered a read-only temporary-cache error importing the core tests. The mutating runner was not executed.

AMEND
