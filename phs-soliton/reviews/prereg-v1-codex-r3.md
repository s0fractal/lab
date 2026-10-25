# PREREG v1 review, round 3 (Codex, read-only, scoped)

Commit reviewed: c97f738. Prompt: [prereg-v1-codex-r3-prompt.txt](prereg-v1-codex-r3-prompt.txt). Reviewer's final message, verbatim.

Confirmed HEAD: `c97f738d06e8e84bac0fbb30e9fe90b12e125938`. No files modified.

1. **CLOSED — finding 2, tag correspondence.** Analysis resolves the registered Git tag, checks the tagged PREREG bytes, and requires manifests and gate to name that commit. Missing and foreign commits are rejected.

2. **CLOSED — finding 5, check-then-import window.** Workers hash and execute the same bytes read once. The importer loaded successfully; replacement training-source bytes were rejected before execution.

3. **CLOSED — finding 7, infinite validation loss.** Checkpoint selection now requires a finite loss. The original extreme-logit counterexample is ineligible.

4. **CLOSED — aggregation crash.** Missing and null diagnostics are excluded from diagnostic means. Probes covering one affected seed and all five affected seeds completed successfully, preserving accuracy and non-finite counts.

No new defects found in these four fixes.

All 30 focused tests passed. Three new registered mutants were killed by assertions using in-memory mutations. A fuller worker smoke check stopped at PyTorch cache initialization under the read-only sandbox.

ACCEPT
