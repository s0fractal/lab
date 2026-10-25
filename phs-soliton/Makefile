# Run from phs-soliton/. The venv is .venv (uv sync --frozen).
PY := .venv/bin/python

.PHONY: check starter starter-bytes test mutants prereg-check analyze results-check

check: starter test mutants prereg-check

# exit status 0 iff 12/12 starter checks pass (any platform)
starter:
	$(PY) -I starter/check_starter.py > /dev/null

# byte-identical reference output; recorded on arm64 macOS, last digits differ on x86-64 Linux
starter-bytes:
	$(PY) -I starter/check_starter.py | diff - starter/check_starter.out

test:
	$(PY) -m unittest discover -s tests -t . -v

mutants:
	$(PY) tools/run_mutants.py

# PREREG.md is the rendering of prereg.json; test-cell digests and frozen hashes recompute
prereg-check:
	$(PY) tools/prereg.py check

# S6: the frozen analysis, then the views generated from results/results.json
analyze:
	$(PY) analyze.py
	$(PY) tools/render_results.py

results-check:
	$(PY) tools/render_results.py --check
