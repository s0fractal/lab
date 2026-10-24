# Run from phs-soliton/. The venv is .venv (uv sync --frozen).
PY := .venv/bin/python

.PHONY: check starter test mutants

check: starter test mutants

starter:
	$(PY) -I starter/check_starter.py | diff - starter/check_starter.out

test:
	$(PY) -m unittest discover -s tests -t . -v

mutants:
	$(PY) tools/run_mutants.py
