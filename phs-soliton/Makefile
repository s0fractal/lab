# Run from phs-soliton/. The venv is .venv (uv sync --frozen).
PY := .venv/bin/python

.PHONY: check starter starter-bytes test mutants

check: starter test mutants

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
