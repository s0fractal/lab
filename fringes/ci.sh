#!/usr/bin/env bash
# fringes CI: no checkpoint, no training. The integer ring is held to shared vectors in three runtimes (Python, JS,
# C99 plain and under UBSan) and every negative is shown to bite: the other rounding, doctored transcripts, a binary
# with no output, an empty corpus, an export built to overflow 2^53. Then the selftests of the two measuring tools the
# Codex review corrected (probe.py layout, homeostasis.py timing).
# Usage: fringes/ci.sh [python]   (default: the phs-soliton venv, which pins numpy and torch)
set -euo pipefail
cd "$(dirname "$0")"
PY=${1:-../phs-soliton/.venv/bin/python}
W=runs/ci
rm -rf "$W" && mkdir -p "$W"

"$PY" synth_cases.py "$W/cases"
cc -std=c99 -O2 -Wall -Wextra -Werror c/intring.c -o "$W/intring"
cc -std=c99 -O1 -g -fsanitize=undefined -fno-sanitize-recover=all c/intring.c -o "$W/intring-ubsan"

n=0
for f in "$W"/cases/*.json; do
  "$PY" c/check_c.py "$W/intring" "$f"
  "$PY" c/check_c.py "$W/intring-ubsan" "$f"
  "$PY" c/check_c.py "$W/intring" "$f" --bite
  node web/check_intring.js "$f"
  node web/check_intring.js "$f" --bite
  n=$((n + 1))
done
one=$(ls "$W"/cases/*.json | tail -1)
"$PY" c/check_c.py "$W/intring" "$one" --selftest

refused() {  # the command must fail
  if "$@" > "$W/refused.log" 2>&1; then echo "NOT REFUSED: $*"; cat "$W/refused.log"; exit 1; fi
  echo "refused as it should: $(tail -1 "$W/refused.log")"
}
refused "$PY" c/check_c.py "$(command -v true)" "$one"
# UBSan bite: put back the signed left shift of the bias (Codex review F2); the sanitizer must stop it
sed 's/i64 acc = b\[v\] \* ((i64)1 << F);/i64 acc = b[v] << F;/' c/intring.c > "$W/intring-ub.c"
grep -q "acc = b\[v\] << F;" "$W/intring-ub.c"
cc -std=c99 -O1 -fsanitize=undefined -fno-sanitize-recover=all "$W/intring-ub.c" -o "$W/intring-ub"
refused "$PY" c/check_c.py "$W/intring-ub" "$one"
"$PY" -c "import json, sys; c = json.load(open(sys.argv[1])); c['seqs'] = []; json.dump(c, open(sys.argv[2], 'w'))" \
  "$one" "$W/empty.json"
refused "$PY" c/check_c.py "$W/intring" "$W/empty.json"
refused node web/check_intring.js "$W/empty.json"

"$PY" probe.py --selftest
"$PY" homeostasis.py --selftest "$(cd .. && pwd)"
echo "fringes ci: $n synthetic cases x (C, C+UBSan, C bite, JS, JS bite), transcript, corpus, range and UBSan negatives, selftests: ok"
