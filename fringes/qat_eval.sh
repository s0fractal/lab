#!/usr/bin/env bash
# After qat.py: export each quantisation-trained model and run it in the integer ring at A = B = its training bits.
set -euo pipefail
cd "$(dirname "$0")"
PY=../phs-soliton/.venv/bin/python
LAB=$(cd .. && pwd)
for spec in "$@"; do
  k=${spec%%:*}; a=${spec##*:}
  ck=runs/qat-$k-A$a-s900-step04000.pt
  until [ -e "runs/qat-$k-A$a-s900.json" ]; do sleep 30; done
  $PY -I export_js.py "$LAB" "$ck" runs/js/qat-$k-A$a.json >/dev/null
  $PY -I intring.py "$LAB" runs/js/qat-$k-A$a.json "$a" "$a" 256 runs/bits/qat-$k-A$a.json 2>&1 | tail -1
done
echo "qat eval done"
