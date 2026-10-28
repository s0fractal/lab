#!/usr/bin/env bash
# Multi-seed check (post hoc play) on fresh seeds at step 4000: the homeostatic clock on Soliton, the Pure baseline
# on the same cells, and the LDA probe for both. Usage: seeds.sh SEED [SEED ...]
set -euo pipefail
cd "$(dirname "$0")"
PY=../phs-soliton/.venv/bin/python
LAB=$(cd .. && pwd)
for s in "$@"; do
  for k in soliton pure; do
    until [ -e runs/$k-T032-s$s-step04000.pt ]; do sleep 20; done
  done
done
for s in "$@"; do
  $PY -I homeostasis.py "$LAB" runs/soliton-T032-s$s-step04000.pt runs/homeo-soliton-$s.json 1024 0.3 1 >/dev/null 2>&1 &
  $PY -I homeostasis.py "$LAB" runs/pure-T032-s$s-step04000.pt runs/homeo-pure-$s.json 1024 0.3 1 none >/dev/null 2>&1 &
done
wait
for s in "$@"; do
  for k in soliton pure; do
    $PY -I probe.py "$LAB" runs/$k-T032-s$s-step04000.pt runs/probe-$k-s$s.json 1 2048 >/dev/null 2>&1 &
  done
done
wait
echo "seeds $* done"
