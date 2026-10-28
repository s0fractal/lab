#!/usr/bin/env bash
# The probe battery on one pair of checkpoints (post hoc play). Usage: battery.sh STEP
set -euo pipefail
cd "$(dirname "$0")"
PY=../phs-soliton/.venv/bin/python
LAB=$(cd .. && pwd)
step=$(printf %05d "$1")
for k in soliton pure; do
  ck=runs/$k-T032-s900-step$step.pt
  $PY -I probe.py "$LAB" "$ck" runs/probe-$k-$1.json 1 4096 2>/dev/null &
  $PY -I probe_time.py "$LAB" "$ck" runs/time-$k-$1.json 1 2048 2>/dev/null &
  $PY -I probe_extra.py "$LAB" "$ck" runs/extra-$k-$1.json 2048 2>/dev/null &
done
wait
for k in soliton pure; do
  $PY -I dial.py runs/probe-$k-$1.json runs/dial-$k-$1.json > runs/dial-$k-$1.txt
done
specs_soliton="pin:1.0 pin:0.6 pin:1.5 lat:0.6 lat:1.5 kappa:0.6 kappa:1.5"
specs_pure="kappa:1.0 kappa:0.6 kappa:1.5"
for k in soliton pure; do
  eval "specs=\$specs_$k"
  for s in $specs; do
    $PY -I probe_spring.py "$LAB" runs/$k-T032-s900-step$step.pt "${s%%:*}" "${s##*:}" runs/spring-$k-${s%%:*}-${s##*:}-$1.json >/dev/null 2>&1 &
  done
  wait
done
echo "battery $1 done"
