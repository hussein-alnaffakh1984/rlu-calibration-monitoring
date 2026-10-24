#!/usr/bin/env bash
# Regenerates every number, table and figure in the paper. About 1-2 h on two cores.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
mkdir -p "${RLU_OUT_DIR:-results}"
for s in 01_prepare 02_changepoints 03_patient_effects 04_monitor 05_pbrtqc 06_volume 07_calibration_log 08_tables_figures; do
  echo "== $s"; $PY "$s.py" | tee "${RLU_OUT_DIR:-results}/log_$s.txt"
done
