#!/usr/bin/env bash
# Formation energy of ZnO from Zn + 1/2 O2 with PBE (Quantum ESPRESSO).
#   conda activate dft && ./run_all.sh
# Resumable: finished pw.x runs (outputs containing "JOB DONE") are reused, so after a
# sleep or Ctrl+C just run it again.
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 NPROC=${NPROC:-4}
command -v pw.x >/dev/null || { echo "pw.x not found (conda activate dft)"; exit 1; }
ls pseudo/*.UPF >/dev/null 2>&1 || bash pseudo/make_pseudo.sh
echo "== Convergence of dH_f";     python3 01_convergence.py
echo "== Relaxations";             python3 02_relax.py 60 10
echo "== Formation energy";        python3 03_formation_energy.py
