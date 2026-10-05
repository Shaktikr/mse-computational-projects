#!/usr/bin/env bash
# Reproduce Ogata, Li & Yip (Science 2002): ideal shear strength + stacking faults of Al and Cu.
#   ./run_all.sh          -> reuse every finished calculation (instant: the outputs are in runs/)
#   ./run_all.sh --fresh  -> move the shipped results to reference_results/ and recompute everything
# Resumable: if the laptop sleeps or you press Ctrl+C, run it again; finished pw.x runs
# (outputs containing "JOB DONE") are skipped.  ~8-10 h on 4 cores, most of it the Cu PAW runs.
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 NPROC=${NPROC:-4}
if [ "$1" == "--fresh" ]; then
  mkdir -p reference_results
  mv runs results_*.json figures reference_results/ 2>/dev/null || true
  # the Al stacking-fault results come from project 10 and are not recomputed here
  cp reference_results/results_*_from_creep_project.json . 2>/dev/null || true
  echo "Shipped results moved to reference_results/ - recomputing from scratch."
fi
command -v pw.x >/dev/null || { echo "pw.x not found (conda activate dft)"; exit 1; }
GU_AL=0.0,0.02,0.05,0.08,0.10,0.12,0.15,0.18,0.20,0.22,0.25,0.28,0.30,0.33,0.36
GR_AL=0.02,0.05,0.08,0.11,0.14,0.17,0.20,0.22,0.24,0.26,0.29,0.32
echo "== Cu bulk (convergence, EOS, elastic constants)";  python3 01_cu_bulk.py all 24
echo "== Cu pseudopotential check (PAW)";               python3 06_cu_paw_check.py
echo "== Al shear, unrelaxed";  DEG=0.04 python3 02_ideal_shear.py Al unrelaxed 24 $GU_AL
echo "== Al shear, relaxed";    DEG=0.04 python3 02_ideal_shear.py Al relaxed 24 $GR_AL
echo "== Cu shear, relaxed";    python3 02_ideal_shear.py CuPAW relaxed 24 0.02,0.05,0.08,0.10,0.12,0.14,0.16,0.18
echo "== Cu shear, unrelaxed";  python3 02_ideal_shear.py CuPAW unrelaxed 24 0.0,0.02,0.05,0.08,0.11,0.14,0.17,0.20,0.23,0.26,0.30
echo "== Cu stacking faults (ANNNI)";      python3 03_sfe.py CuPAW annni 16,24,32
echo "== Cu stacking faults (tilted cell)"; python3 03_sfe.py CuPAW gsfe 12 6 60 0.6
python3 03_sfe.py CuPAW gsfe 12 6 60 0.5 0.0,0.5
python3 03_sfe.py CuPAW gsfe 20 6 60 none 0.0,1.0,0.6
echo "== Analysis";            python3 04_analysis.py
