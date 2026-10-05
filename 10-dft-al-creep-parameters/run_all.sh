#!/usr/bin/env bash
# Runs the whole Al creep-parameter project in the correct order.
#   ./run_all.sh          -> reuse every finished calculation (instant if outputs are present)
#   ./run_all.sh --fresh  -> move the shipped results to reference_results/ and recompute everything
# Every step is resumable: if the laptop sleeps or you press Ctrl+C, just run the script again;
# finished pw.x runs (outputs containing "JOB DONE") are skipped.
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
export NPROC=${NPROC:-4}           # physical cores (i5-8250U = 4)

if [ "$1" == "--fresh" ]; then
  mkdir -p reference_results
  mv runs results_*.json fig*.png reference_results/ 2>/dev/null || true
  echo "Shipped results moved to reference_results/ - recomputing from scratch."
fi

command -v pw.x >/dev/null || { echo "pw.x not found - activate the conda env first (conda activate dft)"; exit 1; }

step () { echo; echo "=== $(date +%H:%M)  $1"; }
step "1  convergence tests (~1 min)";                 python3 01_convergence.py
step "2  EOS + elastic constants (~30 min)";           python3 02_bulk.py 24 32 40
step "3  vacancy formation, 32 sites (~30 min)";       python3 03_vacancy.py k40
step "4  vacancy migration, saddle (~45 min)";         python3 05_migration.py 30 4 0,0.5
step "5  E_f, E_m on 6^3 and 8^3 k-meshes (~40 min)";  python3 05c_kpoint_check.py 6 8
step "6  GSFE first pass + relaxed geometries (~1.5 h)"; python3 04_gsfe.py
step "7  ANNNI stacking-fault energy (~10 min)";       python3 04c_isf_annni.py 16 24 32 40
step "8  corrected GSFE, nbnd = 40 (~1.5 h)";          python3 04f_gsfe_final.py 16
step "9  write CI-NEB input for a cluster";           python3 05b_write_neb_input.py
step "10 figures + creep analysis";                   python3 06_creep_analysis.py | tee results_summary.txt
echo; echo "Done. Figures: fig1..fig6 .png   Numbers: results_summary.txt"
