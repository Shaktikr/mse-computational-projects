#!/usr/bin/env bash
# Vacancy formation -> migration (CI-NEB) -> diffusion -> diffusional creep, for fcc Al.
# Requires project 01 results (DFT lattice parameter and Debye temperature).
# Usage:  bash run_all.sh [nprocs] [element]      (~1-2 h on 2 cores for Al, 2x2x2 cell)
set -euo pipefail
cd "$(dirname "$0")"
source ../01-dft-fundamentals-qe/tools/env.sh "${1:-2}"
EL="${2:-Al}"
python3 scripts/01_vacancy_formation.py    --element "$EL"
python3 scripts/02_vacancy_migration_neb.py --element "$EL" --method midpoint
python3 scripts/03_diffusion_and_creep.py   --element "$EL"
