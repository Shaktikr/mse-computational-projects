#!/usr/bin/env bash
# Run the complete DFT tutorial: convergence -> EOS -> elastic constants -> formation enthalpy.
# Usage:  bash run_all.sh [nprocs]        (~1-2 h on 2 cores, minutes on a cluster node)
set -euo pipefail
cd "$(dirname "$0")"
source tools/env.sh "${1:-2}"
for s in Al Ni NiAl; do
    python3 scripts/01_convergence.py --system "$s"
    python3 scripts/02_eos.py         --system "$s" --npoints 7
    python3 scripts/03_elastic.py     --system "$s"
done
python3 scripts/04_formation_enthalpy.py
python3 scripts/05_summary.py
