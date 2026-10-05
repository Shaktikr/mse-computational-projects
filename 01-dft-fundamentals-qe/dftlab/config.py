"""Central configuration: where the pseudopotentials live and how pw.x is launched.

Everything can be overridden with environment variables, so the same scripts run on a
laptop (serial pw.x) and on an HPC cluster (mpirun / srun) without editing code:

    export DFTLAB_PW_COMMAND="mpirun -np 40 pw.x"
    export DFTLAB_PSEUDO_DIR=/home/$USER/pseudo
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Directory holding the *.UPF pseudopotential files.
PSEUDO_DIR = Path(os.environ.get("DFTLAB_PSEUDO_DIR", PROJECT_ROOT / "pseudo"))

#: Command used to run pw.x. On a cluster set e.g. "srun pw.x" or "mpirun -np 48 pw.x".
PW_COMMAND = os.environ.get("DFTLAB_PW_COMMAND", "pw.x")

#: Element -> pseudopotential file. These ultrasoft (RRKJUS) PBE potentials were generated
#: locally with ld1.x from the pslibrary 1.0.0 recipes (A. Dal Corso, Comput. Mater. Sci. 95,
#: 337 (2014)); the generation inputs sit next to the UPF files in pseudo/.
#: For production work you may prefer the SSSP library (materialscloud.org/sssp).
PSEUDOPOTENTIALS = {
    "Al": "Al.pbe-n-rrkjus_psl.1.0.0.UPF",  # 3s2 3p1, suggested ecutwfc >= 29 Ry
    "Ni": "Ni.pbe-n-rrkjus_psl.1.0.0.UPF",  # 4s2 3d8, suggested ecutwfc >= 41 Ry
}

#: Settings chosen from scripts/01_convergence.py (results/<system>/convergence.json):
#: total energy converged to ~1 meV/atom (Al: 35 Ry, k-spacing 0.12 1/A gives 0.5/1.0 meV;
#: Ni and NiAl need 60 Ry because of the localised Ni 3d states; k-spacing 0.15 1/A is
#: converged to < 0.1 meV/atom for both). ecutrho = 8 x ecutwfc (ultrasoft potentials).
DEFAULTS = {
    "Al": {"ecutwfc": 35.0, "kspacing": 0.12},
    "Ni": {"ecutwfc": 60.0, "kspacing": 0.15},
    "NiAl": {"ecutwfc": 60.0, "kspacing": 0.15},
}

#: Rydberg -> eV
RY_TO_EV = 13.605693122994
#: eV/A^3 -> GPa
EV_PER_A3_TO_GPA = 160.21766208
