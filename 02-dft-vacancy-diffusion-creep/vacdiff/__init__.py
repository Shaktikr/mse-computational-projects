"""vacdiff - from first-principles point-defect energetics to diffusion-controlled creep.

    defects    supercells, vacancy formation energy, relaxation (QE or EMT)
    neb        nudged-elastic-band (climbing image) vacancy migration barrier
    diffusion  vacancy-mediated self-diffusion coefficient D(T) and Arrhenius analysis
    creep      Nabarro-Herring, Coble and power-law creep rates built on D(T)

The DFT calculator factory is re-used from project 01 (../01-dft-fundamentals-qe/dftlab).
"""

import sys
from pathlib import Path

_DFTLAB = Path(__file__).resolve().parents[2] / "01-dft-fundamentals-qe"
if str(_DFTLAB) not in sys.path:
    sys.path.insert(0, str(_DFTLAB))

__version__ = "0.1.0"
