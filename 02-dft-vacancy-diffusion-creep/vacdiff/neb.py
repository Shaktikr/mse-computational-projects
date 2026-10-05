"""Vacancy migration barrier with the climbing-image nudged elastic band (CI-NEB).

A vacancy moves when a nearest-neighbour atom jumps into it. Initial state: vacancy at
site 0. Final state: the neighbouring atom j sits at site 0 (so the vacancy is now at j).
The minimum-energy path between the two relaxed end points is discretised into images
connected by springs; the climbing image is pushed up to the saddle point.

    E_m = E(saddle) - E(initial)

References: Henkelman, Uberuaga & Jonsson, J. Chem. Phys. 113, 9901 (2000) (CI-NEB);
Smidstrup et al., J. Chem. Phys. 140, 214106 (2014) (IDPP initial path).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.mep import NEB
from ase.optimize import FIRE

from .defects import Settings, _qe


@dataclass
class NEBResult:
    n_images: int
    energies_eV: list
    Em_eV: float
    saddle_index: int
    converged: bool

    def to_dict(self):
        return asdict(self)


def final_state(perfect: Atoms, jumper: int) -> Atoms:
    """Unrelaxed final state with the same atom ordering as the initial state.

    Initial state = perfect cell with atom 0 deleted. Final state = perfect cell in which
    atom `jumper` has moved onto site 0, then atom 0 deleted - i.e. the vacancy now sits
    on the jumper's old site. Both must be relaxed before the NEB.
    """
    final = perfect.copy()
    final.positions[jumper] = perfect.positions[0]
    del final[0]
    return final


def midpoint_state(perfect: Atoms, jumper: int) -> Atoms:
    """Jumping atom placed half-way between its site and the vacant site 0, and FIXED there.

    For a nearest-neighbour vacancy jump in a pure fcc (or bcc) metal the path is symmetric,
    so the saddle point lies exactly at the midpoint and the force on the jumper vanishes
    there by symmetry. Relaxing all other atoms with the jumper fixed gives the saddle energy
    from ONE relaxation - far cheaper than an NEB. Not valid for asymmetric paths
    (B2 compounds, solute-vacancy pairs, concentrated alloys): use run_neb there.
    """
    from ase.constraints import FixAtoms

    at = perfect.copy()
    d = perfect.get_distance(jumper, 0, mic=True, vector=True)
    at.positions[jumper] = perfect.positions[jumper] + 0.5 * d
    del at[0]
    at.set_constraint(FixAtoms(indices=[jumper - 1]))
    return at


def run_neb(initial: Atoms, final: Atoms, s: Settings, n_images: int = 3, fmax: float = 0.05,
            steps: int = 200, tag: str = "neb") -> NEBResult:
    """Relaxed end points in, CI-NEB out. `n_images` = number of INTERMEDIATE images."""
    images = [initial.copy()] + [initial.copy() for _ in range(n_images)] + [final.copy()]
    neb = NEB(images, climb=True, k=0.1, method="improvedtangent")
    neb.interpolate("idpp", mic=True)

    def attach(img, i):
        if s.calc == "emt":
            from ase.calculators.emt import EMT
            img.calc = EMT()
        else:
            img.calc = _qe(img, Path(s.workdir) / tag / f"image_{i:02d}", s, relax=False)

    for i, img in enumerate(images):
        attach(img, i)
    e_ends = [images[0].get_potential_energy(), images[-1].get_potential_energy()]

    opt = FIRE(neb, logfile=str(Path(s.workdir) / f"{tag}.log") if s.calc == "qe" else None)
    converged = bool(opt.run(fmax=fmax, steps=steps))
    energies = [e_ends[0]] + [img.get_potential_energy() for img in images[1:-1]] + [e_ends[1]]
    energies = np.array(energies) - energies[0]
    i_s = int(np.argmax(energies))
    return NEBResult(n_images=n_images, energies_eV=energies.tolist(), Em_eV=float(energies[i_s]),
                     saddle_index=i_s, converged=converged)
