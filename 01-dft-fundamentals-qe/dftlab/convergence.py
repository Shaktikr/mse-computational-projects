"""Convergence tests - the first thing to do for ANY new DFT system.

Two numerical parameters control the accuracy (and cost) of a plane-wave calculation:

  * ecutwfc  - kinetic-energy cut-off of the plane-wave basis. Higher = more basis
               functions = more accurate, cost grows roughly as ecut^(3/2).
  * k-points - sampling of the Brillouin zone. Metals need dense meshes because the
               Fermi surface cuts the zone sharply; smearing helps.

We increase each parameter until the total energy per atom changes by less than a
tolerance (1 meV/atom is a common target for energy differences; elastic constants and
phonons usually need tighter settings).
"""

from __future__ import annotations

import numpy as np


def first_converged(values, energies_per_atom, tol_eV: float = 1e-3):
    """Smallest parameter value whose energy is within `tol_eV` of all denser values.

    Using the most converged point as reference is standard practice.
    """
    values = list(values)
    e = np.asarray(energies_per_atom, float)
    ref = e[-1]
    for i, v in enumerate(values):
        if np.all(np.abs(e[i:] - ref) < tol_eV):
            return v
    return values[-1]


def ecut_scan(atoms, calc_factory, ecuts):
    """Energy per atom vs ecutwfc. `calc_factory(atoms, tag, ecutwfc=...)`."""
    out = []
    for ec in ecuts:
        at = atoms.copy()
        at.calc = calc_factory(at, f"ecut_{int(ec)}", ecutwfc=ec)
        out.append(at.get_potential_energy() / len(at))
    return np.array(out)


def kpoint_scan(atoms, calc_factory, kspacings):
    """Energy per atom vs k-point spacing (1/A). `calc_factory(atoms, tag, kspacing=...)`."""
    out = []
    for ks in kspacings:
        at = atoms.copy()
        at.calc = calc_factory(at, f"k_{ks:.3f}", kspacing=ks)
        out.append(at.get_potential_energy() / len(at))
    return np.array(out)
