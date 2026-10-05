"""Equation of state (EOS) - equilibrium volume, lattice parameter and bulk modulus.

The third-order Birch-Murnaghan EOS (Birch, Phys. Rev. 71, 809 (1947)):

    E(V) = E0 + (9 V0 B0 / 16) * { [x - 1]^3 * B0' + [x - 1]^2 * [6 - 4 x] },   x = (V0/V)^(2/3)

Fitting E(V) from ~7-9 DFT points around the minimum gives
    V0  equilibrium volume          -> lattice parameter a0
    B0  bulk modulus (stiffness under hydrostatic pressure)
    B0' its pressure derivative (typically 4-5 for metals)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import curve_fit

from .config import EV_PER_A3_TO_GPA


def birch_murnaghan(V, E0, V0, B0, B0p):
    """Third-order Birch-Murnaghan energy (B0 in eV/A^3)."""
    x = (V0 / V) ** (2.0 / 3.0)
    return E0 + 9.0 * V0 * B0 / 16.0 * ((x - 1.0) ** 3 * B0p + (x - 1.0) ** 2 * (6.0 - 4.0 * x))


@dataclass
class EOSResult:
    E0: float  # eV per cell
    V0: float  # A^3 per cell
    B0_GPa: float
    B0_prime: float
    rms_residual_meV: float

    def to_dict(self) -> dict:
        return asdict(self)


def fit_birch_murnaghan(volumes, energies) -> EOSResult:
    """Least-squares fit of E(V). Starting guess from a parabola through the data."""
    V = np.asarray(volumes, float)
    E = np.asarray(energies, float)
    a, b, c = np.polyfit(V, E, 2)  # E ~ aV^2 + bV + c
    V0_guess = -b / (2 * a)
    B0_guess = 2 * a * V0_guess  # B = V d2E/dV2
    p0 = [E.min(), V0_guess, B0_guess, 4.0]
    popt, _ = curve_fit(birch_murnaghan, V, E, p0=p0, maxfev=20000)
    resid = E - birch_murnaghan(V, *popt)
    return EOSResult(
        E0=float(popt[0]),
        V0=float(popt[1]),
        B0_GPa=float(popt[2] * EV_PER_A3_TO_GPA),
        B0_prime=float(popt[3]),
        rms_residual_meV=float(1000 * np.sqrt(np.mean(resid**2))),
    )


def scan_volumes(atoms, calc_factory, strains=np.linspace(-0.03, 0.03, 7)):
    """Isotropically scale the cell by (1+s) in each direction and compute E for each s.

    `calc_factory(atoms, tag)` must return a fresh calculator; `tag` labels the run directory.
    Returns (volumes, energies) per cell.
    """
    volumes, energies = [], []
    cell0 = atoms.get_cell().copy()
    for i, s in enumerate(strains):
        at = atoms.copy()
        at.set_cell(cell0 * (1.0 + s), scale_atoms=True)
        at.calc = calc_factory(at, f"eos_{i:02d}")
        energies.append(at.get_potential_energy())
        volumes.append(at.get_volume())
    return np.array(volumes), np.array(energies)
