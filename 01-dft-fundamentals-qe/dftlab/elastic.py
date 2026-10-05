"""Cubic elastic constants from energy-strain curves (Mehl's volume-conserving strains).

A cubic crystal has three independent elastic constants C11, C12, C44. Using Voigt notation
with engineering shear strains, the elastic energy density of a small homogeneous strain e is

    U = dE / V0 = 1/2 * e^T C e

We impose two *volume-conserving* strains so that the pressure term drops out:

  1. Orthorhombic:  e = (d, -d, d^2/(1-d^2), 0, 0, 0)   ->  U = (C11 - C12) d^2 + O(d^4)
  2. Monoclinic:    e = (0, 0, d^2/(4-d^2), 0, 0, d)    ->  U = 1/2 C44 d^2 + O(d^4)

and combine them with the bulk modulus from the EOS, B = (C11 + 2 C12)/3:

    C11 = B + 2/3 (C11 - C12)
    C12 = B - 1/3 (C11 - C12)

Reference: M. J. Mehl, Phys. Rev. B 47, 2493 (1993); Mehl et al., in Intermetallic Compounds
vol. 1 (Wiley, 1994) - the same approach was used for B2 NiAl and CoAl.

From C_ij we derive the polycrystalline (Voigt-Reuss-Hill) moduli and the classic ductility
indicators used for high-temperature structural intermetallics:
    Pugh ratio B/G  (> 1.75 ductile-like, < 1.75 brittle-like)
    Cauchy pressure C12 - C44  (> 0 metallic bonding, < 0 directional bonding)
    Zener anisotropy A = 2 C44 / (C11 - C12)  (= 1 for an isotropic crystal)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .config import EV_PER_A3_TO_GPA

KB = 1.380649e-23  # J/K
HBAR = 1.054571817e-34  # J s
H = 6.62607015e-34  # J s
AMU = 1.66053906660e-27  # kg


def deformation_gradient(voigt_strain) -> np.ndarray:
    """Symmetric deformation gradient F = I + eps from a Voigt strain (engineering shears)."""
    e1, e2, e3, e4, e5, e6 = voigt_strain
    eps = np.array(
        [[e1, e6 / 2, e5 / 2],
         [e6 / 2, e2, e4 / 2],
         [e5 / 2, e4 / 2, e3]]
    )
    return np.eye(3) + eps


def orthorhombic_strain(d: float):
    return (d, -d, d * d / (1.0 - d * d), 0.0, 0.0, 0.0)


def monoclinic_strain(d: float):
    return (0.0, 0.0, d * d / (4.0 - d * d), 0.0, 0.0, d)


def strained(atoms, voigt_strain):
    """Return a copy of `atoms` with the cell deformed by the strain (atoms move with it)."""
    at = atoms.copy()
    F = deformation_gradient(voigt_strain)
    at.set_cell(atoms.get_cell() @ F.T, scale_atoms=True)
    return at


def quadratic_coefficient(deltas, energies) -> float:
    """Fit E(d) = c0 + c1 d + c2 d^2 + c3 d^3 + c4 d^4 and return c2 (eV)."""
    coeffs = np.polyfit(np.asarray(deltas), np.asarray(energies), 4)
    return float(coeffs[2])


@dataclass
class CubicElastic:
    C11: float
    C12: float
    C44: float
    B: float

    # --- derived quantities -------------------------------------------------
    @property
    def C_prime(self) -> float:
        return 0.5 * (self.C11 - self.C12)

    @property
    def G_voigt(self) -> float:
        return (self.C11 - self.C12 + 3 * self.C44) / 5.0

    @property
    def G_reuss(self) -> float:
        return 5.0 * (self.C11 - self.C12) * self.C44 / (4 * self.C44 + 3 * (self.C11 - self.C12))

    @property
    def G_hill(self) -> float:
        return 0.5 * (self.G_voigt + self.G_reuss)

    @property
    def E_hill(self) -> float:
        G, B = self.G_hill, self.B
        return 9 * B * G / (3 * B + G)

    @property
    def poisson(self) -> float:
        G, B = self.G_hill, self.B
        return (3 * B - 2 * G) / (2 * (3 * B + G))

    @property
    def pugh_ratio(self) -> float:
        return self.B / self.G_hill

    @property
    def cauchy_pressure(self) -> float:
        return self.C12 - self.C44

    @property
    def zener_anisotropy(self) -> float:
        return 2 * self.C44 / (self.C11 - self.C12)

    def born_stable(self) -> bool:
        """Born mechanical stability criteria for a cubic crystal."""
        return self.C11 - self.C12 > 0 and self.C11 + 2 * self.C12 > 0 and self.C44 > 0

    def debye_temperature(self, mass_amu_per_atom: float, volume_per_atom_A3: float) -> float:
        """Debye temperature from the Hill moduli (Anderson, J. Phys. Chem. Solids 24, 909 (1963)).

        theta_D = (h/k_B) * (3 / (4 pi V_atom))^(1/3) * v_m,
        v_m = [ (2/v_t^3 + 1/v_l^3) / 3 ]^(-1/3),  v_t = sqrt(G/rho),  v_l = sqrt((B + 4G/3)/rho)
        """
        rho = mass_amu_per_atom * AMU / (volume_per_atom_A3 * 1e-30)  # kg/m^3
        G = self.G_hill * 1e9
        B = self.B * 1e9
        vt = math.sqrt(G / rho)
        vl = math.sqrt((B + 4 * G / 3) / rho)
        vm = ((2 / vt**3 + 1 / vl**3) / 3.0) ** (-1.0 / 3.0)
        return H / KB * (3.0 / (4 * math.pi * volume_per_atom_A3 * 1e-30)) ** (1 / 3) * vm

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(
            C_prime=self.C_prime,
            G_hill=self.G_hill,
            E_hill=self.E_hill,
            poisson=self.poisson,
            pugh_ratio_B_over_G=self.pugh_ratio,
            cauchy_pressure=self.cauchy_pressure,
            zener_anisotropy=self.zener_anisotropy,
            born_stable=bool(self.born_stable()),
        )
        return d


def cubic_constants_from_fits(c2_ortho_eV: float, c2_mono_eV: float, V0: float, B_GPa: float) -> CubicElastic:
    """Combine the two quadratic coefficients (eV) with V0 (A^3) and B (GPa)."""
    c11_minus_c12 = c2_ortho_eV / V0 * EV_PER_A3_TO_GPA  # U = (C11-C12) d^2
    c44 = 2.0 * c2_mono_eV / V0 * EV_PER_A3_TO_GPA  # U = 1/2 C44 d^2
    c11 = B_GPa + 2.0 / 3.0 * c11_minus_c12
    c12 = B_GPa - 1.0 / 3.0 * c11_minus_c12
    return CubicElastic(C11=c11, C12=c12, C44=c44, B=B_GPa)


def run_strain_series(atoms, calc_factory, kind: str, deltas=(-0.02, -0.01, 0.0, 0.01, 0.02),
                      use_symmetry: bool = True):
    """Compute E(d) for the 'ortho' or 'mono' strain family. Returns (deltas, energies).

    For a CUBIC crystal both strain families give E(-d) = E(d) exactly (the -d cell is the
    +d cell rotated by 90 degrees about z), so with use_symmetry only d >= 0 is computed
    and mirrored - halving the cost. Set use_symmetry=False for non-cubic test cases.
    """
    d, e, _ = run_strain_series_with_stress(atoms, calc_factory, kind, deltas, use_symmetry)
    return d, e


def run_strain_series_with_stress(atoms, calc_factory, kind: str, deltas=(-0.02, -0.01, 0.0, 0.01, 0.02),
                                  use_symmetry: bool = True):
    """Like run_strain_series but also returns the Voigt stress (eV/A^3) of every cell.

    Under the symmetry operation that maps +d onto -d, the stress component that drives
    each family changes sign (sigma1 <-> sigma2 for 'ortho', sigma6 -> -sigma6 for 'mono'),
    which is applied when mirroring.
    """
    fn = {"ortho": orthorhombic_strain, "mono": monoclinic_strain}[kind]
    deltas = np.asarray(deltas, float)
    cache = {}
    energies, stresses = [], []
    for d in deltas:
        key = round(abs(d), 10) if use_symmetry else round(d, 10)
        if key not in cache:
            at = strained(atoms, fn(key if use_symmetry else d))
            at.calc = calc_factory(at, f"{kind}_{len(cache):02d}")
            cache[key] = (at.get_potential_energy(), at.get_stress(voigt=True))
        e, s = cache[key]
        s = np.array(s, float)
        if use_symmetry and d < 0:
            if kind == "ortho":
                s[[0, 1]] = s[[1, 0]]
            else:
                s[5] = -s[5]
        energies.append(e)
        stresses.append(s)
    return deltas, np.array(energies), np.array(stresses)


def constants_from_stresses(deltas, stresses_ortho, stresses_mono, B_GPa: float) -> "CubicElastic":
    """Stress-strain route (first-order in strain, so far less sensitive to k-point noise):

        ortho e = (d, -d, ~0):  sigma1 - sigma2 = 2 (C11 - C12) d
        mono  e6 = d:           sigma6 = C44 d
    ASE's stress is +dE/(V de), i.e. positive in tension, matching U = 1/2 e.C.e.
    """
    from .config import EV_PER_A3_TO_GPA

    d = np.asarray(deltas, float)
    so = np.asarray(stresses_ortho) * EV_PER_A3_TO_GPA
    sm = np.asarray(stresses_mono) * EV_PER_A3_TO_GPA
    c11_minus_c12 = np.polyfit(d, (so[:, 0] - so[:, 1]) / 2.0, 1)[0]
    c44 = np.polyfit(d, sm[:, 5], 1)[0]
    return CubicElastic(C11=B_GPa + 2 * c11_minus_c12 / 3, C12=B_GPa - c11_minus_c12 / 3, C44=c44, B=B_GPa)
