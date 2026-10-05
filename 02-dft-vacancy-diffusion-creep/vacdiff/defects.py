"""Point defects in supercells: vacancy formation energy.

    E_f^v = E[N-1 atoms, vacancy, relaxed] - (N-1)/N * E[N atoms, perfect]

Both cells must have the SAME lattice parameter (the DFT equilibrium a0 of the perfect
crystal - never the experimental one) and the same plane-wave cut-off / k-point density
so that numerical errors cancel. Finite-size error decreases with supercell size:
2x2x2 (32 sites) is the classic minimum for fcc metals; check with 3x3x3 (108 sites).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.io import read
from ase.optimize import BFGS

from dftlab.calculators import kgrid_from_spacing, qe_calculator, set_initial_magnetization


def fcc_supercell(symbol: str, a: float, n: int = 2) -> Atoms:
    """n x n x n conventional fcc supercell (4 n^3 atoms)."""
    return bulk(symbol, "fcc", a=a, cubic=True).repeat((n, n, n))


def b2_supercell(sym_a: str, sym_b: str, a: float, n: int = 3) -> Atoms:
    """n x n x n B2 supercell (2 n^3 atoms)."""
    return bulk(f"{sym_a}{sym_b}", "cesiumchloride", a=a).repeat((n, n, n))


def remove_atom(atoms: Atoms, index: int = 0) -> Atoms:
    at = atoms.copy()
    del at[index]
    return at


def nearest_neighbour(atoms: Atoms, index: int = 0) -> int:
    """Index of a nearest neighbour of atom `index` (minimum-image convention)."""
    d = atoms.get_distances(index, range(len(atoms)), mic=True)
    d[index] = np.inf
    return int(np.argmin(d))


# --------------------------------------------------------------------------------------
# Energies with either QE (DFT) or EMT (fast test potential)
# --------------------------------------------------------------------------------------

@dataclass
class Settings:
    calc: str = "qe"  # 'qe' or 'emt'
    ecutwfc: float = 35.0
    kspacing: float = 0.20
    degauss: float = 0.02
    fmax: float = 0.01  # eV/A force criterion for relaxations
    workdir: str = "runs"

    def to_dict(self):
        return asdict(self)


def _qe(atoms: Atoms, directory: Path, s: Settings, relax: bool):
    spin = "Ni" in atoms.get_chemical_symbols() and set(atoms.get_chemical_symbols()) == {"Ni"}
    if spin:
        set_initial_magnetization(atoms)
    extra = {}
    if relax:
        # let pw.x relax the ions itself (re-uses wavefunctions between steps: much cheaper
        # than restarting pw.x for every ASE optimiser step). 1 Ry/bohr = 25.71 eV/A.
        extra = {"calculation": "relax", "forc_conv_thr": s.fmax / 25.711, "etot_conv_thr": 1e-6}
    return qe_calculator(atoms, directory, ecutwfc=s.ecutwfc, kspacing=s.kspacing, degauss=s.degauss,
                         spin_polarized=spin, extra_control=extra)


def energy(atoms: Atoms, s: Settings, tag: str) -> float:
    """Single-point energy (eV)."""
    at = atoms.copy()
    if s.calc == "emt":
        from ase.calculators.emt import EMT
        at.calc = EMT()
    else:
        pwo = Path(s.workdir) / tag / "espresso.pwo"
        if _finished(pwo):  # restart-safe: reuse a completed run
            return float(read(pwo, index=-1).get_potential_energy())
        at.calc = _qe(at, Path(s.workdir) / tag, s, relax=False)
    return float(at.get_potential_energy())


def _finished(pwo: Path) -> bool:
    return pwo.exists() and "JOB DONE" in pwo.read_text(errors="ignore")


def relax_positions(atoms: Atoms, s: Settings, tag: str) -> tuple[Atoms, float]:
    """Relax atomic positions at fixed cell. Returns (relaxed atoms, energy in eV)."""
    at = atoms.copy()
    if s.calc == "emt":
        from ase.calculators.emt import EMT
        at.calc = EMT()
        BFGS(at, logfile=None).run(fmax=s.fmax, steps=500)
        return at, float(at.get_potential_energy())
    pwo = Path(s.workdir) / tag / "espresso.pwo"
    if _finished(pwo):  # restart-safe: a completed relaxation is reused
        final = read(pwo, index=-1)
        return final, float(final.get_potential_energy())
    if pwo.exists():  # interrupted relaxation: continue from the last ionic geometry
        try:
            last = read(pwo, index=-1)
            if len(last) == len(at):
                at.positions = last.positions
                print(f"  resuming {tag} from its last ionic step")
        except Exception:  # noqa: BLE001 - unreadable partial output, start afresh
            pass
    at.calc = _qe(at, Path(s.workdir) / tag, s, relax=True)
    at.get_potential_energy()  # runs pw.x 'relax'
    final = read(pwo, index=-1)
    return final, float(final.get_potential_energy())


@dataclass
class VacancyResult:
    n_sites: int
    E_perfect_eV: float
    E_vac_unrelaxed_eV: float
    E_vac_relaxed_eV: float
    Ef_unrelaxed_eV: float
    Ef_relaxed_eV: float
    relaxation_energy_eV: float
    max_nn_displacement_A: float

    def to_dict(self):
        return asdict(self)


def vacancy_formation(perfect: Atoms, s: Settings, label: str = "") -> tuple[VacancyResult, Atoms]:
    N = len(perfect)
    E_perf = energy(perfect, s, f"{label}perfect")
    vac = remove_atom(perfect, 0)
    E_unrel = energy(vac, s, f"{label}vac_unrelaxed")
    vac_rel, E_rel = relax_positions(vac, s, f"{label}vac_relaxed")
    disp = np.linalg.norm(vac_rel.get_positions() - vac.get_positions(), axis=1)
    res = VacancyResult(
        n_sites=N, E_perfect_eV=E_perf, E_vac_unrelaxed_eV=E_unrel, E_vac_relaxed_eV=E_rel,
        Ef_unrelaxed_eV=E_unrel - (N - 1) / N * E_perf,
        Ef_relaxed_eV=E_rel - (N - 1) / N * E_perf,
        relaxation_energy_eV=E_rel - E_unrel,
        max_nn_displacement_A=float(disp.max()),
    )
    return res, vac_rel


def kpoints_used(atoms: Atoms, s: Settings):
    return kgrid_from_spacing(atoms, s.kspacing)
