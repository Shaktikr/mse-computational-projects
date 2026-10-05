"""Calculator factories.

`qe_calculator` writes a standard pw.x input through ASE. Read it side by side with the
hand-written inputs in ../inputs/ - every keyword set here appears there with a comment.

`get_calculator("emt")` returns ASE's Effective Medium Theory potential. EMT is NOT DFT; it
is a fast toy potential for Al, Ni, Cu, Ag, Au, Pd, Pt used only to test workflows and run
the unit tests in seconds before spending CPU hours on the real calculation.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from ase import Atoms

from . import config


def kgrid_from_spacing(atoms: Atoms, kspacing: float) -> tuple[int, int, int]:
    """Monkhorst-Pack grid whose spacing between k-points is <= `kspacing` (1/Angstrom).

    The reciprocal lattice vectors b_i include the factor 2*pi, so n_i = ceil(|b_i|/kspacing).
    Using a spacing instead of a fixed grid keeps the sampling density identical for
    cells of different size (primitive cell, strained cell, supercell ...).
    """
    rec = 2.0 * math.pi * atoms.cell.reciprocal()
    return tuple(max(1, math.ceil(np.linalg.norm(b) / kspacing)) for b in rec)


def qe_calculator(
    atoms: Atoms,
    directory: str | Path,
    ecutwfc: float,
    kspacing: float | None = None,
    kpts: tuple[int, int, int] | None = None,
    spin_polarized: bool = False,
    smearing: str = "mv",
    degauss: float = 0.02,
    ecutrho_factor: float = 8.0,
    calculation: str = "scf",
    conv_thr: float = 1e-9,
    extra_system: dict | None = None,
    extra_control: dict | None = None,
):
    """Return an ASE `Espresso` calculator with sensible metallic defaults.

    Parameters mirror the pw.x namelists:
      ecutwfc       plane-wave cut-off for wavefunctions (Ry)
      ecutrho       cut-off for the charge density (Ry) = ecutrho_factor * ecutwfc
      occupations   'smearing' because all systems here are metals
      smearing      'mv' = Marzari-Vanderbilt cold smearing, degauss in Ry
      nspin         2 for ferromagnetic Ni, starting_magnetization sets the initial guess
    """
    from ase.calculators.espresso import Espresso, EspressoProfile

    species = sorted(set(atoms.get_chemical_symbols()))
    pseudos = {s: config.PSEUDOPOTENTIALS[s] for s in species}
    if kpts is None:
        if kspacing is None:
            raise ValueError("give kpts or kspacing")
        kpts = kgrid_from_spacing(atoms, kspacing)

    control = {
        "calculation": calculation,
        "tprnfor": True,  # print forces
        "tstress": True,  # print stress tensor
        "disk_io": "low",
        "pseudo_dir": str(config.PSEUDO_DIR),
    }
    control.update(extra_control or {})
    system = {
        "ecutwfc": ecutwfc,
        "ecutrho": ecutrho_factor * ecutwfc,
        "occupations": "smearing",
        "smearing": smearing,
        "degauss": degauss,
    }
    if spin_polarized:
        # nspin=2: collinear spin-polarised run. ASE converts the atoms' initial magnetic
        # moments into starting_magnetization(i) (a fraction of the valence charge, -1..1),
        # so make sure set_initial_magnetization(atoms) has been called.
        system["nspin"] = 2
        if not any(atoms.get_initial_magnetic_moments()):
            set_initial_magnetization(atoms)
    system.update(extra_system or {})
    electrons = {"conv_thr": conv_thr, "mixing_beta": 0.4}

    profile = EspressoProfile(command=config.PW_COMMAND, pseudo_dir=str(config.PSEUDO_DIR))
    return Espresso(
        profile=profile,
        directory=str(directory),
        pseudopotentials=pseudos,
        input_data={"control": control, "system": system, "electrons": electrons},
        kpts=kpts,
    )


def emt_calculator(*_, **__):
    """Fast EMT test potential (NOT DFT) - accepts and ignores DFT keyword arguments."""
    from ase.calculators.emt import EMT

    return EMT()


def get_calculator(name: str, atoms: Atoms, directory: str | Path, **kwargs):
    """Dispatch on `name` ('qe' or 'emt')."""
    if name == "qe":
        return qe_calculator(atoms, directory, **kwargs)
    if name == "emt":
        return emt_calculator()
    raise ValueError(f"unknown calculator {name!r}; use 'qe' or 'emt'")


def set_initial_magnetization(atoms: Atoms, value: float = 0.4) -> None:
    """Give Ni/Co/Fe atoms a starting magnetisation (fraction of valence charge in pw.x)."""
    atoms.set_initial_magnetic_moments(
        [value if s in ("Ni", "Co", "Fe") else 0.0 for s in atoms.get_chemical_symbols()]
    )


def magnetic(atoms: Atoms) -> bool:
    """Ni-containing elemental cells are run spin polarised (fcc Ni is ferromagnetic).

    B2 NiAl is non-magnetic (Ni d-band is filled by Al electrons), so it is run with nspin=1.
    """
    return set(atoms.get_chemical_symbols()) == {"Ni"}
