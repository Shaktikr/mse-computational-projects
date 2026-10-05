"""Crystal structures used in the tutorials.

Primitive cells are used for bulk properties because they contain the fewest atoms
(cheapest DFT). The conventional cubic cells are used when building supercells for
defect calculations (see project 02).
"""

from __future__ import annotations

from ase import Atoms
from ase.build import bulk


def fcc(symbol: str, a: float, cubic: bool = False) -> Atoms:
    """Face-centred cubic metal (Al, Ni, Cu ...). 1 atom primitive, 4 atoms cubic."""
    return bulk(symbol, "fcc", a=a, cubic=cubic)


def bcc(symbol: str, a: float, cubic: bool = False) -> Atoms:
    """Body-centred cubic metal (Fe, Cr, Mo, W, Nb ...). 1 atom primitive, 2 atoms cubic."""
    return bulk(symbol, "bcc", a=a, cubic=cubic)


def b2(symbol_a: str, symbol_b: str, a: float) -> Atoms:
    """Ordered B2 (CsCl-type) intermetallic, e.g. NiAl, CoAl, FeAl.

    Two atoms in a simple cubic cell: A at (0,0,0) and B at (1/2,1/2,1/2).
    """
    return bulk(f"{symbol_a}{symbol_b}", "cesiumchloride", a=a)


#: Starting lattice parameters (Angstrom) - experimental room-temperature values,
#: used only as the centre of the EOS scan.
EXPERIMENTAL_A0 = {
    "Al": 4.05,
    "Ni": 3.52,
    "NiAl": 2.887,
}


def build(system: str, a: float | None = None) -> Atoms:
    """Build one of the tutorial systems by name: 'Al', 'Ni' or 'NiAl'."""
    a = a or EXPERIMENTAL_A0[system]
    if system in ("Al", "Ni"):
        atoms = fcc(system, a)
    elif system == "NiAl":
        atoms = b2("Ni", "Al", a)
    else:
        raise ValueError(f"unknown system {system!r}")
    return atoms


def lattice_parameter(system: str, volume_per_cell: float) -> float:
    """Convert the volume of the primitive cell back to a cubic lattice parameter."""
    if system in ("Al", "Ni"):  # fcc primitive cell holds a^3/4
        return (4.0 * volume_per_cell) ** (1.0 / 3.0)
    if system == "NiAl":  # B2 cell is the simple cubic cell
        return volume_per_cell ** (1.0 / 3.0)
    raise ValueError(system)
