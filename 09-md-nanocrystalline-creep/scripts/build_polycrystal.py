"""Write a periodic Voronoi polycrystal as a LAMMPS data file (for cluster runs).

    python scripts/build_polycrystal.py --material Ni --box 150 --grains 16 --out nc_Ni.data
    python scripts/build_polycrystal.py --material CoAl --box 120 --grains 12 --out nc_CoAl.data
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mdcreep.polycrystal import voronoi_polycrystal, write_lammps_data  # noqa: E402

LATT = {"Ni": (["Ni"], 3.52, "fcc"), "Cu": (["Cu"], 3.615, "fcc"), "Al": (["Al"], 4.05, "fcc"),
        "CoAl": (["Co", "Al"], 2.86, "b2"), "NiAl": (["Ni", "Al"], 2.887, "b2")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--material", default="Ni", choices=list(LATT))
    ap.add_argument("--box", type=float, default=150.0, help="box edge in Angstrom")
    ap.add_argument("--grains", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="nc.data")
    a = ap.parse_args()
    symbols, lat, kind = LATT[a.material]
    atoms, gid = voronoi_polycrystal(symbols, lat, a.box, a.grains, kind, seed=a.seed)
    write_lammps_data(atoms, a.out, symbols)
    d = a.box * (6 / (np.pi * a.grains)) ** (1 / 3) / 10
    print(f"{a.out}: {len(atoms)} atoms, {a.grains} grains, mean grain size ~ {d:.1f} nm")


if __name__ == "__main__":
    main()
