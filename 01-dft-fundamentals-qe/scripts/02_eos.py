"""Step 2 - equation of state: lattice parameter a0 and bulk modulus B0.

    python scripts/02_eos.py --system Al
    python scripts/02_eos.py --system Ni
    python scripts/02_eos.py --system NiAl

Output: results/<system>/eos.json and eos.png
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import initial_structure, make_factory, out_dir, parser, save_json  # noqa: E402
from dftlab.eos import birch_murnaghan, fit_birch_murnaghan, scan_volumes  # noqa: E402
from dftlab.config import EV_PER_A3_TO_GPA  # noqa: E402
from dftlab.structures import lattice_parameter  # noqa: E402


def main():
    p = parser(__doc__)
    p.add_argument("--npoints", type=int, default=9)
    args = p.parse_args()
    atoms = initial_structure(args)
    strains = np.linspace(-0.03, 0.03, args.npoints)
    V, E = scan_volumes(atoms, make_factory(args, "eos"), strains)
    res = fit_birch_murnaghan(V, E)
    a0 = lattice_parameter(args.system, res.V0)
    nat = len(atoms)

    magmom = None
    if args.calc == "qe" and args.system == "Ni":
        # total magnetisation of the run closest to equilibrium (last SCF iteration)
        from common import RUNS
        i = int(np.argmin(np.abs(V - res.V0)))
        pwo = (RUNS / "qe" / args.system / "eos" / f"eos_{i:02d}" / "espresso.pwo").read_text()
        lines = [ln for ln in pwo.splitlines() if "total magnetization" in ln]
        magmom = float(lines[-1].split("=")[1].split()[0]) / nat

    out = out_dir(args)
    data = {
        "system": args.system,
        "atoms_per_cell": nat,
        "volumes_A3": V.tolist(),
        "energies_eV": E.tolist(),
        **res.to_dict(),
        "a0_A": a0,
        "E0_per_atom_eV": res.E0 / nat,
        "V0_per_atom_A3": res.V0 / nat,
        "magnetic_moment_muB_per_atom": magmom,
    }
    save_json(out / "eos.json", data)

    Vf = np.linspace(V.min() * 0.99, V.max() * 1.01, 200)
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    ax.plot(V / nat, 1000 * (E - res.E0) / nat, "o", label="DFT" if args.calc == "qe" else args.calc.upper())
    ax.plot(Vf / nat, 1000 * (birch_murnaghan(Vf, res.E0, res.V0, res.B0_GPa / EV_PER_A3_TO_GPA, res.B0_prime) - res.E0) / nat,
            "-", label="Birch–Murnaghan fit")
    ax.set(xlabel="Volume (Å$^3$/atom)", ylabel="E − E$_0$ (meV/atom)",
           title=f"{args.system}: a$_0$ = {a0:.3f} Å, B$_0$ = {res.B0_GPa:.0f} GPa")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "eos.png", dpi=150)
    print(f"{args.system}: a0 = {a0:.4f} A, B0 = {res.B0_GPa:.1f} GPa, B0' = {res.B0_prime:.2f}")


if __name__ == "__main__":
    main()
