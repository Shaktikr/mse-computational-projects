"""Step 2 - vacancy migration barrier E_m.

    python scripts/02_vacancy_migration_neb.py --element Al                 # midpoint method (default)
    python scripts/02_vacancy_migration_neb.py --element Al --method neb    # climbing-image NEB
    python scripts/02_vacancy_migration_neb.py --element Ni --calc emt --method neb

Methods
  midpoint  the jumping atom is fixed half-way along the (symmetric) fcc jump and all other
            atoms are relaxed: E_m = E(saddle) - E(relaxed vacancy). One relaxation, exact
            for symmetric paths in pure metals.
  neb       climbing-image nudged elastic band between relaxed end states (general, but
            each optimiser step costs one DFT calculation per image).
Running both with --calc emt is a quick check that they agree.

Needs vacancy_relaxed.extxyz, perfect.extxyz and vacancy.json from step 1.
Output: neb.json, neb.png
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from ase.io import read  # noqa: E402

from common import out_dir, parser, save_json, settings  # noqa: E402
from vacdiff.defects import nearest_neighbour, relax_positions  # noqa: E402
from vacdiff.neb import final_state, midpoint_state, run_neb  # noqa: E402


def main():
    p = parser(__doc__)
    p.add_argument("--method", default="midpoint", choices=["midpoint", "neb"])
    p.add_argument("--images", type=int, default=3, help="NEB: intermediate images (odd -> one at the saddle)")
    p.add_argument("--fmax", type=float, default=0.05)
    args = p.parse_args()
    s = settings(args)
    out = out_dir(args)
    perfect = read(out / "perfect.extxyz")
    initial = read(out / "vacancy_relaxed.extxyz")
    j = nearest_neighbour(perfect, 0)
    print(f"jumping atom {j}: jump distance {perfect.get_distance(0, j, mic=True):.3f} A, method {args.method}")

    if args.method == "midpoint":
        E_init = json.loads((out / "vacancy.json").read_text())["E_vac_relaxed_eV"]
        saddle, E_sad = relax_positions(midpoint_state(perfect, j), s, "saddle_midpoint_relaxed")
        Em = E_sad - E_init
        energies = [0.0, Em, 0.0]  # end states are equivalent by symmetry
        res = {"method": "constrained midpoint relaxation", "energies_eV": energies, "Em_eV": Em,
               "saddle_index": 1, "converged": True, "E_initial_eV": E_init, "E_saddle_eV": E_sad}
    else:
        final, _ = relax_positions(final_state(perfect, j), s, "neb_final_relaxed")
        r = run_neb(initial, final, s, n_images=args.images, fmax=args.fmax)
        res = {"method": "climbing-image NEB", **r.to_dict()}
        energies, Em = r.energies_eV, r.Em_eV
    save_json(out / "neb.json", {"element": args.element, "jumper_index": j, **res})

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    x = np.linspace(0, 1, len(energies))
    if args.method == "midpoint":
        xs = np.linspace(0, 1, 100)
        ax.plot(xs, Em * np.sin(np.pi * xs) ** 2, "--", color="0.6", lw=1, label="schematic path")
        ax.plot(x, energies, "o", ms=8, label="DFT: end states and saddle" if args.calc == "qe" else args.calc.upper())
    else:
        ax.plot(x, energies, "o-", label="NEB images")
    ax.axhline(Em, ls=":", color="0.5")
    ax.set(xlabel="reaction coordinate", ylabel="E − E$_{initial}$ (eV)",
           title=f"{args.element} vacancy migration: E$_m$ = {Em:.2f} eV")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "neb.png", dpi=150)
    print(f"E_m = {Em:.3f} eV")


if __name__ == "__main__":
    main()
