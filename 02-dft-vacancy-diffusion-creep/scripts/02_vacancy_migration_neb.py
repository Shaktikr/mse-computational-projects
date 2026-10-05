"""Step 2 - vacancy migration barrier E_m with the climbing-image NEB.

    python scripts/02_vacancy_migration_neb.py --element Al
    python scripts/02_vacancy_migration_neb.py --element Al --calc emt

Needs vacancy_relaxed.extxyz and perfect.extxyz from step 1.
Output: neb.json, neb.png, neb_path.extxyz (open in ASE-gui / OVITO to watch the jump)
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from ase.io import read, write  # noqa: E402

from common import out_dir, parser, save_json, settings  # noqa: E402
from vacdiff.defects import nearest_neighbour, relax_positions  # noqa: E402
from vacdiff.neb import final_state, run_neb  # noqa: E402


def main():
    p = parser(__doc__)
    p.add_argument("--images", type=int, default=3, help="intermediate images (odd -> one sits at the saddle)")
    p.add_argument("--fmax", type=float, default=0.05)
    args = p.parse_args()
    s = settings(args)
    out = out_dir(args)
    perfect = read(out / "perfect.extxyz")
    initial = read(out / "vacancy_relaxed.extxyz")
    j = nearest_neighbour(perfect, 0)
    final, E_fin = relax_positions(final_state(perfect, j), s, "neb_final_relaxed")
    print(f"jumping atom {j}: distance {perfect.get_distance(0, j, mic=True):.3f} A")

    res = run_neb(initial, final, s, n_images=args.images, fmax=args.fmax)
    save_json(out / "neb.json", {"element": args.element, "jumper_index": j, **res.to_dict()})

    # energy profile along the reaction coordinate
    x = np.linspace(0, 1, len(res.energies_eV))
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.plot(x, res.energies_eV, "o-")
    ax.axhline(res.Em_eV, ls=":", color="0.5")
    ax.set(xlabel="reaction coordinate", ylabel="E − E$_{initial}$ (eV)",
           title=f"{args.element} vacancy migration: E$_m$ = {res.Em_eV:.2f} eV")
    fig.tight_layout()
    fig.savefig(out / "neb.png", dpi=150)
    print(f"E_m = {res.Em_eV:.3f} eV (converged: {res.converged})")


if __name__ == "__main__":
    main()
