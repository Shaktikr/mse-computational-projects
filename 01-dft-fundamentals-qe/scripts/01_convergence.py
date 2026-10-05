"""Step 1 - plane-wave cut-off and k-point convergence.

    python scripts/01_convergence.py --system Al
    python scripts/01_convergence.py --system Ni

Output: results/<system>/convergence.json and convergence.png
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import initial_structure, make_factory, out_dir, parser, save_json  # noqa: E402
from dftlab import config  # noqa: E402
from dftlab.convergence import ecut_scan, first_converged, kpoint_scan  # noqa: E402

ECUTS = {"Al": [25, 30, 35, 40, 50, 60], "Ni": [35, 40, 45, 50, 60, 70], "NiAl": [35, 40, 45, 50, 60, 70]}
KSPACINGS = [0.40, 0.30, 0.25, 0.20, 0.15, 0.12, 0.10, 0.08]


def main():
    args = parser(__doc__).parse_args()
    atoms = initial_structure(args)
    factory = make_factory(args, "convergence")
    ecuts = ECUTS[args.system]

    # k-point test at a generous cut-off, cut-off test at a dense k mesh
    e_cut = ecut_scan(atoms, lambda a, t, **k: factory(a, t, kspacing=0.12, **k), ecuts)
    ec_conv = first_converged(ecuts, e_cut, 1e-3)
    e_k = kpoint_scan(atoms, lambda a, t, **k: factory(a, t, ecutwfc=max(ec_conv, config.DEFAULTS[args.system]["ecutwfc"]), **k), KSPACINGS)
    ks_conv = first_converged(KSPACINGS, e_k, 1e-3)

    out = out_dir(args)
    save_json(out / "convergence.json", {
        "system": args.system,
        "ecutwfc_Ry": ecuts,
        "energy_per_atom_vs_ecut_eV": e_cut.tolist(),
        "ecut_converged_1meV": ec_conv,
        "kspacing_invA": KSPACINGS,
        "energy_per_atom_vs_kspacing_eV": e_k.tolist(),
        "kspacing_converged_1meV": ks_conv,
    })

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].plot(ecuts, 1000 * (e_cut - e_cut[-1]), "o-")
    ax[0].axhspan(-1, 1, color="0.85", label="±1 meV/atom")
    ax[0].set(xlabel="ecutwfc (Ry)", ylabel="E − E$_{ref}$ (meV/atom)", title=f"{args.system}: cut-off")
    ax[0].legend()
    ax[1].plot(KSPACINGS, 1000 * (e_k - e_k[-1]), "s-", color="C1")
    ax[1].axhspan(-1, 1, color="0.85")
    ax[1].invert_xaxis()
    ax[1].set(xlabel="k-point spacing (Å$^{-1}$)  → denser", ylabel="E − E$_{ref}$ (meV/atom)",
              title=f"{args.system}: k-points")
    for a in ax:
        a.set_ylim(-15, 15)
    fig.tight_layout()
    fig.savefig(out / "convergence.png", dpi=150)
    print(f"converged: ecutwfc = {ec_conv} Ry, kspacing = {ks_conv} 1/A")


if __name__ == "__main__":
    main()
