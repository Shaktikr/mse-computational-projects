"""Step 4 (convergence check) - k-point sensitivity of the vacancy formation energy.

    python scripts/04_kpoint_check.py --element Al --kspacings 0.20 0.13

Recomputes the UNRELAXED formation energy E_f = E(N-1) - (N-1)/N E(N) with denser k-point
meshes (the expensive ionic relaxation is not repeated). The relaxation energy from step 1
is then added to estimate the converged relaxed value:

    E_f(relaxed, dense k) ~ E_f(unrelaxed, dense k) + [E_f(relaxed) - E_f(unrelaxed)](step 1)

Al is a notoriously k-point-sensitive case: small supercells with coarse meshes can be off
by 0.1 eV. Output: results/<element>_<n>x<n>x<n>/kpoint_check.json
"""

import json

from common import lattice_parameter, out_dir, parser, save_json, settings  # noqa: E402
from vacdiff.defects import Settings, energy, fcc_supercell, kpoints_used, remove_atom  # noqa: E402


def main():
    p = parser(__doc__)
    p.add_argument("--kspacings", nargs="+", type=float, default=[0.20, 0.13])
    args = p.parse_args()
    base = settings(args)
    out = out_dir(args)
    a0 = lattice_parameter(args)
    perfect = fcc_supercell(args.element, a0, args.n)
    vac = remove_atom(perfect, 0)
    N = len(perfect)
    step1 = json.loads((out / "vacancy.json").read_text())
    relax = step1["Ef_relaxed_eV"] - step1["Ef_unrelaxed_eV"]
    rows = []
    for ks in args.kspacings:
        s = Settings(**{**base.to_dict(), "kspacing": ks})
        tag = f"kcheck_{ks:.3f}"
        E_p = energy(perfect, s, f"{tag}_perfect")
        E_v = energy(vac, s, f"{tag}_vac")
        Ef_u = E_v - (N - 1) / N * E_p
        rows.append({"kspacing": ks, "kmesh": kpoints_used(perfect, s), "Ef_unrelaxed_eV": Ef_u,
                     "Ef_relaxed_estimate_eV": Ef_u + relax})
        print(f"k-spacing {ks:.3f} 1/A, mesh {kpoints_used(perfect, s)}: E_f(unrelaxed) = {Ef_u:.3f} eV, "
              f"E_f(relaxed, est.) = {Ef_u + relax:.3f} eV", flush=True)
    save_json(out / "kpoint_check.json", {"element": args.element, "relaxation_energy_eV": relax, "runs": rows})


if __name__ == "__main__":
    main()
