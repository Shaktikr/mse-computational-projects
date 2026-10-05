"""Step 3 - cubic elastic constants C11, C12, C44 (needs eos.json from step 2).

    python scripts/03_elastic.py --system NiAl
    python scripts/03_elastic.py --system Al --reuse      # re-analyse finished runs

Two routes from the SAME strained calculations:
  * energy-strain (Mehl): fit E(d) = c0 + c2 d^2 + ...   (second order in strain)
  * stress-strain:        fit sigma(d) = C d              (first order in strain)
The energy changes involved are tiny (0.1-1 meV/atom for d = 1-2 %), comparable to the
k-point noise in a metal, so the stress route is usually the more precise one - the two
agreeing is a good convergence check. The stress route is reported as the main result.

Output: results/<system>/elastic.json and elastic.png
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from ase.data import atomic_masses, atomic_numbers  # noqa: E402

from common import load_json, make_factory, out_dir, parser, save_json  # noqa: E402
from dftlab.config import EV_PER_A3_TO_GPA  # noqa: E402
from dftlab.elastic import (constants_from_stresses, cubic_constants_from_fits, quadratic_coefficient,  # noqa: E402
                            run_strain_series_with_stress)
from dftlab.structures import build  # noqa: E402


def main():
    p = parser(__doc__)
    p.add_argument("--dmax", type=float, default=0.02, help="maximum strain amplitude")
    args = p.parse_args()
    out = out_dir(args)
    eos = load_json(out / "eos.json")
    atoms = build(args.system, a=eos["a0_A"])
    deltas = np.linspace(-args.dmax, args.dmax, 5)
    factory = make_factory(args, "elastic")

    d, e_o, s_o = run_strain_series_with_stress(atoms, factory, "ortho", deltas)
    d, e_m, s_m = run_strain_series_with_stress(atoms, factory, "mono", deltas)
    C_energy = cubic_constants_from_fits(quadratic_coefficient(d, e_o), quadratic_coefficient(d, e_m),
                                         eos["V0"], eos["B0_GPa"])
    C = constants_from_stresses(d, s_o, s_m, eos["B0_GPa"])

    masses = [atomic_masses[atomic_numbers[s]] for s in atoms.get_chemical_symbols()]
    theta_D = C.debye_temperature(np.mean(masses), eos["V0_per_atom_A3"])

    data = {"system": args.system, "method": "stress-strain (energy-strain given for comparison)",
            "deltas": d.tolist(), "energies_ortho_eV": e_o.tolist(), "energies_mono_eV": e_m.tolist(),
            "stress_ortho_GPa": (s_o * EV_PER_A3_TO_GPA).tolist(), "stress_mono_GPa": (s_m * EV_PER_A3_TO_GPA).tolist(),
            **C.to_dict(), "debye_temperature_K": theta_D,
            "energy_method": {k: C_energy.to_dict()[k] for k in ("C11", "C12", "C44")}}
    save_json(out / "elastic.json", data)

    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for e, lab, m in [(e_o, "orthorhombic → C11−C12", "o"), (e_m, "monoclinic → C44", "s")]:
        ax[0].plot(100 * d, 1000 * (e - e[len(e) // 2]) / len(atoms), m, label=lab)
        cf = np.polyfit(d, e, 4)
        dd = np.linspace(d.min(), d.max(), 100)
        ax[0].plot(100 * dd, 1000 * (np.polyval(cf, dd) - e[len(e) // 2]) / len(atoms), "-", color=ax[0].lines[-1].get_color())
    ax[0].set(xlabel="strain δ (%)", ylabel="ΔE (meV/atom)",
              title=f"energy route: C11={C_energy.C11:.0f}, C12={C_energy.C12:.0f}, C44={C_energy.C44:.0f}")
    ax[0].legend(fontsize=8)
    so, sm = s_o * EV_PER_A3_TO_GPA, s_m * EV_PER_A3_TO_GPA
    ax[1].plot(100 * d, (so[:, 0] - so[:, 1]) / 2, "o-", label="(σ1−σ2)/2 → C11−C12")
    ax[1].plot(100 * d, sm[:, 5], "s-", label="σ6 → C44")
    ax[1].set(xlabel="strain δ (%)", ylabel="stress (GPa)",
              title=f"stress route: C11={C.C11:.0f}, C12={C.C12:.0f}, C44={C.C44:.0f} GPa")
    ax[1].legend(fontsize=8)
    fig.suptitle(f"{args.system}: cubic elastic constants (PBE)" if args.calc == "qe" else f"{args.system} ({args.calc})")
    fig.tight_layout()
    fig.savefig(out / "elastic.png", dpi=150)
    print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in data.items() if not isinstance(v, (list, dict))})
    print("energy route:", {k: round(v, 1) for k, v in data["energy_method"].items()})


if __name__ == "__main__":
    main()
