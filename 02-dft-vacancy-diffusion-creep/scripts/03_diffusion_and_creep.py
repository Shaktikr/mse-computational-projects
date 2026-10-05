"""Step 3 - from E_f and E_m to D(T) and diffusional-creep rates.

    python scripts/03_diffusion_and_creep.py --element Al

Needs vacancy.json + neb.json (steps 1-2) and, for the attempt frequency, the Debye
temperature from project 01 (elastic.json). Grain-boundary diffusion is not computed here;
literature values (Frost & Ashby 1982) are used for the Coble term.

Output: diffusion_creep.json, fig_diffusion.png, fig_creep_vs_grain_size.png
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import debye_temperature, out_dir, parser, save_json  # noqa: E402
from vacdiff import creep as cr  # noqa: E402
from vacdiff.diffusion import EXPERIMENT, arrhenius, arrhenius_parameters, debye_frequency, self_diffusion_fcc  # noqa: E402

#: Grain-boundary diffusion (delta*D0b in m^3/s, Qb in kJ/mol) and melting point (K),
#: Frost & Ashby "Deformation-Mechanism Maps" (1982), Tables 4.1 (Ni) and 4.1/5 (Al).
GB = {"Al": {"deltaD0b": 5.0e-14, "Qb": 84.0, "Tm": 933.0},
      "Ni": {"deltaD0b": 3.5e-15, "Qb": 115.0, "Tm": 1726.0}}


def main():
    p = parser(__doc__)
    p.add_argument("--Sf", type=float, default=1.0, help="vacancy formation entropy in units of k_B")
    p.add_argument("--stress", type=float, default=5.0, help="applied stress for the creep plot (MPa)")
    args = p.parse_args()
    out = out_dir(args)
    vac = json.loads((out / "vacancy.json").read_text())
    neb = json.loads((out / "neb.json").read_text())
    a0, Ef, Em = vac["a0_A"], vac["Ef_relaxed_eV"], neb["Em_eV"]
    thetaD = debye_temperature(args)
    nu = debye_frequency(thetaD)
    D0, Q_eV, Q_kJ = arrhenius_parameters(a0, Ef, Em, nu, args.Sf)
    lab = "DFT" if args.calc == "qe" else args.calc.upper()
    exp = EXPERIMENT[args.element]
    gb = GB[args.element]
    Omega = (a0 * 1e-10) ** 3 / 4.0  # atomic volume in fcc

    Tm = gb["Tm"]
    T = np.linspace(0.45 * Tm, 0.99 * Tm, 200)
    D_dft = self_diffusion_fcc(T, a0, Ef, Em, nu, args.Sf)
    D_band = [self_diffusion_fcc(T, a0, Ef, Em, nu, s) for s in (0.0, 2.0)]
    D_exp = arrhenius(T, exp["D0"], exp["Q_kJ"])

    # ---------------- diffusion plot
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.fill_between(1000 / T, D_band[0], D_band[1], color="C0", alpha=0.2, label=f"{lab}, S$_f$ = 0–2 k$_B$")
    ax.semilogy(1000 / T, D_dft, "C0-", label=f"{lab}: Q = E$_f$+E$_m$ = {Q_eV:.2f} eV")
    ax.semilogy(1000 / T, D_exp, "k--", label=f"Experiment: Q = {exp['Q_kJ'] / 96.485:.2f} eV")
    ax.set(xlabel="1000/T (K$^{-1}$)", ylabel="D (m$^2$/s)", title=f"{args.element} self-diffusion")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_diffusion.png", dpi=150)
    plt.close(fig)

    # ---------------- creep rate vs grain size at 0.7 Tm
    Tc = 0.7 * Tm
    d = np.geomspace(20e-9, 1e-3, 200)
    D_L = self_diffusion_fcc(Tc, a0, Ef, Em, nu, args.Sf)
    dDgb = cr.deltaD_gb(Tc, gb["deltaD0b"], gb["Qb"])
    e_nh = cr.nabarro_herring(args.stress, Tc, d, D_L, Omega)
    e_c = cr.coble(args.stress, Tc, d, dDgb, Omega)
    d_star = cr.crossover_grain_size(Tc, lambda t: self_diffusion_fcc(t, a0, Ef, Em, nu, args.Sf),
                                     lambda t: cr.deltaD_gb(t, gb["deltaD0b"], gb["Qb"]))
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.loglog(d * 1e6, e_nh, label=f"Nabarro–Herring (D$_L$ from {lab})")
    ax.loglog(d * 1e6, e_c, label="Coble (δD$_{gb}$ from literature)")
    ax.loglog(d * 1e6, e_nh + e_c, "k-", lw=2, label="total diffusional creep")
    ax.axvline(d_star * 1e6, ls=":", color="0.4")
    ax.text(d_star * 1e6 * 1.1, (e_nh + e_c).max() / 30, f"d* = {d_star * 1e6:.1f} µm", fontsize=8)
    ax.set(xlabel="grain size d (µm)", ylabel=r"$\dot\varepsilon$ (s$^{-1}$)",
           title=f"{args.element}: σ = {args.stress:g} MPa, T = 0.7 T$_m$ = {Tc:.0f} K")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_creep_vs_grain_size.png", dpi=150)
    plt.close(fig)

    data = {
        "element": args.element, "a0_A": a0, "Ef_eV": Ef, "Em_eV": Em,
        "Q_dft_eV": Q_eV, "Q_dft_kJ_mol": Q_kJ, "Q_exp_kJ_mol": exp["Q_kJ"],
        "debye_temperature_K": thetaD, "attempt_frequency_Hz": nu, "Sf_over_kB": args.Sf,
        "D0_dft_m2s": D0, "D0_exp_m2s": exp["D0"],
        "D_at_0.8Tm": {"T_K": 0.8 * Tm, "dft": float(self_diffusion_fcc(0.8 * Tm, a0, Ef, Em, nu, args.Sf)),
                       "exp": float(arrhenius(0.8 * Tm, exp["D0"], exp["Q_kJ"]))},
        "creep_at_0.7Tm": {"T_K": Tc, "stress_MPa": args.stress, "crossover_grain_size_um": d_star * 1e6,
                           "rate_d_1um_per_s": float(cr.nabarro_herring(args.stress, Tc, 1e-6, D_L, Omega)
                                                     + cr.coble(args.stress, Tc, 1e-6, dDgb, Omega)),
                           "rate_d_100um_per_s": float(cr.nabarro_herring(args.stress, Tc, 1e-4, D_L, Omega)
                                                       + cr.coble(args.stress, Tc, 1e-4, dDgb, Omega))},
    }
    save_json(out / "diffusion_creep.json", data)
    print(f"Q({lab}) = {Q_kJ:.0f} kJ/mol vs {exp['Q_kJ']:.0f} (exp);  D0({lab}) = {D0:.2e} vs {exp['D0']:.1e} m2/s")


if __name__ == "__main__":
    main()
