"""Step 5 - collect all results into results/SUMMARY.md and compare with experiment.

    python scripts/05_summary.py

Experimental reference values (room temperature unless noted) - check the original
sources before quoting them in a paper:
  Al   a = 4.05 A; B = 76 GPa; C11/C12/C44 = 108/62/28 GPa (0 K extrapolation: 114/62/32)
  Ni   a = 3.52 A; B = 186 GPa; C11/C12/C44 = 261/151/132 GPa (~0 K, Alers et al. 1960);
       magnetic moment 0.61 muB/atom
  NiAl a = 2.887 A; C11/C12/C44 = 211.5/143.2/112.1 GPa (Wasilewski 1966) -> B = 166 GPa;
       dH_f reported between about -0.6 and -0.7 eV/atom (calorimetry)
"""

from common import RESULTS, load_json  # noqa: E402

EXP = {
    "Al": {"a0_A": 4.05, "B0_GPa": 76, "C11": 108, "C12": 62, "C44": 28},
    "Ni": {"a0_A": 3.52, "B0_GPa": 186, "C11": 261, "C12": 151, "C44": 132, "magmom": 0.61},
    "NiAl": {"a0_A": 2.887, "B0_GPa": 166, "C11": 211.5, "C12": 143.2, "C44": 112.1},
}


def fmt(x, nd=1):
    return "–" if x is None else f"{x:.{nd}f}"


def main():
    lines = ["# DFT results (PBE, Quantum ESPRESSO, ultrasoft pseudopotentials)", "",
             "| System | a0 (Å) | B (GPa) | C11 | C12 | C44 | B/G | C12−C44 | A_Zener | θ_D (K) | μ (μB/atom) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in ("Al", "Ni", "NiAl"):
        d = RESULTS / s
        if not (d / "eos.json").exists():
            continue
        eos = load_json(d / "eos.json")
        el = load_json(d / "elastic.json") if (d / "elastic.json").exists() else {}
        e = EXP[s]
        lines.append(
            f"| **{s}** (DFT) | {fmt(eos['a0_A'], 3)} | {fmt(eos['B0_GPa'], 0)} | {fmt(el.get('C11'), 0)} | "
            f"{fmt(el.get('C12'), 0)} | {fmt(el.get('C44'), 0)} | {fmt(el.get('pugh_ratio_B_over_G'), 2)} | "
            f"{fmt(el.get('cauchy_pressure'), 0)} | {fmt(el.get('zener_anisotropy'), 2)} | "
            f"{fmt(el.get('debye_temperature_K'), 0)} | {fmt(eos.get('magnetic_moment_muB_per_atom'), 2)} |"
        )
        lines.append(
            f"| {s} (exp.) | {e['a0_A']} | {e['B0_GPa']} | {e['C11']} | {e['C12']} | {e['C44']} | – | "
            f"{e['C12'] - e['C44']:.0f} | {2 * e['C44'] / (e['C11'] - e['C12']):.2f} | – | {e.get('magmom', '–')} |"
        )
    f = RESULTS / "NiAl" / "formation_enthalpy.json"
    if f.exists():
        dH = load_json(f)["formation_enthalpy_eV_per_atom"]
        lines += ["", f"Formation enthalpy of B2 NiAl: **{dH:.3f} eV/atom** ({dH * 96.485:.1f} kJ/mol-atoms); "
                      "calorimetric values lie between about −0.6 and −0.7 eV/atom."]
    lines += ["", "Elastic constants in GPa. B/G = Pugh ratio (>1.75 suggests ductile behaviour), "
                  "C12−C44 = Cauchy pressure, A_Zener = 2C44/(C11−C12), θ_D from the Voigt–Reuss–Hill moduli."]
    (RESULTS / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
