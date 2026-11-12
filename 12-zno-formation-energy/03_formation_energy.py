"""
03_formation_energy.py - formation energy of ZnO and comparison with the literature.

    Zn (hcp) + 1/2 O2 (gas) -> ZnO (wurtzite)
    dH_f = E(ZnO)/2 - E(Zn)/2 - E(O2)/2            [eV per ZnO formula unit, T = 0 K]

    python3 03_formation_energy.py
"""
import json, os
import qe

r = json.load(open(os.path.join(qe.HERE, "results_relax.json")))
e_zno = r["ZnO"]["E_cell_eV"] / r["ZnO"]["n_formula_units"]
e_zn = r["Zn"]["E_cell_eV"] / r["Zn"]["n_atoms"]
e_o = r["O2"]["E_molecule_eV"] / 2
dh = e_zno - e_zn - e_o

# Literature (eV per formula unit). Sources are listed in README.md.
LIT = [("GGA-PBE (Clark et al. 2010)", -2.8),
       ("hybrid / screened exchange (Clark et al. 2010)", "-3.1 to -3.3"),
       ("HSE with 38 % exact exchange (Santana et al. 2015)", -3.25),
       ("diffusion Monte Carlo, two variants (Santana et al. 2015)", "-3.94 and -4.15"),
       ("experiment (CODATA value listed by NIST)", -3.63)]
# Structure: (this work key, experiment)
GEOM = [("ZnO a (A)", r["ZnO"]["a"], 3.250), ("ZnO c (A)", r["ZnO"]["c"], 5.207),
        ("ZnO u", r["ZnO"]["u"], 0.382), ("Zn a (A)", r["Zn"]["a"], 2.665),
        ("Zn c (A)", r["Zn"]["c"], 4.947), ("O2 bond (A)", r["O2"]["bond_A"], 1.208)]

lines = ["# ZnO formation energy - results", "",
         f"PBE, PAW, ecutwfc = {r['ecut_Ry']} Ry, O2 box = {r['box_A']} A", "",
         "| quantity | this work (PBE) | experiment | difference |", "|---|---|---|---|"]
for name, val, exp in GEOM:
    lines.append(f"| {name} | {val:.4f} | {exp} | {100 * (val - exp) / exp:+.1f} % |")
lines += ["", "| energy | eV |", "|---|---|",
          f"| E(ZnO) per formula unit | {e_zno:.4f} |",
          f"| E(Zn) per atom | {e_zn:.4f} |",
          f"| E(O2) / 2 | {e_o:.4f} |",
          f"| **dH_f (this work, PBE)** | **{dh:.3f}** |", "",
          "| reference | dH_f (eV per ZnO) |", "|---|---|"]
lines += [f"| {n} | {v} |" for n, v in LIT]
lines += ["", f"Per atom: {dh / 2:.3f} eV/atom.  In kJ/mol: {dh * 96.485:.1f}.",
          f"Gap to experiment: {dh + 3.63:+.2f} eV (the known PBE error for oxides)."]
txt = "\n".join(lines) + "\n"
print(txt)
open(os.path.join(qe.HERE, "SUMMARY.md"), "w").write(txt)
json.dump(dict(dHf_eV_per_fu=dh, E_ZnO_fu=e_zno, E_Zn_atom=e_zn, E_O_half_O2=e_o),
          open(os.path.join(qe.HERE, "results_formation_energy.json"), "w"), indent=1)
