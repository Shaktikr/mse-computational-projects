"""
02_relax.py - relax the three structures with PBE, then compute their final energies.

  ZnO (wurtzite)  vc-relax : a, c and the internal parameter u are all optimised
  Zn  (hcp)       vc-relax : a and c optimised (metal -> smearing)
  O2  (molecule)  relax    : bond length optimised, spin triplet (2 unpaired electrons)

A variable-cell run changes the cell while the plane-wave basis stays fixed, so each
vc-relax is followed by a clean single-point (scf) at the relaxed geometry; the
formation energy uses those scf energies only.

    python3 02_relax.py [ecut] [box]        defaults: 60 Ry, 10 Angstrom
                                            (both chosen from 01_convergence.py)
"""
import json, os, sys
import numpy as np
import qe

ECUT = int(sys.argv[1]) if len(sys.argv) > 1 else 60
BOX = int(sys.argv[2]) if len(sys.argv) > 2 else 10
K_ZNO, K_ZN = (8, 8, 5), (20, 20, 10)
R = qe.RUNS + "/relax"
res = {"ecut_Ry": ECUT, "box_A": BOX, "k_ZnO": K_ZNO, "k_Zn": K_ZN}


def hex_ac(cell):
    c = np.array(cell)
    return float(np.linalg.norm(c[0])), float(np.linalg.norm(c[2]))


# ---- ZnO
cell, sym, frac = qe.wurtzite_zno()
f = f"{R}/zno_vcrelax.in"
qe.write_pw_input(f, cell, sym, frac, calc="vc-relax", ecut=ECUT, kpts=K_ZNO, prefix="zno_vc")
r = qe.run_pw(f)
f = f"{R}/zno_scf.in"
qe.write_pw_input(f, r["cell"], sym, r["frac"], ecut=ECUT, kpts=K_ZNO, prefix="zno_scf")
s = qe.run_pw(f)
a, c = hex_ac(r["cell"])
res["ZnO"] = dict(a=a, c=c, u=r["frac"][2][2] - r["frac"][0][2], E_cell_eV=s["energy_eV"],
                  n_formula_units=2, pressure_kbar=s["pressure_kbar"])
print(f"ZnO: a = {a:.4f} A, c = {c:.4f} A, u = {res['ZnO']['u']:.4f}, "
      f"P = {s['pressure_kbar']:.2f} kbar, E = {s['energy_eV']:.4f} eV / cell", flush=True)

# ---- Zn
cell, sym, frac = qe.hcp_zn()
f = f"{R}/zn_vcrelax.in"
qe.write_pw_input(f, cell, sym, frac, calc="vc-relax", ecut=ECUT, kpts=K_ZN, metal=True,
                  prefix="zn_vc")
r = qe.run_pw(f)
f = f"{R}/zn_scf.in"
qe.write_pw_input(f, r["cell"], sym, r["frac"], ecut=ECUT, kpts=K_ZN, metal=True,
                  prefix="zn_scf")
s = qe.run_pw(f)
a, c = hex_ac(r["cell"])
res["Zn"] = dict(a=a, c=c, c_over_a=c / a, E_cell_eV=s["energy_eV"], n_atoms=2,
                 pressure_kbar=s["pressure_kbar"])
print(f"Zn : a = {a:.4f} A, c = {c:.4f} A, c/a = {c / a:.3f}, "
      f"P = {s['pressure_kbar']:.2f} kbar, E = {s['energy_eV']:.4f} eV / cell", flush=True)

# ---- O2
cell, sym, frac = qe.o2_molecule(BOX)
f = f"{R}/o2_relax.in"
qe.write_pw_input(f, cell, sym, frac, calc="relax", ecut=ECUT, kpts=None, tot_mag=2,
                  prefix="o2_rx")
r = qe.run_pw(f, npool=1)        # one k-point: nothing to gain from k-point pools
d = (r["frac"][1][2] - r["frac"][0][2]) * cell[2][2]
res["O2"] = dict(bond_A=d, E_molecule_eV=r["energy_eV"], magnetization=r.get("magnetization"))
print(f"O2 : bond = {d:.4f} A, moment = {r.get('magnetization')} mu_B, "
      f"E = {r['energy_eV']:.4f} eV / molecule", flush=True)

json.dump(res, open(os.path.join(qe.HERE, "results_relax.json"), "w"), indent=1)
