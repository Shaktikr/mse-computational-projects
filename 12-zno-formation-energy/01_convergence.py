"""
01_convergence.py - is the formation energy converged?

The three total energies (ZnO, Zn, O2) are huge numbers; only their DIFFERENCE matters:
    dH_f = E(ZnO)/2 - E(Zn)/2 - E(O2)/2          [eV per ZnO formula unit]
so we converge dH_f itself, at the experimental geometries, with respect to
  (a) plane-wave cut-off,  (b) k-point mesh,  (c) size of the empty box around O2.

    python3 01_convergence.py
"""
import json, os
import qe

R = qe.RUNS + "/conv"


def e_zno(ecut, k):
    f = f"{R}/zno_e{ecut}_k{k[0]}{k[1]}{k[2]}.in"
    qe.write_pw_input(f, *qe.wurtzite_zno(), ecut=ecut, kpts=k, prefix=f"zno{ecut}{k[0]}")
    return qe.run_pw(f)["energy_eV"] / 2


def e_zn(ecut, k):
    f = f"{R}/zn_e{ecut}_k{k[0]}{k[1]}{k[2]}.in"
    qe.write_pw_input(f, *qe.hcp_zn(), ecut=ecut, kpts=k, metal=True, prefix=f"zn{ecut}{k[0]}")
    return qe.run_pw(f)["energy_eV"] / 2


def e_o2(ecut, box):
    f = f"{R}/o2_e{ecut}_box{box}.in"
    qe.write_pw_input(f, *qe.o2_molecule(box), ecut=ecut, kpts=None, tot_mag=2,
                      prefix=f"o2{ecut}{box}")
    return qe.run_pw(f, npool=1)["energy_eV"] / 2       # per O atom; one k-point -> one pool


def dh(ecut, kzno, kzn, box):
    return e_zno(ecut, kzno) - e_zn(ecut, kzn) - e_o2(ecut, box)


out = {}
K_ZNO, K_ZN, BOX = (6, 6, 4), (16, 16, 8), 10
print("(a) cut-off  [k and box fixed]")
out["ecut"] = {}
for ecut in (40, 50, 60, 70):
    out["ecut"][ecut] = dh(ecut, K_ZNO, K_ZN, BOX)
    print(f"  ecutwfc = {ecut} Ry   dH_f = {out['ecut'][ecut]:.4f} eV", flush=True)

print("(b) k-points  [ecut = 50 Ry]")
out["k_zno"], out["k_zn"] = {}, {}
for k in ((4, 4, 2), (6, 6, 4), (8, 8, 5)):
    out["k_zno"]["x".join(map(str, k))] = e_zno(50, k)
    print(f"  ZnO {k}   E = {e_zno(50, k):.4f} eV / f.u.", flush=True)
for k in ((12, 12, 6), (16, 16, 8), (20, 20, 10), (24, 24, 12)):
    out["k_zn"]["x".join(map(str, k))] = e_zn(50, k)
    print(f"  Zn  {k}   E = {e_zn(50, k):.4f} eV / atom", flush=True)

print("(c) O2 box  [ecut = 50 Ry]")
out["box"] = {}
for box in (8, 10, 12):
    out["box"][box] = e_o2(50, box)
    print(f"  box = {box} A   E = {e_o2(50, box):.4f} eV / O atom", flush=True)

json.dump(out, open(os.path.join(qe.HERE, "results_convergence.json"), "w"), indent=1)
