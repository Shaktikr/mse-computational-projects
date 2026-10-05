"""Step 2: bulk properties of fcc Al.
(a) Equation of state E(V) -> equilibrium lattice constant a0 and bulk modulus B0
    (3rd-order Birch-Murnaghan fit).
(b) Elastic constants from the ENERGY-STRAIN method with volume-conserving strains
    (Mehl et al., PRB 41, 10311 (1990)):
      orthorhombic  e = (d, -d, d^2/(1-d^2))          -> dE/V0 = (C11 - C12) d^2 + O(d^4)
      monoclinic    e_xy = e_yx = d/2, e_zz = d^2/(4-d^2) -> dE/V0 = (1/2) C44 d^2 + O(d^4)
    Combined with B = (C11 + 2 C12)/3 from the EOS this gives C11, C12, C44.
    (A first attempt with the stress-strain method at +/-1% strain was too noisy for Al:
     the tiny stresses are dominated by Brillouin-zone sampling error of the Fermi surface.
     The energy method with larger, symmetric strains cancels most of that error.)
(c) Polycrystalline shear modulus G (Voigt-Reuss-Hill), Young's modulus, Poisson ratio,
    Zener anisotropy. G and b are the scaling quantities in every power-law creep equation."""
import os, json, sys
import numpy as np
from scipy.optimize import curve_fit
from qe import *

ECUT = 40
KLIST = [int(x) for x in sys.argv[1:]] or [24, 32, 40]
os.makedirs("runs/02_bulk", exist_ok=True)
os.chdir("runs/02_bulk")
EV_A3_TO_GPA = 160.21766

def bm3(V, E0, V0, B0, Bp):
    eta = (V0 / V) ** (2 / 3)
    return E0 + 9 * V0 * B0 / 16 * ((eta - 1) ** 3 * Bp + (eta - 1) ** 2 * (6 - 4 * eta))

def energy(name, cell, K):
    write_pw_input(name + ".in", cell, ["Al"], [[0, 0, 0]], ecut=ECUT, kpts=(K, K, K),
                   degauss=0.02, outdir="./tmp_" + name, conv_thr=1e-10)
    return run_pw(name + ".in", name + ".out")["energy_eV"]

results = {}
for K in KLIST:
    # ---- (a) EOS ----
    alist = np.linspace(3.94, 4.14, 11)
    V = alist ** 3 / 4
    E = np.array([energy(f"ev_k{K}_{a:.4f}", fcc_primitive(a), K) for a in alist])
    popt, _ = curve_fit(bm3, V, E, p0=[E.min(), V[np.argmin(E)], 0.5, 4.0])
    E0, V0, B0, Bp = popt
    a0 = (4 * V0) ** (1 / 3)
    B = B0 * EV_A3_TO_GPA
    # ---- (b) volume-conserving strains at a0 ----
    cell0 = fcc_primitive(a0)
    deltas = np.array([0.0, 0.01, 0.02, 0.03, 0.04])
    Eo, Em = [], []
    for d in deltas:
        eo = np.diag([d, -d, d * d / (1 - d * d)])
        em = np.zeros((3, 3)); em[0, 1] = em[1, 0] = d / 2; em[2, 2] = d * d / (4 - d * d)
        Eo.append(energy(f"orth_k{K}_{d:.3f}", cell0 @ (np.eye(3) + eo).T, K))
        Em.append(energy(f"mono_k{K}_{d:.3f}", cell0 @ (np.eye(3) + em).T, K))
    Eo, Em = np.array(Eo), np.array(Em)
    # fit dE/V0 = c2 d^2 + c4 d^4
    A = np.vstack([deltas ** 2, deltas ** 4]).T
    c_o = np.linalg.lstsq(A, (Eo - Eo[0]) / V0, rcond=None)[0][0] * EV_A3_TO_GPA   # = C11 - C12
    c_m = np.linalg.lstsq(A, (Em - Em[0]) / V0, rcond=None)[0][0] * EV_A3_TO_GPA   # = C44 / 2
    Cp2 = c_o                    # C11 - C12
    C44 = 2 * c_m
    C11 = B + 2 * Cp2 / 3
    C12 = B - Cp2 / 3
    GV = (C11 - C12 + 3 * C44) / 5
    GR = 5 * (C11 - C12) * C44 / (4 * C44 + 3 * (C11 - C12))
    GH = (GV + GR) / 2
    Ey = 9 * B * GH / (3 * B + GH)
    nu = (3 * B - 2 * GH) / (2 * (3 * B + GH))
    res = dict(K=K, a0_A=a0, V0_A3=V0, B0_GPa=B, Bprime=Bp, E0_eV=E0,
               C11=C11, C12=C12, C44=C44, Cprime=Cp2 / 2,
               G_Voigt=GV, G_Reuss=GR, G_Hill=GH, E_Young=Ey, poisson=nu,
               Zener_A=2 * C44 / (C11 - C12), burgers_A=a0 / np.sqrt(2),
               ev_curve=dict(a=list(alist), V=list(V), E=list(E)),
               deltas=list(deltas), E_orth=list(Eo), E_mono=list(Em))
    results[f"k{K}"] = res
    print(f"k={K}: a0={a0:.4f} B={B:.1f} C11={C11:.1f} C12={C12:.1f} C44={C44:.1f} "
          f"G_H={GH:.1f} E={Ey:.1f} nu={nu:.3f} A={res['Zener_A']:.2f}", flush=True)
    json.dump(results, open("../../results_bulk.json", "w"), indent=1)
os.system("rm -rf tmp_*")
