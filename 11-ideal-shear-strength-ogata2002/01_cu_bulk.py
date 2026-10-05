"""Cu bulk: cutoff / k-point convergence, equation of state, elastic constants (same methods as Al project).
Pseudopotential: Cu.pbe-dn-rrkjus_psl.0.2.UPF (PBE, ultrasoft, 3d10 4s1 valence = 11 e-)."""
import os, json, sys
import numpy as np
from scipy.optimize import curve_fit
from qe import *

M = MATERIALS["Cu"]
os.makedirs("runs/01_cu_bulk", exist_ok=True); os.chdir("runs/01_cu_bulk")
EV_A3_TO_GPA = 160.21766
res = json.load(open("../../results_cu_bulk.json")) if os.path.exists("../../results_cu_bulk.json") else {}

def E(name, cell, ecut, K, conv=1e-9):
    write_pw_input(name + ".in", cell, ["Cu"], [[0, 0, 0]], ecut=ecut, kpts=(K, K, K), degauss=0.02,
                   outdir="./tmp_" + name, conv_thr=conv, **M)
    r = run_pw(name + ".in", name + ".out"); os.system(f"rm -rf tmp_{name}")
    return r

step = sys.argv[1] if len(sys.argv) > 1 else "all"
if step in ("conv", "all"):
    res["ecut"] = [[ec, E(f"ecut_{ec}", fcc_primitive(3.63), ec, 16)["energy_eV"]] for ec in [30, 35, 40, 45, 50, 55, 60, 70]]
    print(res["ecut"], flush=True)
    res["kpts"] = [[k, E(f"k_{k}", fcc_primitive(3.63), 50, k)["energy_eV"]] for k in [8, 12, 16, 20, 24, 28, 32]]
    print(res["kpts"], flush=True)
    json.dump(res, open("../../results_cu_bulk.json", "w"), indent=1)

ECUT = 55
def bm3(V, E0, V0, B0, Bp):
    eta = (V0 / V) ** (2 / 3)
    return E0 + 9 * V0 * B0 / 16 * ((eta - 1) ** 3 * Bp + (eta - 1) ** 2 * (6 - 4 * eta))
if step in ("bulk", "all"):
    for K in [int(x) for x in (sys.argv[2:] or ["24", "32"])]:
        alist = np.linspace(3.54, 3.74, 11); V = alist ** 3 / 4
        Ev = np.array([E(f"ev_k{K}_{a:.4f}", fcc_primitive(a), ECUT, K, 1e-10)["energy_eV"] for a in alist])
        (E0, V0, B0, Bp), _ = curve_fit(bm3, V, Ev, p0=[Ev.min(), V[np.argmin(Ev)], 0.9, 4.5])
        a0 = (4 * V0) ** (1 / 3); B = B0 * EV_A3_TO_GPA
        c0 = fcc_primitive(a0); ds = np.array([0.0, 0.01, 0.02, 0.03, 0.04]); Eo, Em = [], []
        for d in ds:
            eo = np.diag([d, -d, d * d / (1 - d * d)])
            em = np.zeros((3, 3)); em[0, 1] = em[1, 0] = d / 2; em[2, 2] = d * d / (4 - d * d)
            Eo.append(E(f"orth_k{K}_{d:.3f}", c0 @ (np.eye(3) + eo).T, ECUT, K, 1e-10)["energy_eV"])
            Em.append(E(f"mono_k{K}_{d:.3f}", c0 @ (np.eye(3) + em).T, ECUT, K, 1e-10)["energy_eV"])
        A = np.vstack([ds ** 2, ds ** 4]).T
        Cp2 = np.linalg.lstsq(A, (np.array(Eo) - Eo[0]) / V0, rcond=None)[0][0] * EV_A3_TO_GPA
        C44 = 2 * np.linalg.lstsq(A, (np.array(Em) - Em[0]) / V0, rcond=None)[0][0] * EV_A3_TO_GPA
        C11, C12 = B + 2 * Cp2 / 3, B - Cp2 / 3
        res[f"k{K}"] = dict(a0_A=a0, V0_A3=V0, B0_GPa=B, Bprime=Bp, C11=C11, C12=C12, C44=C44,
                            G111=(C11 - C12 + C44) / 3)
        print(K, res[f"k{K}"], flush=True)
        json.dump(res, open("../../results_cu_bulk.json", "w"), indent=1)
