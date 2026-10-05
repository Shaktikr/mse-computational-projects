"""Cross-check of the Cu pseudopotential: repeat a0, B and C11-C12, C44 with a different Cu
pseudopotential (PAW, Cu.pbe-kjpaw.UPF from the QE distribution) using the same settings."""
import os, sys, json
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs/06_cu_paw"))
sys.path.insert(0, "../..")
import numpy as np
from scipy.optimize import curve_fit
from qe import *
PP = dict(pseudo="Cu.pbe-kjpaw.UPF", mass=63.546, element="Cu")
def run(name, cell, K=24):
    write_pw_input(name + ".in", cell, ["Cu"], [[0, 0, 0]], ecut=55, ecutrho=440, kpts=(K, K, K), degauss=0.02,
                   outdir="./tmp_" + name, conv_thr=1e-10, **PP)
    r = run_pw(name + ".in", name + ".out", nproc=1, npool=1); os.system(f"rm -rf tmp_{name}"); return r
al = np.linspace(3.56, 3.72, 9); V = al ** 3 / 4
E = np.array([run(f"ev_{a:.3f}", fcc_primitive(a))["energy_eV"] for a in al])
bm = lambda V, E0, V0, B0, Bp: E0 + 9 * V0 * B0 / 16 * (((V0 / V) ** (2 / 3) - 1) ** 3 * Bp + ((V0 / V) ** (2 / 3) - 1) ** 2 * (6 - 4 * (V0 / V) ** (2 / 3)))
(E0, V0, B0, Bp), _ = curve_fit(bm, V, E, p0=[E.min(), V[np.argmin(E)], 0.9, 4.5])
a0 = (4 * V0) ** (1 / 3); out = dict(a0=a0, B=B0 * 160.21766)
d = 0.01; c0 = fcc_primitive(a0); vals = []
for s in (1, -1):
    sg = -run(f"cp_{'p' if s > 0 else 'm'}", c0 @ (np.eye(3) + np.diag([s * d, -s * d, 0])).T)["stress_kbar"] / 10
    vals.append(sg[0, 0] - sg[1, 1])
out["C11mC12"] = (vals[0] - vals[1]) / 2 / (2 * d)
m = np.zeros((3, 3)); m[0, 1] = m[1, 0] = d / 2
out["C44"] = (-run("c44_p", c0 @ (np.eye(3) + m).T)["stress_kbar"] / 10)[0, 1] / d
print(out, flush=True); json.dump(out, open("../../results_cu_paw_check.json", "w"), indent=1)
