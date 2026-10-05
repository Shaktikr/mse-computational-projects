"""Step 4b: k-point check of the stacking-fault energies.

gamma_isf is ~0.05 eV per 7 A^2 cell, so a k-point error of only 10 meV in the cell energy
shifts it by ~20 mJ/m^2. Here the rigid-shift energies at f = 0 (perfect), 0.6 (near the
unstable fault) and 1.0 (intrinsic fault) are recomputed with denser in-plane meshes.
The relaxation correction (relaxed - rigid, from the 16x16 runs) is then added on top.
Lesson learned here: with a 28 A long c-axis, ONE k-point along c is too coarse for a metal
(the 16x16x1 -> 24x24x1 change in gamma_isf was 110 -> 72 mJ/m^2). Use KZ=3 (environment variable).
The ANNNI estimate (04c_isf_annni.py) is the independent, cheaply converged cross-check."""
import os, json, sys
import numpy as np
from qe import *

g = json.load(open("results_gsfe.json"))
a0, NL, area = g["a0"], g["NL"], g["area_A2"]
EVA2_TO_MJM2 = 16021.766
ann, d111 = a0 / np.sqrt(2), a0 / np.sqrt(3)
a1 = ann * np.array([1.0, 0, 0]); a2 = ann * np.array([0.5, np.sqrt(3) / 2, 0]); bp = (a1 + a2) / 3
shift = {0: (0, 0), 1: (1 / 3, 1 / 3), 2: (2 / 3, 2 / 3)}
cart = np.array([shift[i % 3][0] * a1 + shift[i % 3][1] * a2 + [0, 0, i * d111] for i in range(NL)])
os.chdir("runs/04_gsfe")
out = json.load(open("../../results_gsfe_kcheck.json")) if os.path.exists("../../results_gsfe_kcheck.json") else {}
KZ = int(os.environ.get("KZ", "1"))
for K in [int(v) for v in (sys.argv[1:] or ["24", "32"])]:
    E = {}
    for f in (0.0, 0.6, 1.0):
        cell = np.array([a1, a2, np.array([0, 0, NL * d111]) + f * bp])
        frac = cart @ np.linalg.inv(cell)
        name = f"kchk_f{f:.2f}_k{K}" + (f"x{KZ}" if KZ > 1 else "")
        write_pw_input(name + ".in", cell, ["Al"] * NL, frac, ecut=40, kpts=(K, K, KZ), degauss=0.02,
                       outdir="./tmp_" + name)
        E[f] = run_pw(name + ".in", name + ".out")["energy_eV"]
        os.system(f"rm -rf tmp_{name}")
    out[f"k{K}x{KZ}"] = {f"gamma_rigid_f{f:.1f}": (E[f] - E[0.0]) / area * EVA2_TO_MJM2 for f in (0.6, 1.0)}
    print(K, KZ, out[f"k{K}x{KZ}"], flush=True)
    json.dump(out, open("../../results_gsfe_kcheck.json", "w"), indent=1)
