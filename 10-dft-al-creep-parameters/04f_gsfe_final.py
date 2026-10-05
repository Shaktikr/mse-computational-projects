"""Step 4f: final GSFE with enough empty bands.

DIAGNOSIS (the most instructive bug in this project): the 12-layer cell is 28 A long, so at some
in-plane k-points (e.g. the K point) ~25 bands are occupied, although only 18 are occupied on
average. pw.x's default for metals, nbnd = max(1.2*nelec/2, nelec/2+4) = 22, therefore cut off
occupied states at those k-points: the energy of the perfect 12-layer cell came out 14.5 meV/atom
too high, and every GSFE energy difference was corrupted. pw.x printed no warning.
Fix: nbnd = 40 (verified: energy then agrees with the bulk), and 3 k-points along c.

Rigid-shift curve on the full f grid, then single SCFs on the relaxed geometries from 04_gsfe.py
(f = 0.6 and 1.0) to get the relaxation correction with the corrected settings."""
import os, json, sys
import numpy as np
from qe import *

g = json.load(open("results_gsfe.json")); a0, NL, area = g["a0"], g["NL"], g["area_A2"]
EVA2_TO_MJM2 = 16021.766
ann, d111 = a0 / np.sqrt(2), a0 / np.sqrt(3)
a1 = ann * np.array([1.0, 0, 0]); a2 = ann * np.array([0.5, np.sqrt(3) / 2, 0]); bp = (a1 + a2) / 3
shift = {0: (0, 0), 1: (1 / 3, 1 / 3), 2: (2 / 3, 2 / 3)}
cart = np.array([shift[i % 3][0] * a1 + shift[i % 3][1] * a2 + [0, 0, i * d111] for i in range(NL)])
K = (int(sys.argv[1]) if len(sys.argv) > 1 else 16)
NB = 40
os.makedirs("runs/04f_gsfe_final", exist_ok=True)
os.chdir("runs/04f_gsfe_final")
res_path = f"../../results_gsfe_final_k{K}.json"
res = json.load(open(res_path)) if os.path.exists(res_path) else {}
fs = [0.0, 1.0, 0.6, 0.5, 0.4, 0.2, 0.8]          # key points first
common = dict(ecut=40, kpts=(K, K, 3), degauss=0.02, nbnd=NB)
E = {}
for f in fs:
    cell = np.array([a1, a2, np.array([0, 0, NL * d111]) + f * bp]); frac = cart @ np.linalg.inv(cell)
    name = f"rigid_f{f:.2f}_k{K}"
    write_pw_input(name + ".in", cell, ["Al"] * NL, frac, outdir="./tmp_" + name, **common)
    E[f] = run_pw(name + ".in", name + ".out")["energy_eV"]; os.system(f"rm -rf tmp_{name}")
    res[f"rigid_f{f:.2f}"] = (E[f] - E[0.0]) / area * EVA2_TO_MJM2
    res["E0_per_atom_eV"] = E[0.0] / NL
    print(name, res[f"rigid_f{f:.2f}"], flush=True)
    json.dump(res, open(res_path, "w"), indent=1)
    if f == 0.6:   # relaxed geometries right after the key rigid points
        for fr in (0.6, 1.0):
            cellr = np.array([a1, a2, np.array([0, 0, NL * d111]) + fr * bp])
            pos = final_positions(f"../04_gsfe/rel_f{fr:.2f}.out")
            name = f"relaxedgeom_f{fr:.2f}_k{K}"
            write_pw_input(name + ".in", cellr, ["Al"] * NL, pos, outdir="./tmp_" + name, **common)
            er = run_pw(name + ".in", name + ".out")["energy_eV"]; os.system(f"rm -rf tmp_{name}")
            res[f"relaxed_f{fr:.2f}"] = (er - E[0.0]) / area * EVA2_TO_MJM2
            print(name, res[f"relaxed_f{fr:.2f}"], flush=True)
            json.dump(res, open(res_path, "w"), indent=1)
