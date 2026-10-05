"""Step 4d: diagnose the tilted-cell ISF. The same intrinsic fault can be made by tilting c by +b_p
(f = 1) or by -2 b_p (f = -2): they differ by an in-plane lattice vector, so the PHYSICAL structure is
identical, only the cell shape (and hence the k-point mesh) differs. Any energy difference is numerical."""
import os, json, sys
import numpy as np
from qe import *
g = json.load(open("results_gsfe.json")); a0, NL, area = g["a0"], g["NL"], g["area_A2"]
ann, d111 = a0 / np.sqrt(2), a0 / np.sqrt(3)
a1 = ann * np.array([1.0, 0, 0]); a2 = ann * np.array([0.5, np.sqrt(3) / 2, 0]); bp = (a1 + a2) / 3
shift = {0: (0, 0), 1: (1 / 3, 1 / 3), 2: (2 / 3, 2 / 3)}
cart = np.array([shift[i % 3][0] * a1 + shift[i % 3][1] * a2 + [0, 0, i * d111] for i in range(NL)])
os.chdir("runs/04_gsfe"); K = int(sys.argv[1]) if len(sys.argv) > 1 else 16; out = {}
E = {}
for f in (0.0, 1.0, -2.0):
    cell = np.array([a1, a2, np.array([0, 0, NL * d111]) + f * bp]); frac = cart @ np.linalg.inv(cell)
    name = f"tilt_f{f:+.1f}_k{K}x3"
    write_pw_input(name + ".in", cell, ["Al"] * NL, frac, ecut=40, kpts=(K, K, 3), degauss=0.02, outdir="./tmp_" + name)
    E[f] = run_pw(name + ".in", name + ".out")["energy_eV"]; os.system(f"rm -rf tmp_{name}")
    print(f, (E[f] - E[0.0]) / area * 16021.766, flush=True)
json.dump({str(f): (E[f] - E[0.0]) / area * 16021.766 for f in E}, open("../../results_tilt_test.json", "w"), indent=1)
