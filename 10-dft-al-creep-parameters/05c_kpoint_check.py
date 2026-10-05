"""Step 5c: k-point check of E_f and E_m.

The migration relaxations (05_migration.py) were done with a 4x4x4 k-mesh to keep them affordable.
Here the relaxed geometries (initial state and saddle) are re-evaluated with single SCF runs
on denser meshes. Re-using geometries is safe because atomic positions converge much faster
with k-points than energies do; the test is that E_f from this procedure reproduces the
fully relaxed 6x6x6 value from 03_vacancy.py."""
import os, json, sys
import numpy as np
from qe import *

a0 = json.load(open("results_vacancy.json"))["a0_A"]
cell, frac = fcc_conventional_supercell(a0, 2)
N = len(frac)
os.chdir("runs/05_mig")
geo = {x: final_positions(f"mig_e30_k4_x{x:.2f}.out") for x in (0.0, 0.5)}
out = {}
for K in [int(v) for v in (sys.argv[1:] or ["6", "8"])]:
    common = dict(ecut=30, kpts=(K, K, K), degauss=0.02)
    write_pw_input(f"chk_perf_k{K}.in", cell, ["Al"] * N, frac, outdir="./tmp_c", **common)
    Ep = run_pw(f"chk_perf_k{K}.in", f"chk_perf_k{K}.out")["energy_eV"]
    E = {}
    for x, pos in geo.items():
        name = f"chk_x{x:.2f}_k{K}"
        write_pw_input(name + ".in", cell, ["Al"] * (N - 1), pos, outdir="./tmp_c", **common)
        E[x] = run_pw(name + ".in", name + ".out")["energy_eV"]
    out[f"k{K}"] = dict(Ef_eV=E[0.0] - (N - 1) / N * Ep, Em_eV=E[0.5] - E[0.0])
    print(K, out[f"k{K}"], flush=True)
    os.system("rm -rf tmp_c")
    json.dump(out, open("../../results_kcheck.json", "w"), indent=1)
