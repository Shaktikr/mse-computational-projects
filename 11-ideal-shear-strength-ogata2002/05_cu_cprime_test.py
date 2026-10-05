"""k-point / smearing check of C' = (C11-C12)/2 for Cu from stresses:
volume-conserving orthorhombic strain e = (d, -d, 0): sigma_xx - sigma_yy = 2 (C11 - C12) d (linear order)."""
import os, sys, json
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs/05_cu_cprime"))
import numpy as np
sys.path.insert(0, "../.."); from qe import *
a0 = json.load(open("../../results_cu_bulk.json"))["k24"]["a0_A"]
M = MATERIALS["Cu"]; d = 0.01; out = {}
for K, deg in [(24, 0.02), (32, 0.02), (40, 0.02), (24, 0.04)]:
    vals = []
    for s in (+1, -1):
        cell = fcc_primitive(a0) @ (np.eye(3) + np.diag([s * d, -s * d, 0])).T
        n = f"cp_k{K}_d{deg}_{'p' if s > 0 else 'm'}"
        write_pw_input(n + ".in", cell, ["Cu"], [[0, 0, 0]], ecut=55, kpts=(K, K, K), degauss=deg,
                       outdir="./tmp_" + n, conv_thr=1e-10, **M)
        r = run_pw(n + ".in", n + ".out", nproc=1, npool=1); os.system(f"rm -rf tmp_{n}")
        sg = -r["stress_kbar"] / 10; vals.append(sg[0, 0] - sg[1, 1])
    out[f"k{K}_d{deg}"] = (vals[0] - vals[1]) / 2 / (2 * d)       # = C11 - C12 (GPa), central difference
    print(K, deg, "C11-C12 =", out[f"k{K}_d{deg}"], flush=True)
    json.dump(out, open("../../results_cu_cprime.json", "w"), indent=1)
