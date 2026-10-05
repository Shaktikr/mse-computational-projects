"""Step 1: convergence tests for fcc Al (1-atom primitive cell).
We converge (a) plane-wave cutoff and (b) k-point mesh, so that later results
do not depend on numerical parameters by more than ~1 meV/atom."""
import os, json
import numpy as np
from qe import *

os.makedirs("runs/01_conv", exist_ok=True)
os.chdir("runs/01_conv")
a = 4.04
cell = fcc_primitive(a)
res = {"ecut": [], "kpts": []}

# (a) cutoff test at a fixed, generous k-mesh
for ec in [20, 25, 30, 35, 40, 45, 50, 60]:
    write_pw_input(f"ecut_{ec}.in", cell, ["Al"], [[0, 0, 0]], ecut=ec, kpts=(14, 14, 14),
                   outdir=f"./tmp_e{ec}")
    r = run_pw(f"ecut_{ec}.in", f"ecut_{ec}.out")
    res["ecut"].append([ec, r["energy_eV"], r["pressure_kbar"]])
    print("ecut", ec, r["energy_eV"], r["pressure_kbar"], flush=True)

# (b) k-point test at the chosen cutoff
for k in [6, 8, 10, 12, 14, 16, 18, 20, 24]:
    write_pw_input(f"k_{k}.in", cell, ["Al"], [[0, 0, 0]], ecut=40, kpts=(k, k, k),
                   outdir=f"./tmp_k{k}")
    r = run_pw(f"k_{k}.in", f"k_{k}.out")
    res["kpts"].append([k, r["energy_eV"], r["pressure_kbar"]])
    print("k", k, r["energy_eV"], r["pressure_kbar"], flush=True)

json.dump(res, open("../../results_convergence.json", "w"), indent=1)
os.system("rm -rf tmp_*")
