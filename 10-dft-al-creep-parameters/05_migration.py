"""Step 5: vacancy migration energy E_m.

A nearest-neighbour atom (the "mover") jumps into the vacancy along <110>.
The path is symmetric: the midpoint of a nearest-neighbour bond is an inversion centre of the
fcc lattice, so initial and final states are equivalent and, for a single-hump barrier
(which is the case for Al), the saddle point lies exactly at the midpoint.

Production approach used here ("drag" / constrained-saddle method):
  * initial state  = relaxed vacancy (all atoms free)
  * images at fraction x = 0.25 and 0.50 of the jump: the mover is held at the
    corresponding point on the straight path and all other 30 atoms relax.
    At x = 0.5 the perpendicular force on the mover vanishes by symmetry, so this IS the saddle.
  E_m = E(x = 0.5) - E(x = 0)
The general-purpose tool is the climbing-image NEB (neb.x); 05b_write_neb_input.py writes a
ready-to-run CI-NEB input for the same jump so you can confirm this on a cluster."""
import os, json, sys, glob
import numpy as np
from qe import *

ECUT = int(sys.argv[1]) if len(sys.argv) > 1 else 30
K = int(sys.argv[2]) if len(sys.argv) > 2 else 4
XS = [float(v) for v in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0.0, 0.25, 0.5]
n = 2
a0 = json.load(open("results_vacancy.json"))["a0_A"]
cell, frac = fcc_conventional_supercell(a0, n)
fv = frac[1:]                       # vacancy at origin; fv[0] = (0.25,0.25,0) is the mover
start = fv[0].copy()
target = np.zeros(3)                # the vacant site
tag0 = f"e{ECUT}_k{K}"
os.makedirs("runs/05_mig", exist_ok=True)
os.chdir("runs/05_mig")
common = dict(ecut=ECUT, kpts=(K, K, K), degauss=0.02,
              extra_control="  forc_conv_thr = 5.0d-4\n  etot_conv_thr = 1.0d-5")
N = len(frac)

# perfect crystal at the same settings (to also get E_f consistently)
write_pw_input(f"perf_{tag0}.in", cell, ["Al"] * N, frac, outdir="./tmp_p", **{k: v for k, v in common.items() if k != "extra_control"})
Ep = run_pw(f"perf_{tag0}.in", f"perf_{tag0}.out")["energy_eV"]

res = {"ecut": ECUT, "k": K, "x": [], "E": []}
for x in XS:
    pos = fv.copy()
    # start from an already-relaxed geometry for this x if one exists (cheaper settings), else ideal
    prev = [p for p in glob.glob(f"mig_*_x{x:.2f}.out") if "JOB DONE" in open(p).read()]
    if prev:
        pos = final_positions(prev[0])
    pos[0] = start + x * (target - start)
    fixed = [(1, 1, 1)] * len(pos)
    if x > 0:
        fixed[0] = (0, 0, 0)                     # hold the mover on the path
    name = f"mig_{tag0}_x{x:.2f}"
    write_pw_input(name + ".in", cell, ["Al"] * len(pos), pos, calc="relax",
                   outdir="./tmp_" + name, fixed=fixed, **common)
    r = run_pw(name + ".in", name + ".out")
    res["x"].append(x); res["E"].append(r["energy_eV"])
    print(name, r["energy_eV"], flush=True)
    os.system(f"rm -rf tmp_{name}")
E = np.array(res["E"])
xs = np.array(res["x"])
# mirror symmetry of the path: E(x) = E(1 - x)
res["profile_x"] = list(np.r_[xs, 1 - xs[::-1][1:]])
res["profile_E"] = list(np.r_[E - E[0], (E - E[0])[::-1][1:]])
res["Em_eV"] = E[xs == 0.5][0] - E[0]
res["Ef_relaxed_same_settings_eV"] = E[0] - (N - 1) / N * Ep
print("E_m =", res["Em_eV"], " E_f (same settings) =", res["Ef_relaxed_same_settings_eV"])
json.dump(res, open(f"../../results_migration_{tag0}.json", "w"), indent=1)
