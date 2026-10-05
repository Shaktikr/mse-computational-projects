"""Ideal shear strength on {111}<112> (twinning sense), after Ogata, Li & Yip, Science 298, 807 (2002).

Frame: x = shear direction <112> (the sense in which a shear of gamma = 1/sqrt(2) gives the twin),
       z = (111) plane normal, y = z cross x.
Deformation gradient applied to the 1-atom fcc primitive cell (rows = lattice vectors, v' = F v):
       F = I + gamma * e_x e_z^T + D,     D symmetric, D_xz = 0
  unrelaxed ("affine"): D = 0
  relaxed ("pure shear"): the five other components D_xx, D_yy, D_zz, D_xy, D_yz are iterated until
       the five other Cauchy stress components vanish (|sigma| < TOL), so only sigma_xz acts.
The resolved shear stress tau = sigma_xz. Ideal strength tau_max = max over gamma of tau(gamma)."""
import os, json, sys
import numpy as np
from qe import *
from shear_geometry import twin_frame, cubic_stiffness, rotate, relaxation_stiffness, COMPS

EL = sys.argv[1]                     # Al, Cu (ultrasoft) or CuPAW
MODE = sys.argv[2]                   # unrelaxed | relaxed
K = int(sys.argv[3])                 # k-mesh
GAMMAS = [float(g) for g in sys.argv[4].split(",")]
CFG = {"Al": dict(a0=4.0369, ecut=40, C=(113.6, 60.0, 33.0)),
       "Cu": dict(a0=None, ecut=55, C=(182.1, 118.5, 72.8)),
       "CuPAW": dict(a0=None, ecut=55, C=(170.7, 116.8, 80.2))}[EL]
if EL == "Cu":
    CFG["a0"] = json.load(open("results_cu_bulk.json"))["k24"]["a0_A"]
if EL == "CuPAW":
    CFG["a0"] = json.load(open("results_cu_paw_check.json"))["a0"]
M = MATERIALS[EL]
DEG = float(os.environ.get("DEG", "0.02"))   # smearing width (Ry)
SUF = "" if DEG == 0.02 else f"_d{DEG}"
TOL = 0.05                           # GPa, residual on the five relaxed stress components

# ---- frame: twinning sense of <112> (see shear_geometry.py) ----
R, h0 = twin_frame(CFG["a0"])                     # h0 = primitive cell in the shear frame

# ---- stiffness in the shear frame (preconditioner for the relaxation) ----
Cr = rotate(cubic_stiffness(*CFG["C"]), R)
comps = COMPS                                     # relaxed components (xx, yy, zz, xy, yz)
Kmat = relaxation_stiffness(Cr)
G_lin = Cr[0, 2, 0, 2]

os.makedirs(f"runs/02_shear_{EL}", exist_ok=True); os.chdir(f"runs/02_shear_{EL}")
out_path = f"../../results_shear_{EL}_{MODE}_k{K}{SUF}.json"
out = json.load(open(out_path)) if os.path.exists(out_path) else {"gamma": [], "tau": [], "D": [], "n_scf": [], "resid": []}
D = np.array(out["D"][-1]) if out["D"] else np.zeros(5)

def stress(g, d, tag):
    F = np.eye(3); F[0, 2] = g
    for (i, j), v in zip(comps, d):
        F[i, j] += v
        if i != j: F[j, i] += v
    h = h0 @ F.T
    name = f"{MODE}_k{K}{SUF}_g{g:.4f}_{tag}"
    write_pw_input(name + ".in", h, [M["element"]], [[0, 0, 0]], ecut=CFG["ecut"], kpts=(K, K, K), degauss=DEG,
                   outdir="./tmp_" + name, conv_thr=1e-9, **M)
    r = run_pw(name + ".in", name + ".out"); os.system(f"rm -rf tmp_{name}")
    return -r["stress_kbar"] / 10.0          # physical Cauchy stress, GPa (tension positive)

for g in GAMMAS:
    if any(abs(g - x) < 1e-9 for x in out["gamma"]):
        continue
    if MODE == "unrelaxed":
        s = stress(g, np.zeros(5), "0"); n = 1; resid = 0.0; d = np.zeros(5)
    else:
        d = D.copy(); n = 0
        while True:
            s = stress(g, d, f"it{n}"); n += 1
            other = np.array([s[i, j] for i, j in comps])
            resid = np.abs(other).max()
            print(f"  gamma={g:.3f} it={n} tau={s[0,2]:.3f} resid={resid:.3f}", flush=True)
            if resid < TOL or n >= 12:
                break
            d = d - 0.9 * np.linalg.solve(Kmat, other)          # quasi-Newton step with the elastic stiffness
        D = d
    out["gamma"].append(g); out["tau"].append(float(s[0, 2])); out["D"].append([float(v) for v in d])
    out["n_scf"].append(n); out["resid"].append(float(resid))
    out["G_linear_elastic_GPa"] = float(G_lin)
    print(f"{EL} {MODE} k={K} gamma={g:.3f} tau={s[0,2]:.3f} GPa (scf={n}, resid={resid:.3f})", flush=True)
    json.dump(out, open(out_path, "w"), indent=1)
