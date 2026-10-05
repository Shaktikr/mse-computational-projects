"""Step 4: generalized stacking-fault energy (GSFE) curve of Al on {111} along <112>.

Geometry: a periodic stack of NL close-packed (111) layers (ABCABC...).
In-plane vectors a1, a2 (length a0/sqrt2, 60 deg apart); c is normal to (111).
Tilting the c-vector by u = f * b_p (b_p = a0/sqrt6, the Shockley partial) shears the crystal
across ONE (111) plane per periodic cell -> exactly one fault per cell, no free surfaces.
   f = 0   : perfect crystal
   f ~ 0.5 : unstable stacking fault energy gamma_us (barrier to nucleate a partial)
   f = 1   : intrinsic stacking fault gamma_isf (ABC|BCA -> local hcp stacking)
gamma(f) = [E(f) - E(0)] / A,  A = |a1 x a2|.
Atoms are relaxed only perpendicular to the fault plane (standard GSFE convention) at the key
points f = 0.2, 0.5 and 1.0; elsewhere the rigid-shift energy is computed (relaxation lowers gamma
by only ~2-4% in Al, see f = 0.2)."""
import os, json, sys
import numpy as np
from qe import *

bulk = json.load(open("results_bulk.json"))
a0 = bulk["k32"]["a0_A"]
NL = 12
ann = a0 / np.sqrt(2)          # nearest-neighbour distance
d111 = a0 / np.sqrt(3)         # (111) interplanar spacing
a1 = ann * np.array([1.0, 0, 0])
a2 = ann * np.array([0.5, np.sqrt(3) / 2, 0])
bp = (a1 + a2) / 3             # Shockley partial vector, |bp| = a0/sqrt6
area = np.linalg.norm(np.cross(a1, a2))
EVA2_TO_MJM2 = 16021.766

shift = {0: (0, 0), 1: (1 / 3, 1 / 3), 2: (2 / 3, 2 / 3)}     # A, B, C
cart = np.array([shift[i % 3][0] * a1 + shift[i % 3][1] * a2 + [0, 0, i * d111] for i in range(NL)])

os.makedirs("runs/04_gsfe", exist_ok=True)
os.chdir("runs/04_gsfe")
fs = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
RELAX = {0.0, 0.2, 0.5, 0.6, 1.0}   # full (perpendicular) relaxation here; single SCF (rigid shift) elsewhere
K = (16, 16, 1)      # c is ~28 A long, so one k-point along c is enough
res = {"a0": a0, "NL": NL, "area_A2": area, "f": [], "E_unrel": [], "E_rel": [], "relaxed": []}
for f in fs:
    c = np.array([0, 0, NL * d111]) + f * bp
    cell = np.array([a1, a2, c])
    frac = cart @ np.linalg.inv(cell)        # same Cartesian atoms; only the periodic image shifts
    tag = f"f{f:.2f}"
    common = dict(ecut=40, kpts=K, degauss=0.02)
    if f in RELAX:
        write_pw_input(f"rel_{tag}.in", cell, ["Al"] * NL, frac, calc="relax", outdir="./tmp_r" + tag,
                       fixed=[(0, 0, 1)] * NL,
                       extra_control="  forc_conv_thr = 1.0d-3\n  etot_conv_thr = 1.0d-5", **common)
        r = run_pw(f"rel_{tag}.in", f"rel_{tag}.out")
        Er = r["energy_eV"]
        Eu = r["energies_eV"][0]      # first SCF of the relaxation = unrelaxed (rigid shift) energy
    else:
        write_pw_input(f"scf_{tag}.in", cell, ["Al"] * NL, frac, outdir="./tmp_r" + tag, **common)
        Eu = run_pw(f"scf_{tag}.in", f"scf_{tag}.out")["energy_eV"]
        Er = None
    res["f"].append(f); res["E_unrel"].append(Eu); res["E_rel"].append(Er); res["relaxed"].append(f in RELAX)
    E0u, E0r = res["E_unrel"][0], res["E_rel"][0]
    res["gamma_unrel_mJm2"] = [(e - E0u) / area * EVA2_TO_MJM2 for e in res["E_unrel"]]
    res["gamma_rel_mJm2"] = [None if e is None else (e - E0r) / area * EVA2_TO_MJM2 for e in res["E_rel"]]
    print(f"f={f:.2f}  gamma_unrelaxed={res['gamma_unrel_mJm2'][-1]:7.1f}  "
          f"gamma_relaxed={res['gamma_rel_mJm2'][-1]}", flush=True)
    os.system(f"rm -rf tmp_r{tag}")
    json.dump(res, open("../../results_gsfe.json", "w"), indent=1)
