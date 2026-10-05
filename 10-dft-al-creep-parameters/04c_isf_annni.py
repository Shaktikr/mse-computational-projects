"""Step 4c: intrinsic stacking-fault energy from the ANNNI model (axial next-nearest-neighbour Ising).

The fault energy is expressed through the energies of three perfect close-packed stackings, each
in a small, highly symmetric hexagonal cell with ideal (111) spacing:
    fcc  = ABC   (3 atoms)
    hcp  = AB    (2 atoms)
    dhcp = ABAC  (4 atoms)
    gamma_isf = (E_hcp + 2 E_dhcp - 3 E_fcc) / A        (energies per atom, A = area per atom)
[Denteneer & van Haeringen, J. Phys. C 20, L883 (1987)]
Because the cells are tiny and symmetric, the k-point convergence that is so expensive in the
tilted 12-layer GSFE cell can be done properly here (in-plane meshes up to 48x48)."""
import os, json, sys
import numpy as np
from qe import *

a0 = json.load(open("results_bulk.json"))["k40"]["a0_A"]
ann, d111 = a0 / np.sqrt(2), a0 / np.sqrt(3)
a1 = ann * np.array([1.0, 0, 0]); a2 = ann * np.array([0.5, np.sqrt(3) / 2, 0])
A = np.linalg.norm(np.cross(a1, a2))
EVA2_TO_MJM2 = 16021.766
site = {"A": (0, 0), "B": (1 / 3, 1 / 3), "C": (2 / 3, 2 / 3)}
stacks = {"fcc": "ABC", "hcp": "AB", "dhcp": "ABAC"}
os.makedirs("runs/04c_annni", exist_ok=True)
os.chdir("runs/04c_annni")
out = json.load(open("../../results_isf_annni.json")) if os.path.exists("../../results_isf_annni.json") else {}
for K in [int(v) for v in (sys.argv[1:] or ["16", "24", "32", "40"])]:
    E = {}
    for name, seq in stacks.items():
        n = len(seq)
        cell = np.array([a1, a2, [0, 0, n * d111]])
        frac = [[site[s][0], site[s][1], i / n] for i, s in enumerate(seq)]
        kz = max(1, int(round(K * ann * np.sqrt(3) / 2 / (n * d111))))   # same k-spacing along c
        tag = f"{name}_k{K}"
        write_pw_input(tag + ".in", cell, ["Al"] * n, frac, ecut=40, kpts=(K, K, kz), degauss=0.02,
                       outdir="./tmp_" + tag, conv_thr=1e-10)
        E[name] = run_pw(tag + ".in", tag + ".out")["energy_eV"] / n
        os.system(f"rm -rf tmp_{tag}")
    g = (E["hcp"] + 2 * E["dhcp"] - 3 * E["fcc"]) / A * EVA2_TO_MJM2
    out[f"k{K}"] = dict(gamma_isf_mJm2=float(g), dE_hcp_meV=float((E["hcp"] - E["fcc"]) * 1000),
                        dE_dhcp_meV=float((E["dhcp"] - E["fcc"]) * 1000))
    print(K, out[f"k{K}"], flush=True)
    json.dump(out, open("../../results_isf_annni.json", "w"), indent=1)
