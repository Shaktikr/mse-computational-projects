"""Stacking-fault energies for Al or Cu.
(a) ANNNI:   gamma_isf = (E_hcp + 2 E_dhcp - 3 E_fcc) / A   (2-4 atom cells, cheap, dense k)
(b) GSFE:    tilted NL-layer cell (one fault per cell), rigid shift u = f * b_p along <112>,
             with nbnd set explicitly (the default is too small for long cells - see the Al project),
             plus a z-only relaxation at f = 0.6 and 1.0 (perpendicular relaxation, GSFE convention).
usage: python3 03_sfe.py Cu annni 16,24,32
       python3 03_sfe.py Cu gsfe  K NL NBND"""
import os, json, sys
import numpy as np
from qe import *

EL, MODE = sys.argv[1], sys.argv[2]
M = MATERIALS[EL]
a0 = {"Al": lambda: 4.0369, "Cu": lambda: json.load(open("results_cu_bulk.json"))["k24"]["a0_A"],
      "CuPAW": lambda: json.load(open("results_cu_paw_check.json"))["a0"]}[EL]()
ECUT = {"Al": 40, "Cu": 55, "CuPAW": 55}[EL]
SYM = M["element"]
ann, d111 = a0 / np.sqrt(2), a0 / np.sqrt(3)
a1 = ann * np.array([1.0, 0, 0]); a2 = ann * np.array([0.5, np.sqrt(3) / 2, 0]); bp = (a1 + a2) / 3
A = np.linalg.norm(np.cross(a1, a2)); EVA2_TO_MJM2 = 16021.766
site = {"A": (0, 0), "B": (1 / 3, 1 / 3), "C": (2 / 3, 2 / 3)}
os.makedirs(f"runs/03_sfe_{EL}", exist_ok=True); os.chdir(f"runs/03_sfe_{EL}")
path = f"../../results_sfe_{EL}.json"
out = json.load(open(path)) if os.path.exists(path) else {}

if MODE == "annni":
    for K in [int(v) for v in sys.argv[3].split(",")]:
        E = {}
        for name, seq in {"fcc": "ABC", "hcp": "AB", "dhcp": "ABAC"}.items():
            n = len(seq); cell = np.array([a1, a2, [0, 0, n * d111]])
            frac = [[site[s][0], site[s][1], i / n] for i, s in enumerate(seq)]
            kz = max(1, int(round(K * ann * np.sqrt(3) / 2 / (n * d111))))
            tag = f"annni_{name}_k{K}"
            write_pw_input(tag + ".in", cell, [SYM] * n, frac, ecut=ECUT, kpts=(K, K, kz), degauss=0.02,
                           outdir="./tmp_" + tag, conv_thr=1e-10, nbnd=int(n * {"Al": 4, "Cu": 10, "CuPAW": 10}[EL]), **M)
            E[name] = run_pw(tag + ".in", tag + ".out")["energy_eV"] / n; os.system(f"rm -rf tmp_{tag}")
        g = (E["hcp"] + 2 * E["dhcp"] - 3 * E["fcc"]) / A * EVA2_TO_MJM2
        out.setdefault("annni", {})[f"k{K}"] = dict(gamma_isf=float(g), dE_hcp_meV=float((E["hcp"] - E["fcc"]) * 1e3),
                                                     dE_dhcp_meV=float((E["dhcp"] - E["fcc"]) * 1e3))
        print(EL, "ANNNI", K, out["annni"][f"k{K}"], flush=True)
        json.dump(out, open(path, "w"), indent=1)

if MODE == "gsfe":
    K, NL, NB = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    RELAX_F = ([] if sys.argv[6] == "none" else [float(x) for x in sys.argv[6].split(",")]) if len(sys.argv) > 6 else [0.6, 1.0]
    FLIST = [0.0, 1.0, 0.6, 0.5, 0.7] if len(sys.argv) <= 7 else [float(x) for x in sys.argv[7].split(",")]
    KZ = max(1, int(round(K * ann * np.sqrt(3) / 2 / (NL * d111))))
    cart = np.array([np.array(site["ABC"[i % 3]][0] * a1 + site["ABC"[i % 3]][1] * a2) + [0, 0, i * d111]
                     for i in range(NL)])
    key = f"gsfe_k{K}_NL{NL}"
    res = out.setdefault(key, {"kz": KZ, "rigid": {}, "relaxed": {}})
    common = dict(ecut=ECUT, kpts=(K, K, KZ), degauss=0.02, nbnd=NB, **M)
    E0 = None
    for f in FLIST:
        cell = np.array([a1, a2, np.array([0, 0, NL * d111]) + f * bp]); frac = cart @ np.linalg.inv(cell)
        tag = f"{key}_f{f:.2f}"
        write_pw_input(tag + ".in", cell, [SYM] * NL, frac, outdir="./tmp_" + tag, **common)
        e = run_pw(tag + ".in", tag + ".out")["energy_eV"]; os.system(f"rm -rf tmp_{tag}")
        if f == 0.0:
            E0 = e; res["E0_per_atom"] = e / NL
        res["rigid"][f"{f:.2f}"] = (e - E0) / A * EVA2_TO_MJM2
        print(EL, key, "rigid", f, res["rigid"][f"{f:.2f}"], flush=True)
        json.dump(out, open(path, "w"), indent=1)
        if f in RELAX_F:            # perpendicular relaxation (Al: 0.6 and 1.0; Cu: 0.6 only, to save time)
            rtag = tag + "_relax"
            write_pw_input(rtag + ".in", cell, [SYM] * NL, frac, calc="relax", outdir="./tmp_" + rtag,
                           fixed=[(0, 0, 1)] * NL,
                           extra_control="  forc_conv_thr = 1.0d-3\n  etot_conv_thr = 1.0d-5", **common)
            er = run_pw(rtag + ".in", rtag + ".out")["energy_eV"]; os.system(f"rm -rf tmp_{rtag}")
            res["relaxed"][f"{f:.2f}"] = (er - E0) / A * EVA2_TO_MJM2
            print(EL, key, "relaxed", f, res["relaxed"][f"{f:.2f}"], flush=True)
            json.dump(out, open(path, "w"), indent=1)
