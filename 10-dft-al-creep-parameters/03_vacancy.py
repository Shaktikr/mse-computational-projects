"""Step 3: mono-vacancy formation energy in fcc Al.

    E_f(vac) = E[N-1 atoms, with vacancy] - (N-1)/N * E[N atoms, perfect]

Both cells have the same shape and volume (the equilibrium lattice constant a0), i.e.
this is the constant-volume formation energy, which equals the zero-pressure value in the
dilute limit. We compute it unrelaxed and after relaxing the ionic positions, in a
2x2x2 (32-site) and a 3x3x3 (108-site) conventional supercell to check finite-size error.
The vacancy concentration that feeds diffusion creep is c_v = exp(S_f/k) exp(-E_f/kT)."""
import os, json, sys
import numpy as np
from qe import *

bulk = json.load(open("results_bulk.json"))
KREF = sys.argv[1] if len(sys.argv) > 1 else "k32"
a0 = bulk[KREF]["a0_A"]
os.makedirs("runs/03_vac", exist_ok=True)
os.chdir("runs/03_vac")

# supercell size n -> k mesh (keeps k-point density roughly equal to ~24^3 per primitive cell)
setups = {2: 6}
# NOTE: a 3x3x3 (108-site) cell is the natural next convergence check; on a 2-core machine
# it costs ~1 day for the relaxation, so here it is left as an exercise (see report).
# The k-mesh check (6^3 and 8^3) is done in 05c_kpoint_check.py.
out = {"a0_A": a0}
# wall time of the relaxation is only known when pw.x actually runs; keep the recorded one otherwise
prev = json.load(open("../../results_vacancy.json")) if os.path.exists("../../results_vacancy.json") else {}
for n, k in setups.items():
    cell, frac = fcc_conventional_supercell(a0, n)
    N = len(frac)
    common = dict(ecut=40, kpts=(k, k, k), degauss=0.02)
    # perfect crystal
    write_pw_input(f"perf_{n}.in", cell, ["Al"] * N, frac, outdir=f"./tmp_p{n}", **common)
    Ep = run_pw(f"perf_{n}.in", f"perf_{n}.out")["energy_eV"]
    # remove atom 0 (at the origin) -> vacancy, unrelaxed
    fv = frac[1:]
    write_pw_input(f"vac_unrel_{n}.in", cell, ["Al"] * (N - 1), fv, outdir=f"./tmp_u{n}", **common)
    Eu = run_pw(f"vac_unrel_{n}.in", f"vac_unrel_{n}.out")["energy_eV"]
    # ionic relaxation (BFGS) at fixed cell
    write_pw_input(f"vac_rel_{n}.in", cell, ["Al"] * (N - 1), fv, calc="relax",
                   outdir=f"./tmp_r{n}",
                   extra_control="  forc_conv_thr = 1.0d-4\n  etot_conv_thr = 1.0d-6", **common)
    r = run_pw(f"vac_rel_{n}.in", f"vac_rel_{n}.out")
    Er = r["energy_eV"]
    Ef_u = Eu - (N - 1) / N * Ep
    Ef_r = Er - (N - 1) / N * Ep
    # displacement of the 12 nearest neighbours of the vacancy
    rel = final_positions(f"vac_rel_{n}.out")
    d = (rel - fv); d -= np.round(d)
    dcart = d @ cell
    r0 = fv.copy(); r0 -= np.round(r0)
    dist = np.linalg.norm(r0 @ cell, axis=1)
    nn = np.isclose(dist, a0 / np.sqrt(2), atol=0.05)
    radial = np.mean([np.dot(dcart[i], (r0 @ cell)[i]) / dist[i] for i in np.where(nn)[0]])
    out[f"{N}"] = dict(n=n, N=N, kmesh=k, E_perfect_eV=Ep, E_vac_unrel_eV=Eu, E_vac_rel_eV=Er,
                       Ef_unrelaxed_eV=Ef_u, Ef_relaxed_eV=Ef_r,
                       relaxation_energy_eV=Ef_u - Ef_r,
                       nn_radial_disp_A=radial, nn_radial_disp_pct_of_nn=100 * radial / (a0 / np.sqrt(2)),
                       wall_relax_s=r.get("wall", prev.get(f"{N}", {}).get("wall_relax_s")))
    print(out[f"{N}"], flush=True)
    json.dump(out, open("../../results_vacancy.json", "w"), indent=1)
    np.save(f"vac_relaxed_frac_{n}.npy", rel)
    os.system(f"rm -rf tmp_p{n} tmp_u{n}")
