"""MD creep of a nanocrystalline metal: stress exponent and activation energy.

    python scripts/run_md_creep.py                  # Ni (Foiles EAM), ~40-60 min on 2 cores
    python scripts/run_md_creep.py --quick          # 2-minute smoke test
    python scripts/run_md_creep.py --material CoAl  # B2 CoAl (Vailhe & Farkas EAM)

1. Builds a periodic Voronoi polycrystal (default 4 grains in a 4.5 nm box, d ~ 3 nm).
2. Stress series at fixed T  -> steady-state rates -> stress exponent n.
3. Temperature series at fixed stress -> activation energy Q.
Outputs: results/<material>/creep_curves.csv, summary.json, fig_md_creep.png, final.dump

For publication-quality numbers use larger boxes (>= 15 nm, 10-20 grains), several
random seeds per condition and longer runs (>= 1 ns) on a cluster - see
lammps_inputs/in.nc_creep.lmp for a plain LAMMPS input to run with mpirun.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mdcreep.analysis import activation_energy, steady_state_rate, stress_exponent  # noqa: E402
from mdcreep.lammps_creep import CreepSettings, creep, dump, setup  # noqa: E402
from mdcreep.polycrystal import voronoi_polycrystal, write_lammps_data  # noqa: E402

MATERIALS = {
    "Ni": dict(symbols=["Ni"], a=3.52, lattice="fcc", pair_style="eam", pair_coeff="1 1 {pot}/Ni_u3.eam",
               cna=0.854 * 3.52, T=1300.0, T_series=[1200.0, 1300.0, 1400.0],
               stresses=[0.6, 0.8, 1.0, 1.2], stress_T=1.0),
    "CoAl": dict(symbols=["Co", "Al"], a=2.86, lattice="b2", pair_style="eam/alloy",
                 pair_coeff="* * {pot}/CoAl.eam.alloy Co Al", cna=1.207 * 2.86, T=1300.0,
                 T_series=[1200.0, 1300.0, 1400.0], stresses=[0.5, 1.0, 1.5, 2.0], stress_T=1.0),
}


def run_one(data, m, s, stress, out_dump=None):
    L = setup(str(data), m["pair_style"], m["pair_coeff"].format(pot=ROOT / "potentials"), s, cna_cutoff=m["cna"])
    gb = 1.0 - L._fcc_fraction_0K
    r = creep(L, stress, s)
    if out_dump:
        dump(L, str(out_dump))
    L.close()
    return r, gb


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--material", default="Ni", choices=list(MATERIALS))
    ap.add_argument("--box", type=float, default=45.0, help="box edge (A)")
    ap.add_argument("--grains", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    m = MATERIALS[args.material]
    out = ROOT / "results" / args.material
    out.mkdir(parents=True, exist_ok=True)
    s = CreepSettings(T=m["T"], t_equil_ps=10.0, t_creep_ps=40.0, sample_every=250)
    if args.quick:
        args.box, s.t_equil_ps, s.t_creep_ps, s.sample_every = 30.0, 1.0, 2.0, 100
        m = {**m, "stresses": m["stresses"][-2:], "T_series": m["T_series"][-2:]}

    atoms, gid = voronoi_polycrystal(m["symbols"], m["a"], args.box, args.grains, m["lattice"], seed=7)
    data = out / "polycrystal.data"
    write_lammps_data(atoms, data, m["symbols"])
    d_grain = args.box * (6 / (np.pi * args.grains)) ** (1 / 3) / 10.0  # equivalent-sphere diameter, nm
    print(f"{args.material}: {len(atoms)} atoms, {args.grains} grains, d ~ {d_grain:.1f} nm")

    rows, curves = [], []
    t0 = time.time()
    for i, sig in enumerate(m["stresses"]):  # stress series at fixed T
        r, gb = run_one(data, m, s, sig, out / "final.dump" if i == len(m["stresses"]) - 1 else None)
        rate = steady_state_rate(r["t_ps"], r["strain"])
        rows.append({"series": "stress", "T_K": s.T, "stress_GPa": sig, "rate_per_s": rate, "gb_fraction": gb})
        curves.append(pd.DataFrame({**r, "series": "stress", "T_K": s.T, "stress_GPa": sig}))
        print(f"  T={s.T:.0f} K  sigma={sig:.2f} GPa  rate={rate:.3e} 1/s  ({time.time() - t0:.0f}s)")
    for T in m["T_series"]:  # temperature series at fixed stress
        if np.isclose(T, s.T) and np.isclose(m["stress_T"], m["stresses"]).any():
            continue  # already computed in the stress series
        sT = CreepSettings(**{**s.__dict__, "T": T})
        r, gb = run_one(data, m, sT, m["stress_T"])
        rate = steady_state_rate(r["t_ps"], r["strain"])
        rows.append({"series": "temperature", "T_K": T, "stress_GPa": m["stress_T"], "rate_per_s": rate, "gb_fraction": gb})
        curves.append(pd.DataFrame({**r, "series": "temperature", "T_K": T, "stress_GPa": m["stress_T"]}))
        print(f"  T={T:.0f} K  sigma={m['stress_T']:.2f} GPa  rate={rate:.3e} 1/s  ({time.time() - t0:.0f}s)")

    res = pd.DataFrame(rows)
    pd.concat(curves).to_csv(out / "creep_curves.csv", index=False, float_format="%.6g")
    ss = res[np.isclose(res["T_K"], s.T)]
    n, _ = stress_exponent(ss["stress_GPa"], ss["rate_per_s"])
    tt = res[np.isclose(res["stress_GPa"], m["stress_T"])]
    Q = activation_energy(tt["T_K"], tt["rate_per_s"]) if len(tt) >= 2 else float("nan")
    summary = {"material": args.material, "n_atoms": len(atoms), "grains": args.grains, "box_A": args.box,
               "grain_size_nm": d_grain, "gb_atom_fraction": float(res["gb_fraction"].mean()),
               "stress_exponent_n": n, "activation_energy_eV": Q, "runs": rows,
               "settings": {k: v for k, v in s.__dict__.items()}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for c in curves:
        if c["series"].iloc[0] == "stress":
            ax[0].plot(c["t_ps"], 100 * c["strain"], label=f"{c['stress_GPa'].iloc[0]:.2f} GPa")
    ax[0].set(xlabel="time (ps)", ylabel="creep strain (%)", title=f"{args.material}, d ≈ {d_grain:.1f} nm, T = {s.T:.0f} K")
    ax[0].legend(fontsize=8)
    ax[1].loglog(ss["stress_GPa"], ss["rate_per_s"], "o-")
    ax[1].set(xlabel="stress (GPa)", ylabel="steady-state rate (s$^{-1}$)", title=f"stress exponent n = {n:.2f}")
    if len(tt) >= 2:
        ax[2].semilogy(1e4 / tt["T_K"], tt["rate_per_s"], "s-", color="C3")
        ax[2].set(xlabel="10$^4$/T (K$^{-1}$)", ylabel="rate (s$^{-1}$)", title=f"σ = {m['stress_T']} GPa: Q = {Q:.2f} eV")
    for a in ax:
        a.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "fig_md_creep.png", dpi=150)
    print(json.dumps({k: summary[k] for k in ("grain_size_nm", "gb_atom_fraction", "stress_exponent_n", "activation_energy_eV")}, indent=1))


if __name__ == "__main__":
    main()
