"""MD creep of a nanocrystalline metal: stress exponent and activation energy.

    python scripts/run_md_creep.py                  # Ni (Foiles EAM), ~2 h on 1 core
    python scripts/run_md_creep.py --quick          # 2-minute smoke test
    python scripts/run_md_creep.py --material CoAl  # B2 CoAl (Vailhe & Farkas EAM)

1. Builds a periodic Voronoi polycrystal (default 4 grains in a 4.5 nm box, d ~ 3.5 nm).
2. Stress series at fixed T and temperature series at fixed stress.
3. Every condition is repeated with several independent velocity seeds on the same
   microstructure; the steady-state rate is averaged (mean +/- standard error) - single
   runs on a few-grain box are dominated by discrete grain-boundary sliding events.
4. Fits the stress exponent n and activation energy Q to the averaged rates.
Every finished run is cached in results/<material>/runs/*.json, so an interrupted
campaign resumes where it stopped.

Outputs: results/<material>/creep_curves.csv, rates.csv, summary.json, fig_md_creep.png,
final.dump. For publication-quality numbers use larger boxes (>= 15 nm, 10-20 grains),
several microstructures and >= 1 ns runs on a cluster (lammps_inputs/in.nc_creep.lmp).
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

from mdcreep.analysis import steady_state_rate  # noqa: E402
from mdcreep.lammps_creep import CreepSettings, creep, dump, setup  # noqa: E402
from mdcreep.polycrystal import voronoi_polycrystal, write_lammps_data  # noqa: E402

MATERIALS = {
    "Ni": dict(symbols=["Ni"], a=3.52, lattice="fcc", pair_style="eam", pair_coeff="1 1 {pot}/Ni_u3.eam",
               cna=0.854 * 3.52, T=1300.0, T_series=[1200.0, 1300.0, 1400.0],
               stresses=[0.4, 0.6, 0.8, 1.0], stress_T=0.6),
    "CoAl": dict(symbols=["Co", "Al"], a=2.86, lattice="b2", pair_style="eam/alloy",
                 pair_coeff="* * {pot}/CoAl.eam.alloy Co Al", cna=1.207 * 2.86, T=1300.0,
                 T_series=[1200.0, 1300.0, 1400.0], stresses=[0.5, 1.0, 1.5, 2.0], stress_T=1.0),
}
KB_EV = 8.617333262e-5


def run_one(data, m, s, stress, cache: Path, out_dump=None):
    """One creep run, cached as JSON (returns the curve dict and GB atom fraction)."""
    if cache.exists():
        d = json.loads(cache.read_text())
        return {k: np.array(v) for k, v in d["curve"].items()}, d["gb_fraction"]
    L = setup(str(data), m["pair_style"], m["pair_coeff"].format(pot=ROOT / "potentials"), s, cna_cutoff=m["cna"])
    gb = 1.0 - L._fcc_fraction_0K
    r = creep(L, stress, s)
    if out_dump:
        dump(L, str(out_dump))
    L.close()
    cache.write_text(json.dumps({"curve": {k: v.tolist() for k, v in r.items()}, "gb_fraction": gb}))
    return r, gb


def weighted_fit(x, y, sy):
    """Weighted linear fit y = a + b x. Returns (b, sigma_b)."""
    w = 1.0 / np.maximum(np.asarray(sy), 1e-6) ** 2
    X = np.column_stack([np.ones_like(x), x])
    cov = np.linalg.inv(X.T @ (X * w[:, None]))
    beta = cov @ (X.T @ (w * y))
    return float(beta[1]), float(np.sqrt(cov[1, 1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--material", default="Ni", choices=list(MATERIALS))
    ap.add_argument("--box", type=float, default=45.0, help="box edge (A)")
    ap.add_argument("--grains", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3, help="independent runs per condition")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    m = MATERIALS[args.material]
    out = ROOT / "results" / (args.material + ("_quick" if args.quick else ""))
    (out / "runs").mkdir(parents=True, exist_ok=True)
    base = CreepSettings(T=m["T"], t_equil_ps=10.0, t_creep_ps=40.0, sample_every=250)
    if args.quick:
        args.box, args.seeds = 30.0, 1
        base.t_equil_ps, base.t_creep_ps, base.sample_every = 1.0, 2.0, 100
        m = {**m, "stresses": m["stresses"][-2:], "T_series": m["T_series"][-2:]}

    data = out / "polycrystal.data"
    atoms, gid = voronoi_polycrystal(m["symbols"], m["a"], args.box, args.grains, m["lattice"], seed=7)
    if not data.exists():
        write_lammps_data(atoms, data, m["symbols"])
    d_grain = args.box * (6 / (np.pi * args.grains)) ** (1 / 3) / 10.0  # equivalent-sphere diameter, nm
    print(f"{args.material}: {len(atoms)} atoms, {args.grains} grains, d ~ {d_grain:.1f} nm, {args.seeds} seeds")

    conditions = [(base.T, sig) for sig in m["stresses"]]
    conditions += [(T, m["stress_T"]) for T in m["T_series"] if (T, m["stress_T"]) not in conditions]
    rows, curves = [], []
    t0 = time.time()
    for T, sig in conditions:
        for k in range(args.seeds):
            s = CreepSettings(**{**base.__dict__, "T": T, "seed": 12345 + 1000 * k})
            last = (T == base.T and sig == m["stresses"][-1] and k == 0)
            r, gb = run_one(data, m, s, sig, out / "runs" / f"T{T:.0f}_s{sig:.2f}_seed{k}.json",
                            out / "final.dump" if last else None)
            rate = steady_state_rate(r["t_ps"], r["strain"])
            rows.append({"T_K": T, "stress_GPa": sig, "seed": k, "rate_per_s": rate,
                         "final_strain": float(r["strain"][-1]), "gb_fraction": gb})
            curves.append(pd.DataFrame({**r, "T_K": T, "stress_GPa": sig, "seed": k}))
            print(f"  T={T:.0f} K  sigma={sig:.2f} GPa  seed={k}  rate={rate:.3e} 1/s  ({time.time() - t0:.0f}s)",
                  flush=True)

    runs = pd.DataFrame(rows)
    pd.concat(curves).to_csv(out / "creep_curves.csv", index=False, float_format="%.6g")
    agg = runs.groupby(["T_K", "stress_GPa"])["rate_per_s"].agg(["mean", "std", "count"]).reset_index()
    agg["sem"] = agg["std"].fillna(0) / np.sqrt(agg["count"])
    agg.to_csv(out / "rates.csv", index=False, float_format="%.5g")

    ss = agg[np.isclose(agg["T_K"], base.T)].sort_values("stress_GPa")
    tt = agg[np.isclose(agg["stress_GPa"], m["stress_T"])].sort_values("T_K")
    rel = lambda d: (d["sem"] / d["mean"]).clip(lower=0.05)  # noqa: E731  relative error of ln(rate)
    n, n_err = weighted_fit(np.log(ss["stress_GPa"].to_numpy()), np.log(ss["mean"].to_numpy()), rel(ss).to_numpy())
    if len(tt) >= 2:
        slope, slope_err = weighted_fit(1.0 / (KB_EV * tt["T_K"].to_numpy()), np.log(tt["mean"].to_numpy()),
                                        rel(tt).to_numpy())
        Q, Q_err = -slope, slope_err
    else:
        Q = Q_err = float("nan")
    summary = {"material": args.material, "n_atoms": len(atoms), "grains": args.grains, "box_A": args.box,
               "grain_size_nm": d_grain, "gb_atom_fraction": float(runs["gb_fraction"].mean()),
               "seeds_per_condition": args.seeds, "stress_exponent_n": n, "stress_exponent_err": n_err,
               "activation_energy_eV": Q, "activation_energy_err_eV": Q_err,
               "rates": agg.to_dict("records"), "settings": dict(base.__dict__)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    cdf = pd.concat(curves)
    for i, sig in enumerate(m["stresses"]):
        sel = cdf[np.isclose(cdf["T_K"], base.T) & np.isclose(cdf["stress_GPa"], sig)]
        for k, g in sel.groupby("seed"):
            ax[0].plot(g["t_ps"], 100 * g["strain"], color=f"C{i}", lw=1, alpha=0.8,
                       label=f"{sig:.1f} GPa" if k == 0 else None)
    ax[0].set(xlabel="time (ps)", ylabel="creep strain (%)",
              title=f"{args.material}, d ≈ {d_grain:.1f} nm, T = {base.T:.0f} K ({args.seeds} seeds each)")
    ax[0].legend(fontsize=8)
    ax[1].errorbar(ss["stress_GPa"], ss["mean"], yerr=ss["sem"], fmt="o", capsize=3)
    xs = np.geomspace(ss["stress_GPa"].min(), ss["stress_GPa"].max(), 20)
    c = np.exp(np.mean(np.log(ss["mean"]) - n * np.log(ss["stress_GPa"])))
    ax[1].plot(xs, c * xs**n, "-", color="C0", lw=1)
    ax[1].set(xscale="log", yscale="log", xlabel="stress (GPa)", ylabel="steady-state rate (s$^{-1}$)",
              title=f"stress exponent n = {n:.1f} ± {n_err:.1f}")
    if len(tt) >= 2:
        ax[2].errorbar(1e4 / tt["T_K"], tt["mean"], yerr=tt["sem"], fmt="s", color="C3", capsize=3)
        xT = np.linspace(tt["T_K"].min(), tt["T_K"].max(), 20)
        cT = np.exp(np.mean(np.log(tt["mean"]) + Q / (KB_EV * tt["T_K"])))
        ax[2].plot(1e4 / xT, cT * np.exp(-Q / (KB_EV * xT)), "-", color="C3", lw=1)
        ax[2].set(yscale="log", xlabel="10$^4$/T (K$^{-1}$)", ylabel="rate (s$^{-1}$)",
                  title=f"σ = {m['stress_T']} GPa: Q = {Q:.2f} ± {Q_err:.2f} eV")
    for a in ax:
        a.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "fig_md_creep.png", dpi=150)
    print(json.dumps({k: summary[k] for k in ("grain_size_nm", "gb_atom_fraction", "stress_exponent_n",
                                              "stress_exponent_err", "activation_energy_eV",
                                              "activation_energy_err_eV")}, indent=1))


if __name__ == "__main__":
    main()
