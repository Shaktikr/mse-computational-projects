"""Full creep analysis of a folder of creep curves.

    python scripts/analyze_creep.py data/synthetic_single_phase
    python scripts/analyze_creep.py data/synthetic_composite --E-RT 210 --dEdT -0.07
    python scripts/analyze_creep.py path/to/your/lab/data --name "CoCrFeNi, SPS"

Produces in results/<folder name>/:
    summary.csv           one row per test: sigma, T, min rate, t_r, strain at fracture ...
    report.json           fitted n, Q, threshold stress, Larson-Miller C, Monkman-Grant m
    fig_curves.png        strain-time and strain-rate curves
    fig_norton.png        ln(rate) vs ln(sigma) at each temperature
    fig_arrhenius.png     ln(rate) vs 1/T at fixed stress (interpolated from Norton fits)
    fig_threshold.png     rate^(1/n) vs sigma (threshold stress)
    fig_lmp.png           Larson-Miller master curve
    fig_monkman_grant.png Monkman-Grant plot
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from creepkit import constitutive as cs  # noqa: E402
from creepkit.curves import creep_stages, minimum_creep_rate, strain_rate  # noqa: E402
from creepkit.io import load_folder  # noqa: E402

plt.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3})


def temperature_colors(temps):
    cmap = plt.get_cmap("plasma")
    temps = sorted(temps)
    return {T: cmap(0.1 + 0.75 * i / max(len(temps) - 1, 1)) for i, T in enumerate(temps)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--name", default=None, help="material label for plot titles")
    ap.add_argument("--E-RT", type=float, default=210.0, help="Young's modulus at RT (GPa)")
    ap.add_argument("--dEdT", type=float, default=-0.07, help="dE/dT (GPa/K)")
    ap.add_argument("--window", type=float, default=0.05, help="Savitzky-Golay window fraction")
    args = ap.parse_args()

    folder = Path(args.folder)
    tests = load_folder(folder)
    if not tests:
        sys.exit(f"no creep curves found in {folder}")
    name = args.name or tests[0].metadata.get("alloy", folder.name)
    out = ROOT / "results" / folder.name
    out.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- per-test summary
    rows = []
    for t in tests:
        m = minimum_creep_rate(t, args.window)
        st = creep_stages(t, window_frac=args.window)
        rows.append({
            "specimen": t.specimen_id, "stress_MPa": t.stress_MPa, "T_C": t.temperature_C,
            "T_K": t.temperature_K, "min_rate_per_s": m.rate_per_s, "time_at_min_h": m.time_h,
            "strain_at_min": m.strain, "rupture_time_h": t.rupture_time_h,
            "strain_at_fracture": t.strain_at_fracture, **st,
        })
    df = pd.DataFrame(rows).sort_values(["T_C", "stress_MPa"])
    df.to_csv(out / "summary.csv", index=False, float_format="%.5g")
    temps = sorted(df["T_C"].unique())
    colors = temperature_colors(temps)
    E_of_T = cs.youngs_modulus_linear(args.E_RT, args.dEdT)
    report = {"material": name, "n_tests": len(df)}

    # ---------------------------------------------------------------- curves
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for t in sorted(tests, key=lambda x: (x.temperature_C, x.stress_MPa)):
        c = colors[t.temperature_C]
        ax[0].plot(t.time_h, 100 * t.strain, color=c, lw=1)
        tt, ys, rate = strain_rate(t.time_s, t.strain, args.window)
        k = slice(int(0.03 * len(tt)), int(0.97 * len(tt)))
        ax[1].semilogy(100 * ys[k], rate[k], color=c, lw=1)
    for T in temps:
        ax[0].plot([], [], color=colors[T], label=f"{T:.0f} °C")
    ax[0].set(xlabel="time (h)", ylabel="strain (%)", xscale="log", title="Creep curves")
    ax[0].set_xlim(left=0.05)
    ax[0].legend(fontsize=8)
    ax[1].set(xlabel="strain (%)", ylabel="strain rate (s$^{-1}$)", title="Strain rate vs strain")
    fig.suptitle(name, fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_curves.png")
    plt.close(fig)

    # ---------------------------------------------------------------- Norton per temperature
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    norton = {}
    for T in temps:
        d = df[df["T_C"] == T]
        n, f = cs.fit_norton(d["stress_MPa"], d["min_rate_per_s"])
        norton[T] = f
        ax.loglog(d["stress_MPa"], d["min_rate_per_s"], "o", color=colors[T])
        xs = np.geomspace(d["stress_MPa"].min() * 0.9, d["stress_MPa"].max() * 1.1, 20)
        ax.loglog(xs, np.exp(f.intercept) * xs**f.slope, "-", color=colors[T], label=f"{T:.0f} °C: n = {n:.1f}")
    ax.set(xlabel="stress σ (MPa)", ylabel=r"minimum creep rate $\dot\varepsilon_{min}$ (s$^{-1}$)", title="Norton plot")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_norton.png")
    plt.close(fig)
    report["norton_n_by_T"] = {f"{T:.0f}C": norton[T].slope for T in temps}

    # ---------------------------------------------------------------- Arrhenius at common stresses
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    lo = max(df.groupby("T_C")["stress_MPa"].min())
    hi = min(df.groupby("T_C")["stress_MPa"].max())
    s_common = np.geomspace(lo, hi, 3) if hi > lo else [np.median(df["stress_MPa"])]
    arr = {}
    for i, s in enumerate(s_common):
        Ts = np.array(temps) + 273.15
        rates = [np.exp(norton[T].intercept) * s ** norton[T].slope for T in temps]
        Q, Qe, f = cs.fit_arrhenius(Ts, rates)
        arr[f"{s:.0f}MPa"] = Q
        ax.semilogy(1e4 / Ts, rates, "o-", color=f"C{i}", label=f"σ = {s:.0f} MPa: Q = {Q:.0f} kJ/mol")
    ax.set(xlabel="10$^4$/T (K$^{-1}$)", ylabel=r"$\dot\varepsilon_{min}$ (s$^{-1}$)",
           title="Arrhenius plot (interpolated at constant σ)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_arrhenius.png")
    plt.close(fig)
    report["arrhenius_Q_kJ_by_stress"] = arr

    # ---------------------------------------------------------------- global fits
    pl = cs.fit_power_law(df["stress_MPa"], df["T_K"], df["min_rate_per_s"])
    plc = cs.fit_power_law(df["stress_MPa"], df["T_K"], df["min_rate_per_s"], E_of_T)
    report["global_power_law"] = pl.to_dict()
    report["global_power_law_modulus_compensated"] = plc.to_dict()

    # ---------------------------------------------------------------- threshold stress
    n_true, sig_th, r2 = cs.threshold_stress_by_temperature(df["stress_MPa"], df["T_K"], df["min_rate_per_s"])
    report["threshold"] = {"n_true": n_true, "sigma_th_MPa_by_T": {f"{k - 273.15:.0f}C": v for k, v in sig_th.items()},
                           "mean_r2_by_trial_n": {str(k): v for k, v in r2.items()}}
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for T in temps:
        d = df[df["T_C"] == T]
        y = d["min_rate_per_s"] ** (1.0 / n_true)
        f = cs._linfit(d["stress_MPa"], y)
        xs = np.linspace(0, d["stress_MPa"].max() * 1.05, 50)
        ax.plot(d["stress_MPa"], y, "o", color=colors[T])
        ax.plot(xs, f.intercept + f.slope * xs, "--", color=colors[T],
                label=f"{T:.0f} °C: σ$_{{th}}$ = {sig_th[T + 273.15]:.0f} MPa")
    ax.set_ylim(bottom=0)
    ax.set(xlabel="σ (MPa)", ylabel=fr"$\dot\varepsilon_{{min}}^{{1/{n_true:g}}}$",
           title=f"Threshold stress (best common n = {n_true:g})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_threshold.png")
    plt.close(fig)

    # effective-stress analysis if a meaningful threshold exists
    sth = np.array([max(sig_th[T], 0.0) for T in df["T_K"]])
    if np.all(sth < df["stress_MPa"].to_numpy()) and np.median(sth) > 0.05 * df["stress_MPa"].median():
        pe = cs.fit_power_law(df["stress_MPa"].to_numpy() - sth, df["T_K"], df["min_rate_per_s"], E_of_T)
        report["effective_stress_power_law"] = pe.to_dict()

    # ---------------------------------------------------------------- rupture correlations
    d = df.dropna(subset=["rupture_time_h"])
    if len(d) >= 4:
        C, poly, rms = cs.fit_larson_miller_constant(d["stress_MPa"], d["T_K"], d["rupture_time_h"])
        report["larson_miller"] = {"C_fitted": C, "poly_log10_sigma_vs_LMP": list(map(float, poly)), "rms_log10_sigma": rms}
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        for T in temps:
            dd = d[d["T_C"] == T]
            ax.semilogy(cs.larson_miller(dd["T_K"], dd["rupture_time_h"], C), dd["stress_MPa"], "o", color=colors[T], label=f"{T:.0f} °C")
        x = cs.larson_miller(d["T_K"], d["rupture_time_h"], C)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.semilogy(xs, 10 ** np.polyval(poly, xs), "k-", lw=1, label="master curve")
        ax.set(xlabel=f"LMP = T(C + log t$_r$)/1000,  C = {C:.1f}", ylabel="σ (MPa)", title="Larson–Miller")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "fig_lmp.png")
        plt.close(fig)

        m, CMG, f = cs.fit_monkman_grant(d["min_rate_per_s"], d["rupture_time_h"])
        report["monkman_grant"] = {"m": m, "C_MG": CMG, "r2": f.r2}
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        for T in temps:
            dd = d[d["T_C"] == T]
            ax.loglog(dd["min_rate_per_s"], dd["rupture_time_h"], "o", color=colors[T], label=f"{T:.0f} °C")
        xs = np.geomspace(d["min_rate_per_s"].min(), d["min_rate_per_s"].max(), 20)
        ax.loglog(xs, CMG / xs**m / 3600.0, "k-", lw=1, label=f"m = {m:.2f}, C$_{{MG}}$ = {CMG:.3f}")
        ax.set(xlabel=r"$\dot\varepsilon_{min}$ (s$^{-1}$)", ylabel="t$_r$ (h)", title="Monkman–Grant")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "fig_monkman_grant.png")
        plt.close(fig)

    # if synthetic, record the truth for comparison
    truth = {k: tests[0].metadata[k] for k in ("true_n", "true_Q_kJ") if k in tests[0].metadata}
    if truth:
        report["synthetic_truth"] = truth
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps({k: report[k] for k in ("global_power_law", "threshold") if k in report}, indent=1, default=float))
    print(f"results in {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
