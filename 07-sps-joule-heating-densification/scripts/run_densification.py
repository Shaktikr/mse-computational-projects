"""Densification kinetics during SPS: extracting n, Q and an effective diffusivity.

    python scripts/run_densification.py

Uses SYNTHETIC densification curves generated from a known creep-type model (so that the
recovered n, Q and D_eff can be checked against the truth), then demonstrates:
  A. Master sintering curve (MSC) from constant-heating-rate runs -> Q
  B. Bernard-Granger analysis of isothermal holds -> n, Q, D_eff(T)
  C. The bias introduced when the PYROMETER temperature (die surface) is used instead of
     the true sample temperature - using the Joule-heating results of run_electrothermal.py.
To analyse real data, replace the simulated (t, T, D) arrays by your SPS log: D from the
punch displacement (corrected for thermal expansion of the tooling with a blank run) and
the final density measured by Archimedes' method.
"""

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

from spsmodel.densification import (DensificationModel, activation_energy_isothermal, densification_rate,  # noqa: E402
                                    effective_diffusivity, fit_msc_activation_energy, heating_program, msc_scatter,
                                    stress_exponent, work_of_sintering)

plt.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3})
SIGMA = 50e6  # applied macroscopic stress (Pa)
HOLDS = [1073.0, 1098.0, 1123.0, 1148.0]  # hold temperatures (K) with overlapping density ranges


def main():
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(1)
    m = DensificationModel()
    report = {"true_model": {"n": m.n, "p": m.p, "Q_kJ": m.Q_kJ, "D0_diff": m.D0_diff, "G_m": m.G},
              "note": "SYNTHETIC densification curves generated from the 'true' model above"}

    # ======================================================== A. master sintering curve
    curves = []
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for i, rate in enumerate([50, 100, 200, 400]):
        Tf, dur = heating_program(rate, 1273.0, 0.0)
        t = np.linspace(0, dur, 2000)
        D = m.simulate(t, Tf, SIGMA)
        D = D + rng.normal(0, 3e-4, D.size)  # displacement-measurement noise
        T = Tf(t)
        curves.append((t, T, D))
        ax[0].plot(T - 273.15, D, color=f"C{i}", label=f"{rate} K/min")
    ax[0].set(xlabel="T (°C)", ylabel="relative density D", title="Constant heating rate runs (σ = 50 MPa)")
    ax[0].legend(fontsize=8)
    Qs = np.linspace(100, 320, 45)
    sc = [msc_scatter(curves, q) for q in Qs]
    Q_msc = fit_msc_activation_energy(curves)
    ax[1].semilogy(Qs, sc, "k-")
    ax[1].axvline(Q_msc, color="C3", ls="--", label=f"best Q = {Q_msc:.0f} kJ/mol")
    ax[1].axvline(m.Q_kJ, color="0.5", ls=":", label=f"true Q = {m.Q_kJ:.0f} kJ/mol")
    ax[1].set(xlabel="trial Q (kJ/mol)", ylabel="mean variance of log$_{10}$ Θ", title="MSC: choosing Q")
    ax[1].legend(fontsize=8)
    for i, (t, T, D) in enumerate(curves):
        ax[2].plot(np.log10(work_of_sintering(t, T, Q_msc) + 1e-300), D, color=f"C{i}")
    ax[2].set(xlabel="log$_{10}$ Θ (s/K)", ylabel="D", title="Master sintering curve (collapsed)")
    end = max(np.log10(work_of_sintering(t, T, Q_msc)[-1]) for t, T, _ in curves)
    ax[2].set_xlim(end - 4.0, end + 0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_master_sintering_curve.png")
    plt.close(fig)
    report["MSC_Q_kJ"] = Q_msc

    # ======================================================== B. isothermal holds (Bernard-Granger)
    holds, n_vals = [], []
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for i, Th in enumerate(HOLDS):
        Tf, dur = heating_program(100.0, Th, 1200.0)
        t = np.linspace(0, dur, 3000)
        D = m.simulate(t, Tf, SIGMA) + rng.normal(0, 1e-4, t.size)
        k = t >= dur - 1200.0
        tk, Dk = t[k], D[k]
        holds.append((Th, tk, Dk))
        n, x, y = stress_exponent(tk, Dk, Th, SIGMA, m.E(Th), m.D_green, D_window=(Dk[0] + 0.01, 0.97))
        n_vals.append(n)
        ax[0].plot(tk - tk[0], Dk, color=f"C{i}", label=f"hold {Th - 273.15:.0f} °C")
        ax[1].plot(x, y, ".", ms=2, color=f"C{i}", label=f"n = {n:.2f}")
    ax[0].set(xlabel="hold time (s)", ylabel="D", title="Isothermal holds (after 100 K/min ramp)")
    ax[0].legend(fontsize=8)
    ax[1].set(xlabel="ln(σ$_{eff}$/μ$_{eff}$)", ylabel="ln[(1/μ$_{eff}$) dD/dt]", title="Stress exponent n")
    ax[1].legend(fontsize=8)
    n_mean = float(np.mean(n_vals))
    # a density reached during every hold (middle of the common range)
    D_ref = 0.5 * (max(h[2][0] for h in holds) + min(h[2][-1] for h in holds))
    Q_bg, xs, ys = activation_energy_isothermal(holds, n_mean, D_ref, m.E, m.D_green, SIGMA)
    # effective diffusivity at D_ref for each hold
    Deff = []
    for Th, tk, Dk in holds:
        rate = np.interp(D_ref, Dk, densification_rate(tk, Dk))
        Deff.append(effective_diffusivity(rate, D_ref, Th, SIGMA, m.E(Th), m.D_green, n_mean, m.p, m.b, m.G))
    Tline = np.linspace(1000, 1300, 50)
    ax[2].semilogy(1e4 / Tline, m.D_eff(Tline), "k-", label="true D$_{eff}$")
    ax[2].semilogy(1e4 / np.array([h[0] for h in holds]), Deff, "o", color="C3",
                   label=f"from densification data\n(n = {n_mean:.2f}, Q = {Q_bg:.0f} kJ/mol)")
    ax[2].set(xlabel="10$^4$/T (K$^{-1}$)", ylabel="D$_{eff}$ (m$^2$/s)", title=f"Effective diffusivity (D = {D_ref:.2f})")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_bernard_granger_analysis.png")
    plt.close(fig)
    report["bernard_granger"] = {"n_per_hold": n_vals, "n_mean": n_mean, "D_ref": D_ref, "Q_kJ": Q_bg,
                                 "D_eff_estimated": dict(zip([f"{h[0]:.0f}K" for h in holds], map(float, Deff))),
                                 "D_eff_true": {f"{h[0]:.0f}K": float(m.D_eff(h[0])) for h in holds}}

    # ======================================================== C. pyrometer vs true sample temperature
    hist_file = out / "history_Ni_compact.csv"
    if hist_file.exists():
        h = pd.read_csv(hist_file)
        T_true = lambda tt: np.interp(tt, h["t"], h["T_centre"])  # noqa: E731
        dT = float(np.median(h["T_centre"].iloc[-30:] - h["T_pyro"].iloc[-30:]))
        # Arrhenius analysis of holds when every hold is offset by the same Joule-heating error
        holds_true, holds_meas = [], []
        for Th_meas in HOLDS:
            Th_true = Th_meas + dT
            Tf, dur = heating_program(100.0, Th_true, 1200.0)
            t = np.linspace(0, dur, 3000)
            D = m.simulate(t, Tf, SIGMA)
            k = t >= dur - 1200.0
            holds_true.append((Th_true, t[k], D[k]))
            holds_meas.append((Th_meas, t[k], D[k]))
        Dr = 0.5 * (max(x[2][0] for x in holds_true) + min(x[2][-1] for x in holds_true))
        Q_t, *_ = activation_energy_isothermal(holds_true, m.n, Dr, m.E, m.D_green, SIGMA)
        Q_m, *_ = activation_energy_isothermal(holds_meas, m.n, Dr, m.E, m.D_green, SIGMA)
        report["temperature_measurement_bias"] = {
            "centre_minus_pyrometer_K": dT, "Q_using_true_T_kJ": Q_t, "Q_using_pyrometer_T_kJ": Q_m,
            "comment": "A constant offset dT between sample and pyrometer biases Q by roughly Q*(1 - T_meas^2/T_true^2)."}
        del T_true

    (out / "densification_report.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps({k: v for k, v in report.items() if k != "true_model"}, indent=1, default=float))


if __name__ == "__main__":
    main()
