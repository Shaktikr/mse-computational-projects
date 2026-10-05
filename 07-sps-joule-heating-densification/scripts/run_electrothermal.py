"""Joule heating in SPS: conductive (Ni) vs insulating (Al2O3) powder compacts.

    python scripts/run_electrothermal.py
    python scripts/run_electrothermal.py --rate 200 --T-hold 1373 --hold 300

Outputs (results/):
    history_<case>.csv        time series: set-point, pyrometer, sample centre, current, voltage, power
    fig_temperature_fields.png  r-z temperature maps at the end of the ramp + current paths
    fig_histories.png         temperature lag / overshoot and electrical power for both cases
    electrothermal_summary.json
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

from spsmodel.electrothermal import SAMPLE, VOID, Schedule, SPSModel  # noqa: E402
from spsmodel.materials import AluminaCompact, NickelCompact  # noqa: E402

plt.rcParams.update({"figure.dpi": 150})


def run_case(sample, schedule, dt):
    model = SPSModel(sample)
    # stop at the end of the ramp to grab the field, then continue through the hold
    hist = model.run(schedule, dt=dt)
    return model, hist


def field_snapshot(model, schedule, dt):
    """Temperature and potential fields at the end of the ramp (re-run up to that time)."""
    ramp = Schedule(schedule.T0, schedule.rate_K_per_min, schedule.T_hold, t_hold_s=0.0)
    m = SPSModel(model.sample)
    m.run(ramp, dt=dt)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rate", type=float, default=100.0, help="heating rate (K/min)")
    ap.add_argument("--T-hold", type=float, default=1273.0, help="hold temperature at the pyrometer (K)")
    ap.add_argument("--hold", type=float, default=300.0, help="hold time (s)")
    ap.add_argument("--dt", type=float, default=1.0)
    args = ap.parse_args()
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    sched = Schedule(rate_K_per_min=args.rate, T_hold=args.T_hold, t_hold_s=args.hold)

    cases = {"Ni_compact": NickelCompact(D=0.65), "Al2O3_compact": AluminaCompact(D=0.60)}
    results, snaps, summary = {}, {}, {}
    for name, sample in cases.items():
        model, hist = run_case(sample, sched, args.dt)
        pd.DataFrame(hist).to_csv(out / f"history_{name}.csv", index=False, float_format="%.5g")
        results[name] = hist
        snaps[name] = field_snapshot(model, sched, args.dt)
        i_end = np.searchsorted(hist["t"], (args.T_hold - sched.T0) / (args.rate / 60.0))
        summary[name] = {
            "sample": sample.name, "relative_density": sample.D,
            "dT_centre_minus_pyrometer_end_of_ramp_K": float(hist["T_centre"][i_end] - hist["T_pyro"][i_end]),
            "dT_centre_minus_pyrometer_end_of_hold_K": float(hist["T_centre"][-1] - hist["T_pyro"][-1]),
            "radial_gradient_in_sample_end_of_hold_K": float(hist["T_sample_max"][-1] - hist["T_sample_min"][-1]),
            "current_end_of_hold_A": float(hist["I"][-1]), "voltage_end_of_hold_V": float(hist["V"][-1]),
            "power_end_of_hold_kW": float(hist["P"][-1] / 1000.0),
            "fraction_of_current_through_sample": float(np.median(hist["frac_current_sample"][1:])),
        }
        print(name, json.dumps(summary[name], indent=1))

    # ---------------------------------------------------------------- field figure
    fig, axs = plt.subplots(1, 2, figsize=(11, 6.0))
    for ax, (name, m) in zip(axs, snaps.items()):
        T = np.where(m.mat == VOID, np.nan, m.T) - 273.15
        rr = np.concatenate([-m.r[::-1], m.r]) * 1000
        TT = np.concatenate([T[::-1], T], axis=0)
        im = ax.pcolormesh(rr, m.z * 1000, TT.T, cmap="inferno", shading="auto")
        phi = np.where(m.mat == VOID, np.nan, m.phi)
        PP = np.concatenate([phi[::-1], phi], axis=0)
        ax.contour(rr, m.z * 1000, PP.T, levels=12, colors="cyan", linewidths=0.5)
        mask = np.concatenate([(m.mat == SAMPLE)[::-1], m.mat == SAMPLE], axis=0).astype(float)
        ax.contour(rr, m.z * 1000, mask.T, levels=[0.5], colors="white", linewidths=1.2)
        ax.plot([m.geom.r_die_out * 1000], [0], "c*", ms=12, mec="k")
        ax.set_aspect("equal")
        ax.set(xlabel="r (mm)", ylabel="z (mm)", ylim=(-35, 35),
               title=f"{m.sample.name}\nend of ramp: centre − pyrometer = "
                     f"{m.T[m.centre] - m.T[m.pyro]:+.0f} K")
        plt.colorbar(im, ax=ax, shrink=0.75, label="T (°C)")
    fig.suptitle("Temperature field (colour) and equipotentials (cyan); ★ = pyrometer spot", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_temperature_fields.png")
    plt.close(fig)

    # ---------------------------------------------------------------- history figure
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for i, (name, h) in enumerate(results.items()):
        axs[0].plot(h["t"] / 60, h["T_pyro"] - 273.15, color=f"C{i}", ls="--", lw=1)
        axs[0].plot(h["t"] / 60, h["T_centre"] - 273.15, color=f"C{i}", label=f"{name}: centre (solid) / pyrometer (dashed)")
        axs[1].plot(h["t"] / 60, h["T_centre"] - h["T_pyro"], color=f"C{i}", label=name)
        axs[2].plot(h["t"] / 60, h["P"] / 1000, color=f"C{i}", label=name)
    axs[0].plot(results["Ni_compact"]["t"] / 60, results["Ni_compact"]["T_set"] - 273.15, "k:", lw=1, label="set-point")
    axs[0].set(xlabel="time (min)", ylabel="T (°C)", title="Temperatures")
    axs[1].axhline(0, color="k", lw=0.6)
    axs[1].set(xlabel="time (min)", ylabel="T$_{centre}$ − T$_{pyrometer}$ (K)", title="Error of the measured temperature")
    axs[2].set(xlabel="time (min)", ylabel="electrical power (kW)", title="Power supplied by the SPS")
    for a in axs:
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_histories.png")
    plt.close(fig)
    (out / "electrothermal_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
