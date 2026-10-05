"""Two-particle sintering: neck growth kinetics for different dominant diffusion paths.

    python scripts/two_particle_neck_growth.py            # ~15-30 min on a laptop
    python scripts/two_particle_neck_growth.py --quick    # short test run

Two equal particles (R = 40 grid units) are sintered with
  case 'surface'  : surface diffusion dominant  (M_surf >> M_gb)
  case 'boundary' : grain-boundary diffusion dominant (M_gb >> M_surf)
and the neck radius x is tracked. Plotting log(x/R) vs log(t) and fitting (x/R)^m ~ t gives
the neck-growth exponent m, which classical two-sphere models relate to the mechanism.

Outputs: results/neck_growth.csv, fig_neck_growth.png, fig_two_particle_snapshots.png
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

from pfsinter import Params, SinteringPF  # noqa: E402
from pfsinter.analysis import neck_growth_exponent, neck_radius  # noqa: E402
from pfsinter.geometry import two_particles  # noqa: E402

CASES = {
    "surface": dict(M_surf=4.0, M_gb=0.04, M_vol=0.01, M_vap=0.001),
    "boundary": dict(M_surf=0.4, M_gb=4.0, M_vol=0.01, M_vap=0.001),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--R", type=float, default=40.0)
    ap.add_argument("--t-end", type=float, default=3000.0)
    args = ap.parse_args()
    if args.quick:
        args.R, args.t_end = 20.0, 60.0
    R = args.R
    nx, ny = int(8 * R), int(4 * R)
    rho0, etas0, info = two_particles(nx, ny, R=R, overlap=2.0)
    x_neck = nx // 2
    out = ROOT / "results"
    out.mkdir(exist_ok=True)

    t_out = np.unique(np.round(np.geomspace(0.5, args.t_end, 60), 2))
    rows, snaps = [], {}
    for name, mob in CASES.items():
        sim = SinteringPF(rho0, etas0, Params(dt=0.05, **mob))
        for t_target in t_out:
            # larger steps once the neck has formed (accuracy check: see README)
            sim.p.dt = 0.05 if sim.t < 50 else 0.2
            nsteps = int(round((t_target - sim.t) / sim.p.dt))
            if nsteps > 0:
                sim.step(nsteps)
            rows.append({"case": name, "t": sim.t, "x_over_R": neck_radius(sim.rho, x_neck) / R,
                         "surface_length": sim.surface_length(), "free_energy": sim.free_energy()})
        snaps[name] = (sim.rho.copy(), sim.grain_map())
        print(name, "done: x/R =", round(rows[-1]["x_over_R"], 3))
    df = pd.DataFrame(rows)
    df.to_csv(out / "neck_growth.csv", index=False, float_format="%.5g")

    fits = {}
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for i, (name, d) in enumerate(df.groupby("case", sort=False)):
        ax[0].loglog(d["t"], d["x_over_R"], "o-", ms=3, color=f"C{i}", label=name)
        try:
            m = neck_growth_exponent(d["t"], d["x_over_R"], fit_range=(0.25, 0.6))
            fits[name] = m
            ax[0].plot([], [], " ", label=f"  fitted m = {m:.1f}")
        except Exception:  # noqa: BLE001
            pass
        ax[1].plot(d["t"], d["surface_length"] / d["surface_length"].iloc[0], color=f"C{i}", label=name)
    ax[0].set(xlabel="time (reduced units)", ylabel="neck radius x/R", title="Neck growth:  (x/R)$^m$ ~ t")
    ax[0].legend(fontsize=8)
    ax[1].set(xlabel="time", ylabel="surface length / initial", title="Reduction of free surface")
    ax[1].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "fig_neck_growth.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(1, len(snaps) + 1, figsize=(4.2 * (len(snaps) + 1), 2.6))
    ax[0].imshow(rho0, cmap="gray", origin="lower")
    ax[0].set_title("initial", fontsize=9)
    for a, (name, (rho, gm)) in zip(ax[1:], snaps.items()):
        a.imshow(np.where(gm >= 0, gm + 1, 0), cmap="viridis", origin="lower", alpha=0.6)
        a.contour(rho, levels=[0.5], colors="k", linewidths=1)
        a.set_title(f"{name}-diffusion dominated, t = {args.t_end:g}", fontsize=9)
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    fig.savefig(out / "fig_two_particle_snapshots.png", dpi=150)
    plt.close(fig)
    (out / "neck_growth_fits.json").write_text(json.dumps(
        {"neck_growth_exponent_m": fits, "fit_range_x_over_R": [0.25, 0.6], "R": R,
         "note": "apparent exponents over the fit range; see fig_neck_growth.png for the local exponent vs x/R"},
        indent=2))
    print("neck-growth exponents:", fits)
    from plot_neck_growth import main as replot  # adds the local-exponent panel
    replot()


if __name__ == "__main__":
    main()
