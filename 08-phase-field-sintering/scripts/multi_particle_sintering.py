"""Multi-particle sintering of a random 2D powder packing.

    python scripts/multi_particle_sintering.py            # ~10-20 min
    python scripts/multi_particle_sintering.py --quick

Tracks pore-surface length, number of grains (grain growth / coalescence) and mean grain
size, and writes a snapshot montage plus an animated GIF of the microstructure.
Outputs: results/multi_particle.csv, fig_multi_particle_snapshots.png,
         fig_multi_particle_stats.png, multi_particle.gif
"""

import argparse
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
from pfsinter.geometry import random_packing  # noqa: E402


def render(ax, sim, title):
    gm = sim.grain_map()
    rng = np.random.default_rng(3)
    colors = rng.uniform(0.25, 0.95, size=(sim.etas.shape[0], 3))
    img = np.ones(gm.shape + (3,))
    for g in range(sim.etas.shape[0]):
        img[gm == g] = colors[g]
    ax.imshow(img, origin="lower")
    ax.contour(sim.rho, levels=[0.5], colors="k", linewidths=0.8)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_stats(df, out):
    """Surface-energy reduction and grain count (from results/multi_particle.csv)."""
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    ax[0].semilogx(df["t"], df["surface_length"] / df["surface_length"].iloc[0])
    ax[0].set(xlabel="t (reduced units)", ylabel="surface length / initial", title="Free-surface (energy) reduction")
    ax[1].semilogx(df["t"], df["n_grains"], "o-", ms=3)
    ax[1].set(xlabel="t (reduced units)", ylabel="number of grains", title="Grains surviving",
              ylim=(0, df["n_grains"].max() + 2))
    for a in ax:
        a.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "fig_multi_particle_stats.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--t-end", type=float, default=2000.0)
    args = ap.parse_args()
    N, n, r = (128, 6, 14.0) if args.quick else (256, args.n, 24.0)
    t_end = 40.0 if args.quick else args.t_end
    rho, etas, info = random_packing(N, N, n=n, r_mean=r, r_std=0.18 * r, seed=4)
    sim = SinteringPF(rho, etas, Params(dt=0.05, M_surf=4.0, M_gb=0.4, M_vol=0.04, M_vap=0.002))
    out = ROOT / "results"
    out.mkdir(exist_ok=True)

    t_out = np.unique(np.round(np.geomspace(1.0, t_end, 40), 1))
    snap_times = [t_out[0], t_out[len(t_out) // 3], t_out[2 * len(t_out) // 3], t_out[-1]]
    rows, frames, snaps = [], [], []
    for tt in t_out:
        sim.p.dt = 0.05 if sim.t < 20 else 0.2
        k = int(round((tt - sim.t) / sim.p.dt))
        if k > 0:
            sim.step(k)
        gm = sim.grain_map()
        areas = np.array([np.sum(gm == g) for g in range(sim.etas.shape[0])])
        alive = areas > 20
        rows.append({"t": sim.t, "solid_fraction": sim.density_fraction(), "surface_length": sim.surface_length(),
                     "n_grains": int(alive.sum()), "mean_grain_area": float(areas[alive].mean())})
        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        render(ax, sim, f"t = {sim.t:.0f}")
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
        if any(np.isclose(tt, s) for s in snap_times):
            snaps.append((sim.t, sim.rho.copy(), sim.etas.copy()))
    df = pd.DataFrame(rows)
    df.to_csv(out / "multi_particle.csv", index=False, float_format="%.5g")

    fig, axs = plt.subplots(1, len(snaps), figsize=(3.4 * len(snaps), 3.5))
    for a, (t, r_, e_) in zip(axs, snaps):
        sim.rho, sim.etas = r_, e_
        render(a, sim, f"t = {t:.0f}")
    fig.suptitle("Phase-field sintering of a random powder packing (colours = grains, line = pore surface)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_multi_particle_snapshots.png", dpi=150)
    plt.close(fig)

    plot_stats(df, out)

    try:
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(out / "multi_particle.gif", save_all=True, append_images=imgs[1:], duration=150, loop=0)
    except Exception as exc:  # noqa: BLE001
        print("GIF not written:", exc)
    print(df.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
