"""Re-plot results/neck_growth.csv with the LOCAL neck-growth exponent.

    python scripts/plot_neck_growth.py

For (x/R)^m = B t the local exponent is m_local = d ln t / d ln(x/R). Classical two-sphere
models predict a constant m only while x/R << 1 (m ~ 5 volume, 6 grain-boundary, 7 surface
diffusion). Plotting m_local against x/R shows where (and whether) a simulation is in
that regime - and how neck growth slows as the neck approaches its equilibrium shape.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def local_exponent(t, y, window: int = 7):
    lt, ly = np.log(np.asarray(t)), np.log(np.asarray(y))
    m = np.full(len(lt), np.nan)
    h = window // 2
    for i in range(h, len(lt) - h):
        s = slice(i - h, i + h + 1)
        slope = np.polyfit(lt[s], ly[s], 1)[0]
        m[i] = 1.0 / slope if slope > 0 else np.nan
    return m


def main():
    out = ROOT / "results"
    df = pd.read_csv(out / "neck_growth.csv")
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for i, (name, d) in enumerate(df.groupby("case", sort=False)):
        d = d[d["t"] > 0]
        ax[0].loglog(d["t"], d["x_over_R"], "o-", ms=3, color=f"C{i}", label=f"{name} diffusion dominant")
        m = local_exponent(d["t"], d["x_over_R"])
        ax[1].plot(d["x_over_R"], m, "o-", ms=3, color=f"C{i}", label=name)
        ax[2].plot(d["t"], d["surface_length"] / d["surface_length"].iloc[0], color=f"C{i}", label=name)
    for m_c in (5, 6, 7):
        ax[1].axhline(m_c, ls=":", color="0.5", lw=0.8)
    ax[1].text(0.97, 0.97, "dotted: classical m = 5, 6, 7\n(valid for x/R ≪ 1)", transform=ax[1].transAxes, ha="right",
               va="top", fontsize=8)
    ax[0].set(xlabel="time (reduced units)", ylabel="neck radius x/R", title="Neck growth")
    ax[1].set(xlabel="x/R", ylabel="local exponent m", title="Local neck-growth exponent", ylim=(0, 40))
    ax[2].set(xlabel="time", ylabel="surface length / initial", title="Reduction of free surface", xscale="log")
    for a in ax:
        a.grid(alpha=0.3, which="both")
        a.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig_neck_growth.png", dpi=150)
    print("wrote results/fig_neck_growth.png")


if __name__ == "__main__":
    main()
