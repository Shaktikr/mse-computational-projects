"""Draw deformation-mechanism maps for the materials in materials/*.json.

    python scripts/make_maps.py                         # all materials, default grain sizes
    python scripts/make_maps.py --material nickel --d 1e-6 10e-6 100e-6
    python scripts/make_maps.py --material my_alloy.json --points tests.csv

--points: CSV with columns stress_MPa, T_C[, d_m, label] (e.g. your creep test matrix) - they
are plotted on the maps so you can see which mechanism should control each test.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from defmap import Material, stress_grainsize_map, stress_temperature_map  # noqa: E402
from defmap.maps import legend_handles  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--material", nargs="*", default=["nickel", "aluminium", "copper"])
    ap.add_argument("--d", nargs="*", type=float, default=[1e-6, 100e-6], help="grain sizes (m)")
    ap.add_argument("--T-over-Tm", type=float, default=0.6, help="homologous T for the grain-size map")
    ap.add_argument("--points", default=None)
    args = ap.parse_args()
    points = pd.read_csv(args.points).to_dict("records") if args.points else None
    out = ROOT / "results"
    out.mkdir(exist_ok=True)

    for name in args.material:
        path = Path(name) if name.endswith(".json") else ROOT / "materials" / f"{name}.json"
        m = Material.from_json(path)
        stem = path.stem
        for d in args.d:
            fig, ax = plt.subplots(figsize=(7.8, 6.6))
            stress_temperature_map(m, d, ax=ax, points=points)
            ax.legend(handles=legend_handles(), fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False)
            fig.tight_layout()
            f = out / f"{stem}_stress_temperature_d{d * 1e6:g}um.png"
            fig.savefig(f, dpi=150)
            plt.close(fig)
            print(f"wrote {f.relative_to(ROOT)}")
        fig, ax = plt.subplots(figsize=(7.8, 6.6))
        stress_grainsize_map(m, args.T_over_Tm * m.Tm_K, ax=ax,
                             points=[p for p in points if "d_m" in p] if points else None)
        ax.legend(handles=legend_handles(), fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False)
        fig.tight_layout()
        f = out / f"{stem}_stress_grainsize_T{args.T_over_Tm:g}Tm.png"
        fig.savefig(f, dpi=150)
        plt.close(fig)
        print(f"wrote {f.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
