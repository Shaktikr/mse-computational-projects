"""Compose docs/overview.png from the key figure of every project (run after all projects)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PANELS = [
    ("01 DFT: elastic constants of B2 NiAl", "01-dft-fundamentals-qe/results/NiAl/elastic.png"),
    ("02 DFT: vacancy diffusion vs experiment", "02-dft-vacancy-diffusion-creep/results/Al_2x2x2/fig_diffusion.png"),
    ("03 Creep analysis: threshold stress", "03-creep-data-analysis/results/synthetic_composite/fig_threshold.png"),
    ("04 Deformation-mechanism map (Ni)", "04-deformation-mechanism-maps/results/nickel_stress_temperature_d100um.png"),
    ("05 ML creep life: SHAP interpretation", "05-ml-creep-rupture-life/results/fig_shap.png"),
    ("06 ML HEA: Al$_x$CoCrFeNi screening", "06-ml-hea-phase-strength/results/fig_screening_AlxCoCrFeNi.png"),
    ("07 SPS: temperature field", "07-sps-joule-heating-densification/results/fig_temperature_fields.png"),
    ("08 Phase-field sintering", "08-phase-field-sintering/results/fig_multi_particle_snapshots.png"),
    ("09 MD creep of nanocrystalline Ni", "09-md-nanocrystalline-creep/results/Ni/fig_md_creep.png"),
    ("10 DFT: generalised stacking-fault energy of Al", "10-dft-al-creep-parameters/fig5_gsfe.png"),
    ("11 Ideal shear strength: Al vs Cu (Ogata 2002)", "11-ideal-shear-strength-ogata2002/figures/figA_shear_curves.png"),
]


def main():
    fig, axs = plt.subplots(4, 3, figsize=(16, 16))
    for ax, (title, rel) in zip(axs.flat, PANELS):
        ax.axis("off")
        p = ROOT / rel
        if p.exists():
            ax.imshow(mpimg.imread(p))
        else:
            ax.text(0.5, 0.5, "(run the project to create this figure)", ha="center", va="center")
        ax.set_title(title, fontsize=11)
    for ax in list(axs.flat)[len(PANELS):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "overview.png", dpi=90)
    print("wrote docs/overview.png")


if __name__ == "__main__":
    main()
