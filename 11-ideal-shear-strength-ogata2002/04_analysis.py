"""Collect results, compute tau_max / gamma_m / G, make figures, write summary JSON."""
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "lines.linewidth": 2,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
FILES = {("Al", "unrelaxed"): "results_shear_Al_unrelaxed_k24_d0.04.json",
         ("Al", "relaxed"): "results_shear_Al_relaxed_k24_d0.04.json",
         ("Cu", "unrelaxed"): "results_shear_CuPAW_unrelaxed_k24.json",
         ("Cu", "relaxed"): "results_shear_CuPAW_relaxed_k24.json"}
summary = {}
curves = {}
for (el, mode), fn in FILES.items():
    if not os.path.exists(fn):
        continue
    d = json.load(open(fn))
    g = np.array(d["gamma"]); t = np.array(d["tau"]); o = np.argsort(g); g, t = g[o], t[o]
    if g[0] > 0:                      # add the origin
        g = np.r_[0.0, g]; t = np.r_[0.0, t]
    cs = CubicSpline(g, t)
    gf = np.linspace(0, g.max(), 2000); tf = cs(gf)
    i = np.argmax(tf)
    G = t[g > 0][0] / g[g > 0][0]     # initial slope from the smallest strain
    summary[f"{el}_{mode}"] = dict(tau_max_GPa=float(tf[i]), gamma_m=float(gf[i]), G_GPa=float(G),
                                    tau_over_G=float(tf[i] / G), n_points=int(len(g)))
    curves[(el, mode)] = (g, t, gf, tf)

fig, ax = plt.subplots(figsize=(5.6, 3.8))
for el, col in (("Al", C1), ("Cu", C2)):
    for mode, ls, mk in (("relaxed", "-", "o"), ("unrelaxed", "--", "s")):
        if (el, mode) not in curves:
            continue
        g, t, gf, tf = curves[(el, mode)]
        ax.plot(gf, tf, ls, color=col, lw=1.8 if mode == "relaxed" else 1.2)
        ax.plot(g, t, mk, color=col, ms=4.5, mfc="white" if mode == "unrelaxed" else col, mew=1.2,
                label=f"{el}, {mode}")
        s = summary[f"{el}_{mode}"]
        if mode == "relaxed":
            xy_txt = {"Al": (0.265, 3.75), "Cu": (0.165, 0.45)}[el]
            ax.annotate(f"{el} relaxed: τ_max = {s['tau_max_GPa']:.2f} GPa\nat γ_m = {s['gamma_m']:.3f}",
                        (s["gamma_m"], s["tau_max_GPa"]), xytext=xy_txt,
                        fontsize=8.5, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED))
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_xlabel("Engineering shear strain γ on (111)[11-2]")
ax.set_ylabel("Shear stress τ = σ_xz (GPa)")
ax.set_title("Ideal shear stress–strain curves (PBE, this work)", loc="left", fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_xlim(0, 0.37); ax.set_ylim(-0.4, 4.4)
os.makedirs("figures", exist_ok=True); fig.savefig("figures/figA_shear_curves.png"); plt.close(fig)

for k, v in summary.items():
    print(k, v)

# ---------------- stacking faults: Al (companion project) and Cu ----------------
from scipy.interpolate import PchipInterpolator
al = json.load(open("results_gsfe_Al_from_creep_project.json"))
al_ann = json.load(open("results_isf_annni_Al_from_creep_project.json"))
cu = json.load(open("results_sfe_CuPAW.json"))
fa = sorted(float(k.split("_f")[1]) for k in al if k.startswith("rigid_f"))
ga = [al[f"rigid_f{x:.2f}"] for x in fa]
k12 = cu["gsfe_k12_NL6"]; fc = sorted(float(x) for x in k12["rigid"]); gc = [k12["rigid"][f"{x:.2f}"] for x in fc]
sfe = dict(
    Al=dict(gamma_us=al["relaxed_f0.60"], gamma_isf_direct=al["relaxed_f1.00"],
            gamma_isf_annni=al_ann["k40"]["gamma_isf_mJm2"] + (al["relaxed_f1.00"] - al["rigid_f1.00"])),
    Cu=dict(gamma_us=max(k12["relaxed"].values()), gamma_us_at=max(k12["relaxed"], key=k12["relaxed"].get),
            gamma_isf_direct_k20=cu["gsfe_k20_NL6"]["rigid"]["1.00"], gamma_isf_annni_k32=cu["annni"]["k32"]["gamma_isf"],
            gamma_us_rigid_k20_f06=cu["gsfe_k20_NL6"]["rigid"]["0.60"], gamma_us_rigid_k12_f06=k12["rigid"]["0.60"]))
summary["sfe"] = sfe
json.dump(summary, open("results_summary.json", "w"), indent=1)
print(sfe)
fig, ax = plt.subplots(figsize=(5.6, 4.2))
ff = np.linspace(0, 1, 200)
ax.plot(ff, PchipInterpolator(fa, ga)(ff), color=C1, lw=1.5)
ax.plot(fa, ga, "o", color=C1, ms=5, mfc="white", mew=1.4, label="Al rigid shift (16×16×3 k)")
ax.plot([0.6, 1.0], [al["relaxed_f0.60"], al["relaxed_f1.00"]], "s", color=C1, ms=7, label="Al relaxed ⊥ plane")
fcs = [x for x in fc if x >= 0.5]; gcs = [k12["rigid"][f"{x:.2f}"] for x in fcs]
fs = np.linspace(0.5, 1, 100)
ax.plot(fs, PchipInterpolator(fcs, gcs)(fs), color=C2, lw=1.5)
ax.plot(fc, gc, "o", color=C2, ms=5, mfc="white", mew=1.4, label="Cu rigid shift (12×12×2 k)")
rel = {float(k): v for k, v in k12["relaxed"].items()}
ax.plot(list(rel), list(rel.values()), "s", color=C2, ms=7, label="Cu relaxed ⊥ plane")
ax.plot([0.6, 1.0], [sfe["Cu"]["gamma_us_rigid_k20_f06"], sfe["Cu"]["gamma_isf_direct_k20"]], "D", color=C3, ms=6,
        label="Cu rigid, denser 20×20×3 k")
ax.text(0.03, 120, "Cu between u = 0 and 0.5\nnot computed", fontsize=7.5, color=MUTED)
ax.annotate("Al: γ_us = 176\nγ_isf = 134–140", (1.0, al["relaxed_f1.00"]), xytext=(0.80, 205), fontsize=8.5,
            arrowprops=dict(arrowstyle="-", color=MUTED))
ax.annotate("Cu: γ_us = 162\nγ_isf ≈ 43", (1.0, sfe["Cu"]["gamma_isf_direct_k20"]), xytext=(0.86, 98), fontsize=8.5,
            arrowprops=dict(arrowstyle="-", color=MUTED))
ax.set_xlabel("Shear displacement u / b_p along ⟨112⟩"); ax.set_ylabel("γ (mJ/m²)")
ax.set_title("Generalized stacking-fault energy (PBE, this work)", loc="left", fontsize=10)
ax.set_ylim(0, 240); ax.set_xlim(-0.02, 1.1)
ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
fig.savefig("figures/figB_gsfe.png"); plt.close(fig)
