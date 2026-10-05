"""Step 6: collect all DFT results, make figures, and feed them into creep equations.

Creep relations used (Frost & Ashby, Deformation-Mechanism Maps, 1982; Mukherjee-Bird-Dorn):
  lattice self-diffusion     D = D0 exp(-Q/kT),            Q = E_f + E_m   (vacancy mechanism)
  vacancy concentration      c_v = exp(S_f/k) exp(-E_f/kT)
  power-law (dislocation)    e_dot = A (D G b / kT) (sigma/G)^n
  Nabarro-Herring            e_dot = 14 sigma Omega D / (kT d^2)
  Mohamed-Langdon SFE term   e_dot  proportional to (gamma_SF / G b)^3
  Shockley partial separation (isotropic elasticity)
     d_edge  = G b_p^2/(8 pi gamma) (2+nu)/(1-nu),  d_screw = G b_p^2/(8 pi gamma) (2-3nu)/(1-nu)
Quantities that DFT at 0 K does not give (diffusion prefactor D0, formation entropy S_f,
Dorn constant A, stress exponent n) are taken from experiment and labelled as such."""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

kB = 8.617333e-5            # eV/K
EV = 1.602176634e-19        # J
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0",
                     "grid.linewidth": 0.6, "lines.linewidth": 2, "figure.dpi": 150,
                     "savefig.bbox": "tight"})

conv = json.load(open("results_convergence.json"))
bulk = json.load(open("results_bulk.json"))
vac = json.load(open("results_vacancy.json"))
mig = json.load(open("results_migration_e30_k4.json"))
kchk = json.load(open("results_kcheck.json"))
KBEST = sorted(kchk, key=lambda s: int(s[1:]))[-1]          # densest k-mesh in the check
gsfe = json.load(open("results_gsfe.json"))
B = bulk["k40"]                       # densest k-mesh = production values

# ---------------- Figure 1: convergence ----------------
fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
e = np.array(conv["ecut"]); k = np.array(conv["kpts"])
ax[0].plot(e[:, 0], (e[:, 1] - e[-1, 1]) * 1000, "o-", color=C1, ms=5)
ax[0].axhspan(-1, 1, color=C3, alpha=0.12, lw=0)
ax[0].set_xlabel("ecutwfc (Ry)"); ax[0].set_ylabel("E − E(60 Ry)  (meV/atom)")
ax[0].set_title("Plane-wave cutoff", loc="left", fontsize=10)
ax[0].annotate("chosen: 40 Ry", (40, (e[e[:, 0] == 40][0, 1] - e[-1, 1]) * 1000), xytext=(42, 6),
               color=INK, arrowprops=dict(arrowstyle="-", color=MUTED))
ax[1].plot(k[:, 0], (k[:, 1] - k[-1, 1]) * 1000, "o-", color=C2, ms=5)
ax[1].axhspan(-5, 5, color=C3, alpha=0.12, lw=0)
ax[1].set_xlabel("k-mesh (n × n × n)"); ax[1].set_ylabel("E − E(24³)  (meV/atom)")
ax[1].set_title("k-point sampling", loc="left", fontsize=10)
fig.savefig("fig1_convergence.png"); plt.close(fig)

# ---------------- Figure 2: E-V curve ----------------
def bm3(V, E0, V0, B0, Bp):
    eta = (V0 / V) ** (2 / 3)
    return E0 + 9 * V0 * B0 / 16 * ((eta - 1) ** 3 * Bp + (eta - 1) ** 2 * (6 - 4 * eta))
V = np.array(B["ev_curve"]["V"]); E = np.array(B["ev_curve"]["E"])
Vf = np.linspace(V.min(), V.max(), 200)
fig, ax = plt.subplots(figsize=(4.6, 3.3))
ax.plot(Vf, (bm3(Vf, B["E0_eV"], B["V0_A3"], B["B0_GPa"] / 160.21766, B["Bprime"]) - B["E0_eV"]) * 1000,
        color=C1, lw=1.5, label="Birch–Murnaghan fit")
ax.plot(V, (E - B["E0_eV"]) * 1000, "o", color=C1, ms=6, mfc="white", mew=1.5, label="DFT (PBE)")
ax.axvline(4.0496 ** 3 / 4, color=MUTED, ls="--", lw=1)
ax.text(4.0496 ** 3 / 4 + 0.04, 17, "expt. a = 4.0496 Å\n(298 K)", color=MUTED, fontsize=8)
ax.set_xlabel("Volume per atom (Å³)"); ax.set_ylabel("E − E₀ (meV/atom)")
ax.set_title(f"a₀ = {B['a0_A']:.3f} Å,  B₀ = {B['B0_GPa']:.1f} GPa", loc="left", fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.18, 1.0))
fig.savefig("fig2_eos.png"); plt.close(fig)

# ---------------- Figure 3: elastic constants vs k ----------------
ks = [24, 32, 40]
fig, ax = plt.subplots(figsize=(4.6, 3.3))
for key, col, exp in [("C11", C1, 114.3), ("C12", C2, 61.9), ("C44", C3, 31.6)]:
    vals = [bulk[f"k{kk}"][key] for kk in ks]
    ax.plot(ks, vals, "o-", color=col, ms=5)
    ax.axhline(exp, color=col, ls=":", lw=1)
    ax.text(40.6, vals[-1], f"{key.replace('C', 'C')} {vals[-1]:.0f}", color=INK, va="center", fontsize=9)
ax.set_xlim(22, 45); ax.set_xticks(ks)
ax.set_xlabel("k-mesh (n × n × n)"); ax.set_ylabel("GPa")
ax.set_title("Elastic constants vs k-points (dotted: expt. 0 K)", loc="left", fontsize=10)
fig.savefig("fig3_elastic_k.png"); plt.close(fig)

# ---------------- Figure 4: migration profile ----------------
import sys as _s; _s.path.insert(0, "."); from qe import parse
e0 = parse("runs/05_mig/mig_e30_k4_x0.00.out")["energy_eV"]
p25 = "runs/05_mig/partial_unconverged_x0.25.out"
e25 = parse(p25)["energies_eV"][-1] - e0 if os.path.exists(p25) else None
Em4 = mig["Em_eV"]; EmB = kchk[KBEST]["Em_eV"]
fig, ax = plt.subplots(figsize=(4.6, 3.3))
xf = np.linspace(0, 1, 200)
ax.plot(xf, EmB * np.sin(np.pi * xf) ** 2, color=C1, lw=1.5)
ax.plot([0, 0.5, 1], [0, EmB, 0], "o", color=C1, ms=7, mfc="white", mew=1.5,
        label=f"relaxed end points & saddle ({KBEST[1:]}³ k-mesh)")
if e25 is not None:
  ax.plot([0.25, 0.75], [e25 * EmB / Em4] * 2, "v", color=MUTED, ms=6, mfc="white",
        label="x = 0.25 (partly relaxed, upper bound)")
ax.annotate(f"E_m = {EmB:.2f} eV", (0.5, EmB), xytext=(0.6, EmB * 0.95), color=INK)
ax.set_ylim(-0.03, EmB * 1.25)
ax.set_xlabel("Jump coordinate (0 = initial site, 1 = vacant site)")
ax.set_ylabel("E − E_initial (eV)")
ax.set_title("Vacancy jump barrier in Al", loc="left", fontsize=10)
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
fig.savefig("fig4_migration.png"); plt.close(fig)

# ---------------- Figure 5: GSFE ----------------
annni = json.load(open("results_isf_annni.json"))
# diagnostic runs (optional): the first-pass results with the default (too small) nbnd
gk = json.load(open("results_gsfe_kcheck.json")) if os.path.exists("results_gsfe_kcheck.json") else None
tilt = json.load(open("results_tilt_test.json")) if os.path.exists("results_tilt_test.json") else None
gf = json.load(open("results_gsfe_final_k16.json"))        # corrected: nbnd = 40, 16x16x3
fr = sorted(float(k.split("_f")[1]) for k in gf if k.startswith("rigid_f"))
gr_rigid = np.array([gf[f"rigid_f{x:.2f}"] for x in fr])
corr_us = gf["relaxed_f0.60"] - gf["rigid_f0.60"]
corr_isf = gf["relaxed_f1.00"] - gf["rigid_f1.00"]
KA = sorted(annni, key=lambda s: int(s[1:]))[-1]
g_isf = annni[KA]["gamma_isf_mJm2"] + corr_isf           # converged ANNNI + relaxation correction
g_isf_direct = gf["relaxed_f1.00"]                        # direct tilted cell, 16x16x3
g_us = gf["relaxed_f0.60"]
fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.4), gridspec_kw=dict(width_ratios=[1.1, 1]))
from scipy.interpolate import CubicSpline
ff = np.linspace(0, 1, 200)
ax[0].plot(ff, CubicSpline(fr, gr_rigid)(ff), color=C1, lw=1.5)
ax[0].plot(fr, gr_rigid, "o", color=C1, ms=5, mfc="white", mew=1.5, label="rigid shift")
ax[0].plot([0.6, 1.0], [gf["relaxed_f0.60"], gf["relaxed_f1.00"]], "s", color=C2, ms=7,
           label="relaxed ⊥ to plane")
ax[0].annotate(f"γ_us = {g_us:.0f}", (0.6, g_us), xytext=(0.12, g_us + 5), color=INK,
               arrowprops=dict(arrowstyle="-", color=MUTED))
ax[0].annotate(f"γ_isf = {g_isf_direct:.0f}\n(ANNNI, dense k: {g_isf:.0f})", (1.0, g_isf_direct), xytext=(0.42, 80), fontsize=9, color=INK,
               arrowprops=dict(arrowstyle="-", color=MUTED))
ax[0].set_ylim(0, 230)
ax[0].set_xlabel("Shear displacement  u / b_p  along ⟨112⟩"); ax[0].set_ylabel("γ (mJ/m²)")
ax[0].set_title("GSFE of Al on (111)  (16×16×3 k, nbnd = 40)", loc="left", fontsize=10)
ax[0].legend(frameon=False, fontsize=7.5, loc="lower center")
ks = sorted(int(k[1:]) for k in annni)
ax[1].plot(ks, [annni[f"k{k}"]["gamma_isf_mJm2"] for k in ks], "o-", color=C1, ms=5, label="ANNNI, 2–4-atom cells")
ax[1].plot([16], [gf["rigid_f1.00"]], "D", color=C3, ms=8, label="12-layer cell, nbnd = 40")
bad_pts = [(16, 110.3)] + ([(24, gk["k24x1"]["gamma_rigid_f1.0"]), (24, gk["k24x3"]["gamma_rigid_f1.0"])] if gk else []) + ([(16, tilt["1.0"])] if tilt else [])
ax[1].plot([p[0] for p in bad_pts], [p[1] for p in bad_pts], "x", color="#d03b3b", ms=8, mew=2,
           label="12-layer cell, default nbnd = 22 (wrong)")
ax[1].set_xlabel("in-plane k-mesh (n × n)"); ax[1].set_ylabel("γ_isf, rigid (mJ/m²)")
ax[1].set_title("Cross-checking γ_isf", loc="left", fontsize=10)
ax[1].set_ylim(50, 200); ax[1].set_xlim(12, 44)
ax[1].legend(frameon=False, fontsize=7.5, loc="upper right")
fig.savefig("fig5_gsfe.png"); plt.close(fig)

# ---------------- creep quantities ----------------
a0 = B["a0_A"]; b = a0 / np.sqrt(2) * 1e-10; bp = a0 / np.sqrt(6) * 1e-10
G = B["G_Hill"] * 1e9; nu = B["poisson"]; Omega = B["V0_A3"] * 1e-30
Ef = kchk[KBEST]["Ef_eV"]                   # densest k-mesh (8^3); geometry check: 6^3 reproduces the fully relaxed 0.647 eV
Em = kchk[KBEST]["Em_eV"]                   # saddle/initial geometries from 4x4x4, energies at densest mesh
Q = Ef + Em
gamma_isf = g_isf * 1e-3; gamma_us = g_us * 1e-3

# experimental / literature inputs (labelled in the report)
D0_LM, Q_LM = 1.71e-4, 142e3 / 96485.0   # Lundy & Murdock (1962), high-T tracer data, m^2/s, eV
D0_BR, Q_BR = 1.9e-5, 1.28                # Burke & Ramachandran (1972), lower-T data, m^2/s, eV
A_dorn, n_pl = 3.4e6, 4.4                 # Frost & Ashby, Al
Sf = 0.7                                  # k_B, typical literature value for Al (not computed here)
Tm = 933.47

def D_dft(T):          # DFT activation energy, with the lower-T experimental prefactor
    return D0_BR * np.exp(-Q / (kB * T))
def D_LM(T):
    return D0_LM * np.exp(-Q_LM / (kB * T))
def D_BR(T):
    return D0_BR * np.exp(-Q_BR / (kB * T))

T = np.linspace(400, 930, 200)
summary = dict(a0_A=a0, b_A=b * 1e10, bp_A=bp * 1e10, G_Hill_GPa=G / 1e9, poisson=nu,
               Ef_eV=Ef, Em_eV=Em, Q_eV=Q, Q_kJmol=Q * 96.485, Q_LM_eV=Q_LM, Q_BR_eV=Q_BR,
               gamma_isf_mJm2=gamma_isf * 1e3, gamma_us_mJm2=gamma_us * 1e3,
               cv_Tm_Sf0=np.exp(-Ef / (kB * Tm)), cv_Tm_Sf07=np.exp(Sf) * np.exp(-Ef / (kB * Tm)),
               gamma_over_Gb=gamma_isf / (G * b), gamma_isf_direct_mJm2=g_isf_direct,
               relax_corr_us=corr_us, relax_corr_isf=corr_isf)
pre = G * bp ** 2 / (8 * np.pi * gamma_isf)
summary["d_edge_A"] = pre * (2 + nu) / (1 - nu) * 1e10
summary["d_screw_A"] = pre * (2 - 3 * nu) / (1 - nu) * 1e10
summary["d_edge_over_b"] = summary["d_edge_A"] / (b * 1e10)
for TT in (500, 600, 800):
    summary[f"D_dft_{TT}K"] = D_dft(TT); summary[f"D_LM_{TT}K"] = D_LM(TT); summary[f"D_BR_{TT}K"] = D_BR(TT)
# creep-rate examples: same law, DFT inputs vs fully experimental inputs
Tc, sig_pl = 600.0, 10e6
G_exp, b_exp = 26.0e9, 2.86e-10           # room-temperature polycrystal G of Al (approx.)
summary["edot_PL_600K_10MPa_DFT"] = A_dorn * D_dft(Tc) * G * b / (kB * EV * Tc) * (sig_pl / G) ** n_pl
summary["edot_PL_600K_10MPa_expinputs"] = A_dorn * D_LM(Tc) * G_exp * b_exp / (kB * EV * Tc) * (sig_pl / G_exp) ** n_pl
summary["edot_NH_800K_1MPa_100um_DFT"] = 14 * 1e6 * Omega * D_dft(800) / (kB * EV * 800 * (100e-6) ** 2)
summary["edot_NH_800K_1MPa_100um_expinputs"] = 14 * 1e6 * 16.6e-30 * D_LM(800) / (kB * EV * 800 * (100e-6) ** 2)
summary["factor_per_0.1eV_at_600K"] = np.exp(0.1 / (kB * 600))
summary["factor_G_exponent_(n-1)"] = (G_exp / G) ** (n_pl - 1)

# Figure 6: diffusion Arrhenius
fig, ax = plt.subplots(figsize=(4.8, 3.4))
ax.semilogy(1000 / T, D_dft(T), color=C1, label=f"DFT: Q = {Q:.2f} eV (D₀ from B&R)")
ax.semilogy(1000 / T, D_BR(T), color=C3, ls=":", label="Expt. Burke & Ramachandran: 1.28 eV")
ax.semilogy(1000 / T, D_LM(T), color=C2, ls="--", label="Expt. Lundy & Murdock: 1.47 eV")
ax.set_xlabel("1000 / T  (1/K)"); ax.set_ylabel("D (m²/s)")
ax.set_title("Al self-diffusion", loc="left", fontsize=10)
ax.legend(frameon=False, fontsize=7.5, loc="lower left")
sec = ax.secondary_xaxis("top", functions=(lambda x: 1000 / np.maximum(x, 1e-6), lambda t: 1000 / np.maximum(t, 1e-6)))
sec.set_xticks([500, 600, 700, 800, 900]); sec.set_xlabel("T (K)", fontsize=8)
fig.savefig("fig6_diffusion.png"); plt.close(fig)

for kk, vv in summary.items():
    print(f"{kk:34s} {vv:.4g}")
json.dump({k: float(v) for k, v in summary.items()}, open("results_creep.json", "w"), indent=1)
