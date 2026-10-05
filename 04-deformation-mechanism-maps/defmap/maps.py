"""Plot deformation-mechanism maps.

stress_temperature_map: axes T/Tm (x) and log10(sigma_s/mu) (y) at a fixed grain size,
    coloured fields = dominant mechanism, thin lines = contours of constant shear strain
    rate (1e-10 ... 1e2 1/s), right axis = shear stress in MPa at 300 K, top axis = T in C.
stress_grainsize_map: axes grain size (x) and log10(sigma_s/mu) (y) at fixed temperature.

Experimental points (e.g. your creep tests) can be overlaid: pass tensile stress (MPa),
temperature (C) and optionally measured strain rate; they are converted to shear
quantities with the von Mises factors.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from .mechanisms import MECHANISMS, Material, dominant, rates

FIELD_COLORS = ["#d9d9d9", "#a6cee3", "#6baed6", "#fdbf6f", "#b2df8a", "#33a02c"]
IDEAL_COLOR = "#ffffff"


def _field_image(ax, X, Y, dom, extent):
    cmap = ListedColormap([IDEAL_COLOR] + FIELD_COLORS)
    ax.imshow(dom + 1, origin="lower", aspect="auto", extent=extent, cmap=cmap,
              vmin=-0.5, vmax=len(FIELD_COLORS) + 0.5, interpolation="nearest")
    ax.contour(X, Y, dom, levels=np.arange(-0.5, len(MECHANISMS)), colors="k", linewidths=1.2)


def _label_fields(ax, X, Y, dom, xlog=False):
    for i, name in enumerate(MECHANISMS):
        mask = dom == i
        if mask.sum() < 30:
            continue
        x = np.exp(np.log(X[mask]).mean()) if xlog else X[mask].mean()
        y = Y[mask].mean()
        ax.text(x, y, name.replace(" (", "\n("), ha="center", va="center", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, lw=0))


def stress_temperature_map(m: Material, d_m: float, ax=None, n=400, s_range=(-6.5, -1.0),
                           rate_levels=range(-10, 3, 2), points=None):
    t = np.linspace(0.02, 1.0, n)
    ls = np.linspace(*s_range, n)
    TT, LS = np.meshgrid(t, ls)
    r = rates(m, 10**LS, TT * m.Tm_K, d_m)
    dom = dominant(r, m, 10**LS)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 5.6))
    _field_image(ax, TT, LS, dom, [t[0], t[-1], ls[0], ls[-1]])
    cs = ax.contour(TT, LS, np.log10(r["total"]), levels=list(rate_levels), colors="0.25", linewidths=0.6, linestyles="--")
    ax.clabel(cs, fmt=lambda v: f"$10^{{{int(v)}}}$", fontsize=7)
    _label_fields(ax, TT, LS, dom)
    ax.set(xlabel="homologous temperature T/T$_m$", ylabel=r"normalised shear stress log$_{10}$(σ$_s$/μ)",
           title=f"{m.name}, d = {d_m * 1e6:g} µm")
    # secondary axes
    sec_y = ax.secondary_yaxis("right", functions=(lambda y: 10**y * m.mu0_MPa, lambda s: np.log10(np.maximum(s, 1e-30) / m.mu0_MPa)))
    sec_y.set_yscale("log")
    sec_y.set_ylabel("shear stress at 300 K (MPa)")
    sec_x = ax.secondary_xaxis("top", functions=(lambda x: x * m.Tm_K - 273.15, lambda c: (c + 273.15) / m.Tm_K))
    sec_x.set_xlabel("temperature (°C)")
    if points:
        _overlay(ax, m, points, mode="T")
    return ax, dom, r


def stress_grainsize_map(m: Material, T_K: float, ax=None, n=400, d_range=(1e-8, 1e-2), s_range=(-6.5, -1.0),
                         rate_levels=range(-10, 3, 2), points=None):
    d = np.geomspace(*d_range, n)
    ls = np.linspace(*s_range, n)
    DD, LS = np.meshgrid(d, ls)
    r = rates(m, 10**LS, T_K, DD)
    dom = dominant(r, m, 10**LS)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.set_xscale("log")
    cmap = ListedColormap([IDEAL_COLOR] + FIELD_COLORS)
    ax.pcolormesh(DD, LS, dom + 1, cmap=cmap, vmin=-0.5, vmax=len(FIELD_COLORS) + 0.5, shading="auto")
    ax.contour(DD, LS, dom, levels=np.arange(-0.5, len(MECHANISMS)), colors="k", linewidths=1.2)
    cs = ax.contour(DD, LS, np.log10(r["total"]), levels=list(rate_levels), colors="0.25", linewidths=0.6, linestyles="--")
    ax.clabel(cs, fmt=lambda v: f"$10^{{{int(v)}}}$", fontsize=7)
    _label_fields(ax, DD, LS, dom, xlog=True)
    ax.set(xlabel="grain size d (m)", ylabel=r"log$_{10}$(σ$_s$/μ)",
           title=f"{m.name}, T = {T_K:.0f} K (T/T$_m$ = {T_K / m.Tm_K:.2f})")
    if points:
        _overlay(ax, m, points, mode="d", T_K=T_K)
    return ax, dom, r


def _overlay(ax, m: Material, points, mode="T", T_K=None):
    """points: list of dicts {stress_MPa (tensile), T_C, d_m (for mode 'd'), label}."""
    for p in points:
        T = p["T_C"] + 273.15
        s_over_mu = (p["stress_MPa"] / np.sqrt(3)) / m.mu(T)
        x = T / m.Tm_K if mode == "T" else p["d_m"]
        ax.plot(x, np.log10(s_over_mu), marker=p.get("marker", "*"), ms=11, mfc=p.get("color", "red"), mec="k")
        if p.get("label"):
            ax.annotate(p["label"], (x, np.log10(s_over_mu)), xytext=(5, 5), textcoords="offset points", fontsize=7)


def legend_handles():
    from matplotlib.patches import Patch
    return [Patch(fc=c, ec="k", label=n) for c, n in zip(FIELD_COLORS, MECHANISMS)]
