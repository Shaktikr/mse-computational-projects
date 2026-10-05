"""Quantities extracted from the phase-field simulations."""

from __future__ import annotations

import numpy as np


def neck_radius(rho: np.ndarray, x_neck: int, threshold: float = 0.5) -> float:
    """Half-width of the solid (rho > threshold) along the column through the neck,
    with sub-pixel linear interpolation at both edges."""
    col = rho[:, x_neck]
    above = np.where(col > threshold)[0]
    if len(above) == 0:
        return 0.0
    lo, hi = above[0], above[-1]

    def edge(i_in, i_out):
        if i_out < 0 or i_out >= len(col):
            return float(i_in)
        a, b = col[i_out], col[i_in]
        return i_out + (threshold - a) / (b - a) * (i_in - i_out)

    return 0.5 * (edge(hi, hi + 1) - edge(lo, lo - 1))


def neck_growth_exponent(t, x_over_R, fit_range=(0.15, 0.45)):
    """Fit (x/R)^m = B t  ->  m = 1 / slope of log(x/R) vs log(t), within a x/R window.

    Classical values: m ~ 2 viscous flow, 3 evaporation-condensation, 5 volume diffusion,
    6 grain-boundary diffusion, 7 surface diffusion (Kuczynski 1949; Coble 1958).
    """
    t = np.asarray(t, float)
    y = np.asarray(x_over_R, float)
    m = (y >= fit_range[0]) & (y <= fit_range[1]) & (t > 0)
    slope, _ = np.polyfit(np.log(t[m]), np.log(y[m]), 1)
    return 1.0 / slope
