"""Analysis of MD creep curves."""

from __future__ import annotations

import numpy as np


def steady_state_rate(t_ps, strain, start_frac: float = 0.4):
    """Linear fit of strain vs time over the last (1 - start_frac) of the run. Returns 1/s."""
    t = np.asarray(t_ps)
    e = np.asarray(strain)
    m = t >= start_frac * t[-1]
    slope, _ = np.polyfit(t[m], e[m], 1)
    return slope * 1e12


def stress_exponent(stress, rate):
    s = np.asarray(stress, float)
    r = np.asarray(rate, float)
    ok = r > 0
    n, c = np.polyfit(np.log(s[ok]), np.log(r[ok]), 1)
    return float(n), float(c)


def activation_energy(T_K, rate):
    """Q (eV) from ln(rate) vs 1/kT at constant stress."""
    kB = 8.617333262e-5
    T = np.asarray(T_K, float)
    r = np.asarray(rate, float)
    slope, _ = np.polyfit(1.0 / (kB * T), np.log(r), 1)
    return float(-slope)
