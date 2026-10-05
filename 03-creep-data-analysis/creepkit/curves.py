"""Single creep curve analysis.

Typical constant-load creep curve (strain vs time) has three stages:

  primary    strain rate decreases (work hardening > recovery)
  secondary  (quasi) steady state - the MINIMUM creep rate, the quantity used in
             Norton / Arrhenius analysis
  tertiary   strain rate increases (necking, cavitation, microstructural degradation)
             until rupture at t_r

Numerically differentiating noisy LVDT/extensometer data is the error-prone step, so the
derivative is taken with a Savitzky-Golay filter on a uniformly re-sampled time grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter


@dataclass
class CreepTest:
    """One creep test. Time in hours, strain dimensionless (engineering unless stated)."""

    time_h: np.ndarray
    strain: np.ndarray
    stress_MPa: float
    temperature_C: float
    specimen_id: str = ""
    ruptured: bool = True
    metadata: dict = field(default_factory=dict)

    @property
    def temperature_K(self) -> float:
        return self.temperature_C + 273.15

    @property
    def time_s(self) -> np.ndarray:
        return np.asarray(self.time_h) * 3600.0

    @property
    def true_strain(self) -> np.ndarray:
        return np.log1p(self.strain)

    @property
    def rupture_time_h(self) -> float | None:
        return float(self.time_h[-1]) if self.ruptured else None

    @property
    def strain_at_fracture(self) -> float | None:
        return float(self.strain[-1]) if self.ruptured else None

    @classmethod
    def from_displacement(cls, time_h, displacement_mm, gauge_length_mm, **kw) -> "CreepTest":
        """Convert a crosshead/LVDT displacement record to engineering strain."""
        return cls(np.asarray(time_h), np.asarray(displacement_mm) / gauge_length_mm, **kw)


def resample_uniform(t, y, n: int | None = None):
    """Linear interpolation of (t, y) onto a uniform grid (needed by Savitzky-Golay)."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    order = np.argsort(t)
    t, y = t[order], y[order]
    n = n or len(t)
    tu = np.linspace(t[0], t[-1], n)
    return tu, np.interp(tu, t, y)


def strain_rate(time_s, strain, window_frac: float = 0.05, polyorder: int = 2, n: int | None = None):
    """Smoothed strain rate (1/s) via Savitzky-Golay differentiation.

    window_frac  filter window as a fraction of the record (5% is a good start; increase
                 for noisy data, decrease if the minimum is sharp)
    Returns (t_uniform, strain_smoothed, rate).
    """
    tu, yu = resample_uniform(time_s, strain, n)
    win = max(polyorder + 3, int(window_frac * len(tu)))
    if win % 2 == 0:  # Savitzky-Golay needs an odd window
        win += 1
    if win >= len(tu):
        win = len(tu) - 1 if (len(tu) - 1) % 2 == 1 else len(tu) - 2
    dt = tu[1] - tu[0]
    ys = savgol_filter(yu, win, polyorder)
    rate = savgol_filter(yu, win, polyorder, deriv=1, delta=dt)
    return tu, ys, rate


@dataclass
class MinimumRate:
    rate_per_s: float
    time_h: float
    strain: float


def minimum_creep_rate(test: CreepTest, window_frac: float = 0.05, trim: float = 0.05) -> MinimumRate:
    """Minimum creep rate, ignoring the first/last `trim` fraction (filter edge effects)."""
    t, ys, rate = strain_rate(test.time_s, test.strain, window_frac)
    i0, i1 = int(trim * len(t)), int((1 - trim) * len(t))
    i = i0 + int(np.argmin(rate[i0:i1]))
    return MinimumRate(float(rate[i]), float(t[i] / 3600.0), float(ys[i]))


def creep_stages(test: CreepTest, tolerance: float = 0.25, window_frac: float = 0.05):
    """Split the curve into primary / secondary / tertiary.

    The secondary stage is defined (operationally) as the region where the strain rate is
    within (1 + tolerance) of the minimum rate. Returns the stage boundary times in hours.
    """
    t, _, rate = strain_rate(test.time_s, test.strain, window_frac)
    m = minimum_creep_rate(test, window_frac)
    imin = int(np.argmin(np.abs(t / 3600.0 - m.time_h)))
    limit = (1 + tolerance) * m.rate_per_s
    i_start = imin
    while i_start > 0 and rate[i_start - 1] <= limit:
        i_start -= 1
    i_end = imin
    while i_end < len(t) - 1 and rate[i_end + 1] <= limit:
        i_end += 1
    return {
        "primary_end_h": float(t[i_start] / 3600.0),
        "tertiary_start_h": float(t[i_end] / 3600.0),
        "minimum_rate_time_h": m.time_h,
    }


# --------------------------------------------------------------------------------------
# Theta projection (Evans & Wilshire, "Creep of Metals and Alloys", 1985)
# --------------------------------------------------------------------------------------

def theta_strain(t, eps0, th1, th2, th3, th4):
    """eps(t) = eps0 + th1 (1 - exp(-th2 t)) + th3 (exp(th4 t) - 1)

    th1, th2 describe the decaying (primary) term, th3, th4 the accelerating (tertiary)
    term. There is no explicit steady state: the minimum rate arises where both balance.
    """
    return eps0 + th1 * (1.0 - np.exp(-th2 * t)) + th3 * (np.exp(th4 * t) - 1.0)


def fit_theta_projection(test: CreepTest):
    """Fit the four theta parameters (+ instantaneous strain). Time in hours."""
    t = np.asarray(test.time_h, float)
    e = np.asarray(test.strain, float)
    tr = t[-1]
    p0 = [max(e[0], 0.0), 0.3 * (e[-1] - e[0]) + 1e-6, 10.0 / tr, 0.05 * (e[-1] - e[0]) + 1e-8, 3.0 / tr]
    bounds = ([0, 0, 1e-6 / tr, 0, 1e-6 / tr], [np.inf, np.inf, 1e4 / tr, np.inf, 50 / tr])
    popt, pcov = curve_fit(theta_strain, t, e, p0=p0, bounds=bounds, maxfev=50000)
    names = ["eps0", "theta1", "theta2_per_h", "theta3", "theta4_per_h"]
    return dict(zip(names, map(float, popt))), np.sqrt(np.diag(pcov))
