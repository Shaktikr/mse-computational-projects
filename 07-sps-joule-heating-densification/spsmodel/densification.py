"""Creep-based densification kinetics during SPS / hot pressing.

Bernard-Granger & Guizard (Acta Mater. 55 (2007) 3493) adapted the steady-state creep
equation to porous compacts:

    (1/mu_eff) dD/dt = K * (b D_eff / kT) * (b/G)^p * (sigma_eff/mu_eff)^n          (1)

with the effective (load-bearing) stress and shear modulus of a porous body of relative
density D and green density D0 (Helle, Easterling & Ashby, Acta Metall. 33 (1985) 2163):

    sigma_eff = sigma_mac * (1 - D0) / (D^2 (D - D0))
    E_eff     = E_th * (D - D0) / (1 - D0),     mu_eff = E_eff / (2 (1 + nu_eff))

Workflow used in SPS papers (and implemented below):
  * n  - from the slope of ln[(1/mu_eff) dD/dt] vs ln(sigma_eff/mu_eff) during an
         isothermal hold (sigma_eff/mu_eff falls as D rises)
  * Q  - from an Arrhenius plot of ln[(T/mu_eff) dD/dt (mu_eff/sigma_eff)^n] vs 1/T
         at the same D for several hold temperatures
  * D_eff - eq. (1) solved for D_eff once n, p, K, b and G are fixed:
         D_eff = (1/mu_eff)(dD/dt) kT / [K b (b/G)^p (sigma_eff/mu_eff)^n]
    -> comparing D_eff(T) with lattice / grain-boundary diffusion data identifies the
       mechanism controlling densification.

Master sintering curve (Su & Johnson, J. Am. Ceram. Soc. 79 (1996) 3211): if one
mechanism controls densification, D depends only on the "work of sintering"
    Theta(t) = integral_0^t (1/T) exp(-Q/RT) dt'
so densification curves recorded at different heating rates collapse onto one curve
when plotted against log Theta - for the correct Q only. Scanning Q for the best
collapse gives the activation energy without any isothermal test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.optimize import minimize_scalar

KB = 1.380649e-23
R = 8.314462618


def sigma_eff(sigma_mac, D, D0):
    return sigma_mac * (1.0 - D0) / (D**2 * (D - D0))


def mu_eff(E_th, D, D0, nu_eff=0.3):
    return E_th * (D - D0) / (1.0 - D0) / (2.0 * (1.0 + nu_eff))


@dataclass
class DensificationModel:
    """'True' model used to generate synthetic SPS densification curves (SI units)."""
    n: float = 2.0
    p: float = 2.0
    Q_kJ: float = 190.0
    D0_diff: float = 1.0e-3  # m^2/s, pre-exponential of D_eff
    K: float = 1.0
    b: float = 2.5e-10
    G: float = 2.0e-6  # grain size (m), held constant
    E_RT: float = 200e9
    dEdT: float = -0.05e9  # Pa/K
    D_green: float = 0.55

    def E(self, T):
        return self.E_RT + self.dEdT * (np.asarray(T) - 293.0)

    def D_eff(self, T):
        return self.D0_diff * np.exp(-self.Q_kJ * 1e3 / (R * np.asarray(T)))

    def rate(self, D, T, sigma_mac):
        D = np.minimum(D, 0.9995)
        mu = mu_eff(self.E(T), D, self.D_green)
        se = sigma_eff(sigma_mac, D, self.D_green)
        return mu * self.K * (self.b * self.D_eff(T) / (KB * T)) * (self.b / self.G) ** self.p * (se / mu) ** self.n

    def simulate(self, t, T_of_t, sigma_mac, D_start=None):
        """Integrate dD/dt along a temperature history T_of_t (callable). Returns D(t)."""
        D_start = D_start or self.D_green + 0.03
        sol = solve_ivp(lambda tt, y: [self.rate(y[0], T_of_t(tt), sigma_mac)], (t[0], t[-1]), [D_start],
                        t_eval=t, method="LSODA", rtol=1e-8, atol=1e-10)
        return np.minimum(sol.y[0], 1.0)


def heating_program(rate_K_per_min, T_hold, t_hold_s, T0=573.0):
    t_ramp = (T_hold - T0) / (rate_K_per_min / 60.0)

    def T(t):
        return np.where(np.asarray(t) < t_ramp, T0 + rate_K_per_min / 60.0 * np.asarray(t), T_hold)

    return T, t_ramp + t_hold_s


# ------------------------------------------------------------------------------------- analysis

def densification_rate(t, D, window_frac: float = 0.05, polyorder: int = 2):
    """Savitzky-Golay derivative dD/dt on a uniform grid (robust to displacement noise)."""
    from scipy.signal import savgol_filter

    t = np.asarray(t, float)
    tu = np.linspace(t[0], t[-1], len(t))
    Du = np.interp(tu, t, D)
    win = max(polyorder + 3, int(window_frac * len(tu)) // 2 * 2 + 1)
    rate = savgol_filter(Du, win, polyorder, deriv=1, delta=tu[1] - tu[0])
    return np.interp(t, tu, rate)


def stress_exponent(t, D, T, sigma_mac, E_th, D0, D_window=(0.0, 1.0)):
    """n from the isothermal stage: slope of ln[(1/mu_eff) dD/dt] vs ln(sigma_eff/mu_eff)."""
    rate = densification_rate(t, D)
    mu = mu_eff(E_th, D, D0)
    se = sigma_eff(sigma_mac, D, D0)
    m = (D > D_window[0]) & (D < D_window[1]) & (rate > 0)
    x, y = np.log(se[m] / mu[m]), np.log(rate[m] / mu[m])
    n, c = np.polyfit(x, y, 1)
    return float(n), x, y


def activation_energy_isothermal(hold_data, n, D_ref, E_of_T, D0, sigma_mac):
    """Q from several isothermal holds evaluated at the same relative density D_ref.

    hold_data: list of (T_hold, t, D) arrays restricted to the isothermal part.
    """
    xs, ys = [], []
    for T, t, D in hold_data:
        rate = densification_rate(t, D)
        Dr = np.interp(D_ref, D, D)
        rr = np.interp(D_ref, D, rate)
        mu = mu_eff(E_of_T(T), Dr, D0)
        se = sigma_eff(sigma_mac, Dr, D0)
        xs.append(1.0 / T)
        ys.append(np.log(T / mu * rr * (mu / se) ** n))
    slope, _ = np.polyfit(xs, ys, 1)
    return float(-slope * R / 1000.0), np.array(xs), np.array(ys)


def effective_diffusivity(rate, D, T, sigma_mac, E_th, D0, n, p, b, G, K=1.0):
    mu = mu_eff(E_th, D, D0)
    se = sigma_eff(sigma_mac, D, D0)
    return rate / mu * KB * T / (K * b * (b / G) ** p * (se / mu) ** n)


def work_of_sintering(t, T, Q_kJ):
    integrand = np.exp(-Q_kJ * 1e3 / (R * T)) / T
    return cumulative_trapezoid(integrand, t, initial=0.0) + 1e-300


def msc_scatter(curves, Q_kJ, D_grid=None):
    """Mean variance of log10(Theta) at fixed D across curves (lower = better collapse).

    The variance of log Theta measures the RELATIVE spread of Theta between curves, so it
    does not depend on the absolute magnitude of Theta (which changes enormously with Q).
    D_grid defaults to the density range covered by every curve.
    """
    if D_grid is None:
        lo = max(np.maximum.accumulate(D)[0] for _, _, D in curves) + 0.02
        hi = min(np.maximum.accumulate(D)[-1] for _, _, D in curves) - 0.02
        D_grid = np.linspace(lo, hi, 31)
    logs = []
    for t, T, D in curves:
        th = work_of_sintering(t, T, Q_kJ)
        Dm = np.maximum.accumulate(D)  # densification is irreversible: remove noise wiggles
        Du, iu = np.unique(Dm, return_index=True)
        logs.append(np.interp(D_grid, Du, np.log10(th[iu])))
    return float(np.mean(np.var(np.array(logs), axis=0)))


def fit_msc_activation_energy(curves, Q_range=(80.0, 500.0)):
    res = minimize_scalar(lambda q: msc_scatter(curves, q), bounds=Q_range, method="bounded")
    return float(res.x)
