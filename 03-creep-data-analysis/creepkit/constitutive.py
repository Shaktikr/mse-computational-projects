"""Constitutive analysis of a set of creep tests.

Power-law (Mukherjee-Bird-Dorn / Norton) creep:

    eps_dot_min = A * sigma^n * exp(-Q / RT)                                         (1)

    n  stress exponent      ~1   diffusional creep (Nabarro-Herring / Coble) or
                                  Harper-Dorn;  ~2 grain-boundary sliding / superplasticity;
                            ~3   viscous glide (solute drag, class A alloys);
                            4-5  dislocation climb (class M / pure metals);
                            >>5  particle-strengthened alloys -> look for a threshold stress
    Q  apparent activation energy; compare with lattice (Q_L), pipe or grain-boundary
       diffusion (Q_gb ~ 0.4-0.6 Q_L) to identify the rate-controlling process.

With a threshold stress sigma_th (dispersion / composite strengthening):

    eps_dot_min = A' * ((sigma - sigma_th) / E)^n_true * exp(-Q_true / RT)           (2)

sigma_th is found by plotting eps_dot^(1/n) vs sigma on LINEAR axes for trial n and
extrapolating to zero rate (Lagneborg & Bergman 1976; Li, Nutt & Mohamed 1997); the n
giving the best straight line is taken as n_true.

Rupture-life correlations:
    Larson-Miller  LMP = T [K] * (C + log10 t_r [h])          C ~ 20
    Monkman-Grant  eps_dot_min^m * t_r = C_MG                 m ~ 0.8-1

Hot working (high stress, power-law breakdown) - Garofalo sinh law / Zener-Hollomon:
    Z = eps_dot * exp(Q/RT) = A [sinh(alpha sigma)]^n
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats
from scipy.optimize import least_squares, minimize_scalar

R = 8.314462618  # J/(mol K)


@dataclass
class LinearFit:
    slope: float
    intercept: float
    slope_stderr: float
    r2: float

    def to_dict(self):
        return asdict(self)


def _linfit(x, y) -> LinearFit:
    r = stats.linregress(x, y)
    return LinearFit(float(r.slope), float(r.intercept), float(r.stderr), float(r.rvalue**2))


# --------------------------------------------------------------------------------------
# Norton and Arrhenius (one variable at a time)
# --------------------------------------------------------------------------------------

def fit_norton(stress_MPa, rate):
    """n from ln(rate) vs ln(stress) at ONE temperature. Returns (n, LinearFit)."""
    f = _linfit(np.log(stress_MPa), np.log(rate))
    return f.slope, f


def fit_arrhenius(T_K, rate):
    """Q (kJ/mol) from ln(rate) vs 1/T at ONE stress. Returns (Q, Q_stderr, LinearFit)."""
    f = _linfit(1.0 / np.asarray(T_K), np.log(rate))
    return -f.slope * R / 1000.0, f.slope_stderr * R / 1000.0, f


def youngs_modulus_linear(E_RT_GPa: float, dEdT_GPa_per_K: float):
    """E(T) = E_RT + dE/dT * (T - 293); returns a callable of T in K (GPa)."""
    return lambda T: E_RT_GPa + dEdT_GPa_per_K * (np.asarray(T) - 293.0)


# --------------------------------------------------------------------------------------
# Global fit of equation (1) to all tests at once
# --------------------------------------------------------------------------------------

@dataclass
class PowerLawFit:
    n: float
    n_stderr: float
    Q_kJ: float
    Q_stderr_kJ: float
    lnA: float
    r2: float
    modulus_compensated: bool

    def predict(self, stress_MPa, T_K, E_of_T=None):
        s = np.asarray(stress_MPa, float)
        T = np.asarray(T_K, float)
        if self.modulus_compensated:
            s = s / (1000.0 * E_of_T(T))  # sigma/E dimensionless
        return np.exp(self.lnA + self.n * np.log(s) - self.Q_kJ * 1000.0 / (R * T))

    def to_dict(self):
        return asdict(self)


def fit_power_law(stress_MPa, T_K, rate, E_of_T=None) -> PowerLawFit:
    """Multiple linear regression  ln(rate) = lnA + n ln(s) - Q/(RT).

    If E_of_T (GPa) is given, s = sigma/E(T) (modulus-compensated), which removes the
    temperature dependence of the modulus from Q (Q_true < Q_apparent).
    """
    s = np.asarray(stress_MPa, float)
    T = np.asarray(T_K, float)
    y = np.log(np.asarray(rate, float))
    if E_of_T is not None:
        s = s / (1000.0 * E_of_T(T))
    X = np.column_stack([np.ones_like(s), np.log(s), -1.0 / (R * T)])
    beta, res, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    dof = max(len(y) - 3, 1)
    sigma2 = np.sum((y - yhat) ** 2) / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    return PowerLawFit(
        n=float(beta[1]), n_stderr=float(se[1]),
        Q_kJ=float(beta[2] / 1000.0), Q_stderr_kJ=float(se[2] / 1000.0),
        lnA=float(beta[0]), r2=float(r2), modulus_compensated=E_of_T is not None,
    )


# --------------------------------------------------------------------------------------
# Threshold stress
# --------------------------------------------------------------------------------------

@dataclass
class ThresholdResult:
    n_trial: float
    sigma_th_MPa: float
    r2: float


def threshold_stress(stress_MPa, rate, n_trials=(3.0, 4.4, 5.0, 8.0)):
    """Linear extrapolation of rate^(1/n) vs sigma to zero rate for each trial n.

    Returns a list of ThresholdResult (one per n) and the best one (highest R^2).
    """
    s = np.asarray(stress_MPa, float)
    r = np.asarray(rate, float)
    out = []
    for n in n_trials:
        f = _linfit(s, r ** (1.0 / n))
        out.append(ThresholdResult(float(n), float(-f.intercept / f.slope), f.r2))
    best = max(out, key=lambda x: x.r2)
    return out, best


def threshold_stress_by_temperature(stress_MPa, T_K, rate, n_trials=(3.0, 4.4, 5.0, 8.0)):
    """Pick ONE n_true for all temperatures (highest mean R^2), then sigma_th(T).

    Choosing n separately at each temperature is fragile (R^2 values differ in the 4th
    decimal); physically the deformation mechanism - hence n_true - should not change
    across a modest temperature window, so a common n is the more robust choice.
    Returns (n_true, {T_K: sigma_th_MPa}, {n: mean_r2}).
    """
    s = np.asarray(stress_MPa, float)
    T = np.asarray(T_K, float)
    r = np.asarray(rate, float)
    temps = sorted(set(T.round(2)))
    mean_r2 = {}
    for n in n_trials:
        r2s = [_linfit(s[np.isclose(T, Tk)], r[np.isclose(T, Tk)] ** (1.0 / n)).r2 for Tk in temps]
        mean_r2[n] = float(np.mean(r2s))
    n_best = max(mean_r2, key=mean_r2.get)
    sig = {}
    for Tk in temps:
        m = np.isclose(T, Tk)
        f = _linfit(s[m], r[m] ** (1.0 / n_best))
        sig[float(Tk)] = float(-f.intercept / f.slope)
    return n_best, sig, mean_r2


# --------------------------------------------------------------------------------------
# Rupture-life parameters
# --------------------------------------------------------------------------------------

def larson_miller(T_K, t_r_h, C: float = 20.0):
    """LMP = T (C + log10 t_r) / 1000  (conventional x10^-3 scaling)."""
    return np.asarray(T_K) * (C + np.log10(np.asarray(t_r_h))) / 1000.0


def fit_larson_miller_constant(stress_MPa, T_K, t_r_h, degree: int = 2, C_range=(5.0, 40.0)):
    """Find C that collapses all data onto ONE master curve log10(sigma) = poly(LMP).

    Returns (C, poly_coeffs, rms_residual_in_log10_sigma).
    """
    ls = np.log10(stress_MPa)

    def scatter(C):
        x = larson_miller(T_K, t_r_h, C)
        p = np.polyfit(x, ls, degree)
        return np.sqrt(np.mean((np.polyval(p, x) - ls) ** 2))

    res = minimize_scalar(scatter, bounds=C_range, method="bounded")
    C = float(res.x)
    p = np.polyfit(larson_miller(T_K, t_r_h, C), ls, degree)
    return C, p, float(res.fun)


def predict_rupture_life_lmp(stress_MPa, T_K, C, poly):
    """Invert the master curve: given sigma and T, solve poly(LMP) = log10 sigma for LMP."""
    target = np.log10(stress_MPa)
    p = np.array(poly, float).copy()
    p[-1] -= target
    roots = np.roots(p)
    roots = roots[np.isreal(roots)].real
    if len(roots) == 0:
        return np.nan
    # the physically meaningful branch is the one where stress decreases with LMP
    deriv = np.polyval(np.polyder(poly), roots)
    roots = roots[deriv < 0] if np.any(deriv < 0) else roots
    lmp = roots.min() if len(roots) else np.nan
    return 10 ** (1000.0 * lmp / T_K - C)


def fit_monkman_grant(rate_min, t_r_h):
    """log10 t_r = log10 C_MG - m log10 eps_dot_min.  Returns (m, C_MG, LinearFit)."""
    f = _linfit(np.log10(rate_min), np.log10(np.asarray(t_r_h) * 3600.0))
    return -f.slope, 10 ** f.intercept, f


# --------------------------------------------------------------------------------------
# Hot deformation: Garofalo sinh law
# --------------------------------------------------------------------------------------

@dataclass
class SinhFit:
    alpha_per_MPa: float
    n: float
    Q_kJ: float
    lnA: float

    def zener_hollomon(self, rate, T_K):
        return np.asarray(rate) * np.exp(self.Q_kJ * 1000.0 / (R * np.asarray(T_K)))

    def flow_stress(self, rate, T_K):
        Z = self.zener_hollomon(rate, T_K)
        return np.arcsinh((Z / np.exp(self.lnA)) ** (1.0 / self.n)) / self.alpha_per_MPa

    def to_dict(self):
        return asdict(self)


def fit_sinh_law(stress_MPa, T_K, rate) -> SinhFit:
    """Fit eps_dot = A [sinh(alpha sigma)]^n exp(-Q/RT).

    Initial guess by the classical route: n1 = dln(rate)/dln(sigma) (low stress),
    beta = dln(rate)/d(sigma) (high stress), alpha = beta/n1; then a joint nonlinear
    least-squares refinement in log space.
    """
    s = np.asarray(stress_MPa, float)
    T = np.asarray(T_K, float)
    y = np.log(np.asarray(rate, float))
    n1 = _linfit(np.log(s), y).slope
    beta = _linfit(s, y).slope
    alpha0 = beta / n1
    pl = fit_power_law(s, T, np.exp(y))

    def resid(p):
        alpha, n, Q, lnA = p
        return lnA + n * np.log(np.sinh(alpha * s)) - Q * 1000.0 / (R * T) - y

    p0 = [alpha0, n1 * 0.8, pl.Q_kJ, pl.lnA]
    sol = least_squares(resid, p0, bounds=([1e-6, 0.5, 10, -200], [1.0, 20, 2000, 200]))
    a, n, Q, lnA = sol.x
    return SinhFit(float(a), float(n), float(Q), float(lnA))
