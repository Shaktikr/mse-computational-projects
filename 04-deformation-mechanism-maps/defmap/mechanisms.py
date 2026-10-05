"""Rate equations of Frost & Ashby, "Deformation-Mechanism Maps" (Pergamon, 1982), Ch. 2.

All stresses are SHEAR stresses sigma_s and all rates SHEAR strain rates gamma_dot, as in
the book. For a uniaxial (tensile) test use von Mises equivalence:
    sigma_s = sigma / sqrt(3),     gamma_dot = sqrt(3) * eps_dot

Temperature-dependent shear modulus:
    mu(T) = mu0 * [1 + (T - 300)/Tm * (Tm/mu0 * dmu/dT)]

1. Low-temperature plasticity - obstacle-controlled glide (eq. 2.12, p = q = 1):
    gamma_dot = gamma0_dot * exp[-(dF / kT) * (1 - sigma_s / tau_hat)]
    tau_hat = tau_hat0 * mu/mu0,   dF = (dF/mu0 b^3) * mu0 b^3

2. Power-law creep with power-law breakdown (eqs. 2.21, 2.26):
    gamma_dot = (A D_eff mu b / kT) * [sinh(alpha' sigma_s/mu)]^n / alpha'^n
    D_eff = D_v + (10 a_c / b^2) (sigma_s/mu)^2 D_c        (lattice + dislocation-core)
    -> reduces to (A D_eff mu b / kT)(sigma_s/mu)^n when alpha' sigma_s/mu << 1

3. Diffusional flow (eq. 2.29):
    gamma_dot = (42 sigma_s Omega / (kT d^2)) * D_eff
    D_eff = D_v + (pi / d) * delta D_b                      (Nabarro-Herring + Coble)

Glide and dislocation creep are ALTERNATIVE mechanisms (the faster one operates);
diffusional flow acts IN PARALLEL with them (rates add). Each point of the map is labelled
by the mechanism contributing most to the total rate, and power-law creep / diffusional
flow are split by which diffusion path (lattice / core / boundary) dominates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

KB = 1.380649e-23
R = 8.314462618
CAP_LOG = np.log(1e30)  # rates are capped at 1e30 1/s to keep logs finite


def _log_sinh(x):
    """Numerically stable log(sinh(x)) for x > 0."""
    x = np.asarray(x, float)
    with np.errstate(divide="ignore"):
        return np.where(x > 20.0, x - np.log(2.0), np.log(np.sinh(np.maximum(x, 1e-300))))


@dataclass
class Material:
    name: str
    Omega_m3: float
    b_m: float
    Tm_K: float
    mu0_MPa: float
    TdmudT: float
    D0v_m2s: float
    Qv_kJ: float
    deltaD0b_m3s: float
    Qb_kJ: float
    acD0c_m4s: float
    Qc_kJ: float
    n: float
    A: float
    alpha_plb: float
    tau_hat0_over_mu0: float
    dF_over_mu0b3: float
    gamma0_dot: float
    ideal_strength_over_mu: float
    source: str = ""

    @classmethod
    def from_json(cls, path: str | Path) -> "Material":
        d = json.loads(Path(path).read_text())
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    # temperature-dependent quantities -------------------------------------------------
    def mu(self, T):
        return self.mu0_MPa * (1.0 + (np.asarray(T) - 300.0) / self.Tm_K * self.TdmudT)

    def Dv(self, T):
        return self.D0v_m2s * np.exp(-self.Qv_kJ * 1e3 / (R * np.asarray(T)))

    def deltaDb(self, T):
        return self.deltaD0b_m3s * np.exp(-self.Qb_kJ * 1e3 / (R * np.asarray(T)))

    def acDc(self, T):
        return self.acD0c_m4s * np.exp(-self.Qc_kJ * 1e3 / (R * np.asarray(T)))


MECHANISMS = [
    "plasticity (glide)",
    "power-law creep (lattice diffusion)",
    "power-law creep (core diffusion)",
    "power-law breakdown",
    "diffusional flow (Nabarro-Herring)",
    "diffusional flow (Coble)",
]


def rates(m: Material, s_over_mu, T, d_m):
    """All partial shear strain rates (1/s). Arguments broadcast against each other.

    Returns a dict mechanism -> rate plus 'total'.
    """
    s = np.asarray(s_over_mu, float)
    T = np.asarray(T, float)
    d = np.asarray(d_m, float)
    mu = m.mu(T) * 1e6  # Pa
    kT = KB * T
    b = m.b_m

    # 1. obstacle-controlled glide. tau_hat scales with mu(T), so in normalised units
    #    tau_hat/mu = tau_hat0/mu0 is temperature independent.
    #    Forward minus backward activation (sinh form) so that the rate vanishes at zero
    #    stress; for sigma_s >> kT tau_hat/dF it equals the F&A forward-only expression.
    tau_hat_over_mu = m.tau_hat0_over_mu0
    dF = m.dF_over_mu0b3 * m.mu0_MPa * 1e6 * b**3
    x = dF / kT
    #    The equation describes thermally ACTIVATED obstacle cutting and is only meaningful
    #    when the work done by the stress over the activation volume exceeds kT
    #    (x * s/tau_hat >= 1); below that, glide is switched off - otherwise it would give
    #    an unphysical linear-viscous "glide" field at T -> Tm, sigma -> 0.
    with np.errstate(over="ignore", under="ignore"):
        log_glide = np.log(m.gamma0_dot) - x + np.log(2.0) + _log_sinh(x * s / tau_hat_over_mu)
        glide = np.exp(np.minimum(log_glide, CAP_LOG))
    glide = np.where(x * s / tau_hat_over_mu >= 1.0, glide, 0.0)

    # 2. power-law creep / breakdown, split into lattice and core contributions
    Dv = m.Dv(T)
    Dc_term = 10.0 * m.acDc(T) / b**2 * s**2
    pre = m.A * mu * b / kT
    a = m.alpha_plb
    log_plb = m.n * (_log_sinh(a * s) - np.log(a))
    plc_lattice = pre * Dv * s**m.n
    plc_core = pre * Dc_term * s**m.n
    plc_total = np.exp(np.minimum(np.log(pre * (Dv + Dc_term)) + log_plb, CAP_LOG))
    breakdown_excess = np.maximum(plc_total - plc_lattice - plc_core, 0.0)

    # 3. diffusional flow
    pre_d = 42.0 * s * mu * m.Omega_m3 / (kT * d**2)
    nh = pre_d * Dv
    coble = pre_d * np.pi * m.deltaDb(T) / d

    dislocation = np.maximum(glide, plc_total)
    total = dislocation + nh + coble
    # assign the dislocation part to glide or to the creep sub-mechanisms
    is_glide = glide >= plc_total
    return {
        "plasticity (glide)": np.where(is_glide, glide, 0.0),
        "power-law creep (lattice diffusion)": np.where(is_glide, 0.0, plc_lattice),
        "power-law creep (core diffusion)": np.where(is_glide, 0.0, plc_core),
        "power-law breakdown": np.where(is_glide, 0.0, breakdown_excess),
        "diffusional flow (Nabarro-Herring)": nh,
        "diffusional flow (Coble)": coble,
        "total": total,
    }


def dominant(r: dict, m: Material, s_over_mu):
    """Index into MECHANISMS of the largest contribution; -1 above the ideal strength."""
    stack = np.stack([r[k] for k in MECHANISMS])
    idx = np.argmax(stack, axis=0)
    return np.where(np.asarray(s_over_mu) >= m.ideal_strength_over_mu, -1, idx)
