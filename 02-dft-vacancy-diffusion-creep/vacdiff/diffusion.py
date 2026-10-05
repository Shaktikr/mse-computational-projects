"""Vacancy-mediated self-diffusion from defect energetics.

For an fcc metal (z = 12 nearest neighbours, jump length a/sqrt(2)):

    D = (1/6) z (a/sqrt2)^2 f nu c_v exp(-E_m / kT)
      = f a^2 nu exp(S_f/k) exp(-(E_f + E_m) / kT)          [c_v = exp(S_f/k - E_f/kT)]

so  D0 = f a^2 nu exp(S_f/k)   and   Q = E_f + E_m

f   = 0.7815, the tracer correlation factor for vacancy diffusion in fcc
nu  = attempt frequency; here estimated as the Debye frequency k theta_D / h
      (theta_D from the DFT elastic constants of project 01). A rigorous value needs
      the harmonic transition-state theory (Vineyard) phonon calculation.
S_f = vacancy formation entropy, typically 0.5-3 k for metals (needs phonons too).

The same structure holds for B2 intermetallics, but diffusion there is far more complex
(six-jump cycles, triple defects, antisites) - see Mishin & Farkas, Phil. Mag. A 75, 169 (1997).
"""

from __future__ import annotations

import numpy as np

KB_EV = 8.617333262e-5  # eV/K
KB = 1.380649e-23  # J/K
H = 6.62607015e-34  # J s
F_FCC = 0.7815
F_BCC = 0.7272


def debye_frequency(theta_D_K: float) -> float:
    return KB * theta_D_K / H


def self_diffusion_fcc(T_K, a_A: float, Ef_eV: float, Em_eV: float, nu_Hz: float,
                       Sf_over_k: float = 0.0, f: float = F_FCC):
    """D(T) in m^2/s."""
    a = a_A * 1e-10
    D0 = f * a**2 * nu_Hz * np.exp(Sf_over_k)
    return D0 * np.exp(-(Ef_eV + Em_eV) / (KB_EV * np.asarray(T_K, float)))


def arrhenius_parameters(a_A, Ef_eV, Em_eV, nu_Hz, Sf_over_k=0.0, f=F_FCC):
    """Return (D0 in m^2/s, Q in eV and kJ/mol)."""
    D0 = f * (a_A * 1e-10) ** 2 * nu_Hz * np.exp(Sf_over_k)
    Q = Ef_eV + Em_eV
    return D0, Q, Q * 96.485


def arrhenius(T_K, D0, Q_kJ):
    return D0 * np.exp(-Q_kJ * 1000.0 / (8.314462618 * np.asarray(T_K, float)))


#: Experimental tracer self-diffusion parameters for comparison (D0 in m^2/s, Q in kJ/mol).
#: Al: Lundy & Murdock, J. Appl. Phys. 33, 1671 (1962), also used by Frost & Ashby (1982).
#: Ni: Frost & Ashby (1982), Table 4.1.
EXPERIMENT = {
    "Al": {"D0": 1.7e-4, "Q_kJ": 142.0},
    "Ni": {"D0": 1.9e-4, "Q_kJ": 284.0},
}
