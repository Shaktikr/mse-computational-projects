"""Diffusion-controlled creep rates built on a (DFT-derived) diffusion coefficient.

Diffusional creep (stress exponent n = 1), uniaxial stress sigma, grain size d:

    Nabarro-Herring (lattice diffusion through the grains)
        eps_dot_NH = A_NH * Omega * D_L * sigma / (k T d^2),          A_NH ~ 14
    Coble (diffusion along grain boundaries, width delta)
        eps_dot_C  = A_C  * Omega * delta D_gb * sigma / (k T d^3),   A_C = 150/pi ~ 48

Dislocation (power-law) creep, Mukherjee-Bird-Dorn form:

        eps_dot_PL = A_MBD * (D_L G b / k T) * (sigma / G)^n

The crossover grain size where NH = Coble is  d* = (A_C/A_NH) * delta D_gb / D_L: below d*
grain-boundary diffusion dominates, which is why nanocrystalline / ultrafine-grained
materials (e.g. nanograined B2 aluminides made by milling + SPS) can creep much faster
than their coarse-grained counterparts at the same homologous temperature.

Constants follow Herring, J. Appl. Phys. 21, 437 (1950); Coble, J. Appl. Phys. 34, 1679
(1963); Mukherjee, Bird & Dorn, Trans. ASM 62, 155 (1969).
"""

from __future__ import annotations

import numpy as np

KB = 1.380649e-23
A_NH = 14.0
A_COBLE = 150.0 / np.pi


def nabarro_herring(sigma_MPa, T_K, d_m, D_L, Omega_m3):
    return A_NH * Omega_m3 * D_L * np.asarray(sigma_MPa) * 1e6 / (KB * np.asarray(T_K) * np.asarray(d_m) ** 2)


def coble(sigma_MPa, T_K, d_m, deltaD_gb, Omega_m3):
    return A_COBLE * Omega_m3 * deltaD_gb * np.asarray(sigma_MPa) * 1e6 / (KB * np.asarray(T_K) * np.asarray(d_m) ** 3)


def power_law(sigma_MPa, T_K, D_L, G_MPa, b_m, n=4.5, A=3.0e6):
    """Mukherjee-Bird-Dorn equation (A, n are material constants)."""
    s = np.asarray(sigma_MPa)
    return A * D_L * G_MPa * 1e6 * b_m / (KB * np.asarray(T_K)) * (s / G_MPa) ** n


def crossover_grain_size(T_K, D_L_fn, deltaDgb_fn):
    """d* (m) at which Nabarro-Herring and Coble rates are equal."""
    return (A_COBLE / A_NH) * deltaDgb_fn(T_K) / D_L_fn(T_K)


def deltaD_gb(T_K, delta_D0b_m3s, Qb_kJ):
    """Grain-boundary diffusion product delta*D_gb (m^3/s) in Arrhenius form."""
    return delta_D0b_m3s * np.exp(-Qb_kJ * 1000.0 / (8.314462618 * np.asarray(T_K, float)))
