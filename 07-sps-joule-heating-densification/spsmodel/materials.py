"""Temperature-dependent material properties (SI units, T in K).

The graphite data are representative of fine-grained isostatic graphite used for SPS dies
and punches; powder-compact properties depend strongly on relative density D through
percolation of particle contacts. All values are approximate - replace them with data
for your own tooling grade and powder when quantitative agreement is needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Material:
    name: str

    def rho_e(self, T):  # electrical resistivity, Ohm m
        raise NotImplementedError

    def k(self, T):  # thermal conductivity, W/m/K
        raise NotImplementedError

    def rho_cp(self, T):  # volumetric heat capacity, J/m^3/K
        raise NotImplementedError

    emissivity: float = 0.8


@dataclass
class Graphite(Material):
    name: str = "isostatic graphite"
    emissivity: float = 0.8

    def rho_e(self, T):
        T = np.asarray(T, float)
        return 1.6e-5 - 1.0e-8 * (T - 300.0) + 6.0e-12 * (T - 300.0) ** 2

    def k(self, T):
        return 120.0 * (300.0 / np.asarray(T, float)) ** 0.6

    def rho_cp(self, T):
        T = np.asarray(T, float)
        return 1850.0 * (1900.0 - 1200.0 * np.exp(-(T - 300.0) / 450.0))


def percolation_factor(D, Dc=0.40, t=1.5):
    """Effective-medium scaling of conductivity with relative density (0 below Dc)."""
    return np.clip((np.asarray(D, float) - Dc) / (1.0 - Dc), 0.0, 1.0) ** t


@dataclass
class NickelCompact(Material):
    """Ni powder compact (electrically conductive metal powder)."""
    name: str = "Ni powder compact"
    D: float = 0.65
    emissivity: float = 0.4

    def rho_e(self, T):
        bulk = 6.9e-8 + 3.4e-10 * (np.asarray(T, float) - 293.0)
        return bulk / max(percolation_factor(self.D), 1e-9)

    def k(self, T):
        bulk = 90.0 - 0.02 * (np.asarray(T, float) - 300.0)
        return np.maximum(bulk * percolation_factor(self.D), 0.5)

    def rho_cp(self, T):
        T = np.asarray(T, float)
        return self.D * 8900.0 * (430.0 + 0.15 * (T - 300.0))


@dataclass
class AluminaCompact(Material):
    """Al2O3 powder compact (electrical insulator: heated only by conduction)."""
    name: str = "Al2O3 powder compact"
    D: float = 0.60
    emissivity: float = 0.7

    def rho_e(self, T):
        return np.full_like(np.asarray(T, float), 1.0e8)

    def k(self, T):
        bulk = 30.0 * (300.0 / np.asarray(T, float)) ** 1.1
        return np.maximum(bulk * percolation_factor(self.D, Dc=0.3, t=1.5), 0.3)

    def rho_cp(self, T):
        T = np.asarray(T, float)
        return self.D * 3950.0 * (780.0 + 0.45 * (T - 300.0) - 1.2e-4 * (T - 300.0) ** 2)
