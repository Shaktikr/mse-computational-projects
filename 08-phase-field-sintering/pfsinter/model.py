"""Phase-field model of solid-state sintering.

Order parameters
  rho(x)     conserved density field: 1 inside particles, 0 in the pores
  eta_i(x)   non-conserved grain fields: eta_i = 1 inside particle/grain i

Free energy (Y.U. Wang, Acta Mater. 54 (2006) 953):
  f = A rho^2 (1-rho)^2 + B [ rho^2 + 6(1-rho) S2 - 4(2-rho) S3 + 3 S2^2 ],
  S2 = sum eta_i^2,  S3 = sum eta_i^3
  F = integral f + kappa_rho/2 |grad rho|^2 + sum kappa_eta/2 |grad eta_i|^2

Kinetics
  d rho / dt   = div( M grad mu ),   mu = df/drho - kappa_rho lap(rho)        (Cahn-Hilliard)
  d eta_i / dt = -L ( df/deta_i - kappa_eta lap(eta_i) )                   (Allen-Cahn)
  M = M_vol phi + M_vap (1 - phi) + M_surf 16 rho^2 (1-rho)^2 + M_gb sum_{i!=j} eta_i eta_j
  phi = rho^3 (10 - 15 rho + 6 rho^2)

The surface term is active only on the particle surface (rho ~ 1/2), the grain-boundary
term only where two grains meet - so the ratio M_surf / M_gb selects the dominant
diffusion path. Rigid-body motion of particles (Wang's advection term) is NOT included,
so this version captures neck growth, pore rounding and grain growth, but not the
centre-to-centre approach (shrinkage) that rigid-body motion produces.

Time stepping: semi-implicit Fourier-spectral scheme on a periodic grid with a
stabilising constant-mobility term (Zhu, Chen, Shen & Tikare, Phys. Rev. E 60, 3564 (1999)),
which allows time steps ~100x larger than explicit Euler.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # scipy.fft is faster and multithreaded; numpy.fft as fallback
    from scipy import fft as _fft
except ImportError:  # pragma: no cover
    _fft = np.fft


@dataclass
class Params:
    A: float = 16.0
    B: float = 1.0
    kappa_rho: float = 10.0
    kappa_eta: float = 5.0
    L: float = 10.0
    M_vol: float = 0.04
    M_vap: float = 0.002
    M_surf: float = 4.0
    M_gb: float = 0.4
    dx: float = 1.0
    dt: float = 0.05


class SinteringPF:
    def __init__(self, rho: np.ndarray, etas: np.ndarray, p: Params = Params()):
        self.p = p
        self.rho = rho.astype(float).copy()
        self.etas = etas.astype(float).copy()  # shape (n_grains, ny, nx)
        ny, nx = rho.shape
        kx = 2 * np.pi * _fft.fftfreq(nx, d=p.dx)
        ky = 2 * np.pi * _fft.fftfreq(ny, d=p.dx)
        self.KX, self.KY = np.meshgrid(kx, ky)
        self.K2 = self.KX**2 + self.KY**2
        self.K4 = self.K2**2
        self.t = 0.0
        self.M_max = p.M_vol + p.M_surf + 0.5 * p.M_gb + p.M_vap

    # ----------------------------------------------------------------- thermodynamics
    def dfdrho(self, rho, S2, S3):
        p = self.p
        return 2 * p.A * rho * (1 - rho) * (1 - 2 * rho) + p.B * (2 * rho - 6 * S2 + 4 * S3)

    def dfdeta(self, rho, eta, S2):
        p = self.p
        return p.B * (12 * (1 - rho) * eta - 12 * (2 - rho) * eta**2 + 12 * eta * S2)

    def mobility(self, rho, etas):
        p = self.p
        r = np.clip(rho, 0.0, 1.0)
        phi = r**3 * (10 - 15 * r + 6 * r**2)
        S1 = etas.sum(axis=0)
        gb = S1**2 - (etas**2).sum(axis=0)  # = sum_{i != j} eta_i eta_j
        return p.M_vol * phi + p.M_vap * (1 - phi) + p.M_surf * 16 * r**2 * (1 - r) ** 2 + p.M_gb * np.clip(gb, 0, None)

    def free_energy(self):
        p = self.p
        rho, etas = self.rho, self.etas
        S2 = (etas**2).sum(0)
        S3 = (etas**3).sum(0)
        f = p.A * rho**2 * (1 - rho) ** 2 + p.B * (rho**2 + 6 * (1 - rho) * S2 - 4 * (2 - rho) * S3 + 3 * S2**2)
        gx, gy = np.gradient(rho, p.dx)
        grad = 0.5 * p.kappa_rho * (gx**2 + gy**2)
        for e in etas:
            ex, ey = np.gradient(e, p.dx)
            grad += 0.5 * p.kappa_eta * (ex**2 + ey**2)
        return float(np.sum(f + grad) * p.dx**2)

    # ----------------------------------------------------------------- time stepping
    def step(self, nsteps: int = 1):
        p = self.p
        fft2, ifft2 = _fft.fft2, _fft.ifft2
        # linear stabilisation: the stiff parts (bulk curvature S and gradient energy) are
        # added implicitly and subtracted explicitly, so the scheme stays consistent but
        # tolerates much larger time steps (Zhu et al. 1999; Eyre-type convex splitting)
        A0 = self.M_max
        S_rho = 2.0 * p.A + 6.0 * p.B
        S_eta = 36.0 * p.B
        lin_rho = A0 * (p.kappa_rho * self.K4 + S_rho * self.K2)
        denom_rho = 1.0 + p.dt * lin_rho
        denom_eta = 1.0 + p.dt * p.L * (p.kappa_eta * self.K2 + S_eta)
        for _ in range(nsteps):
            rho, etas = self.rho, self.etas
            S2 = (etas**2).sum(0)
            S3 = (etas**3).sum(0)
            rho_h = fft2(rho)
            mu = self.dfdrho(rho, S2, S3) + p.kappa_rho * np.real(ifft2(self.K2 * rho_h))
            mu_h = fft2(mu)
            M = self.mobility(rho, etas)
            jx = M * np.real(ifft2(1j * self.KX * mu_h))
            jy = M * np.real(ifft2(1j * self.KY * mu_h))
            div_h = 1j * self.KX * fft2(jx) + 1j * self.KY * fft2(jy)
            rho_new = np.real(ifft2((rho_h + p.dt * div_h + p.dt * lin_rho * rho_h) / denom_rho))
            new_etas = np.empty_like(etas)
            for i, e in enumerate(etas):
                g = self.dfdeta(rho, e, S2)
                e_h = fft2(e)
                new_etas[i] = np.real(ifft2((e_h - p.dt * p.L * fft2(g) + p.dt * p.L * S_eta * e_h) / denom_eta))
            self.rho, self.etas = rho_new, new_etas
            self.t += p.dt
        return self

    # ----------------------------------------------------------------- diagnostics
    def grain_map(self, threshold: float = 0.5):
        """Index of the dominant grain at each pixel (-1 in pores)."""
        g = np.argmax(self.etas, axis=0)
        return np.where(self.rho > threshold, g, -1)

    def surface_length(self):
        """Total pore/particle interface length (2D 'surface area') ~ integral |grad rho| ."""
        gx, gy = np.gradient(self.rho, self.p.dx)
        return float(np.sum(np.sqrt(gx**2 + gy**2)) * self.p.dx**2)

    def density_fraction(self):
        return float(np.mean(self.rho > 0.5))
