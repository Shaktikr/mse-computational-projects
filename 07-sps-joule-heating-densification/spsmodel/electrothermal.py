"""Axisymmetric (r-z) finite-volume model of Joule heating in an SPS tool stack.

Geometry (all graphite except the sample), symmetric about z = 0:

        z ^   ______________________
          |  |       spacer         |   <- water-cooled ram contact (h_c to 300 K), +V
          |  |______________________|
          |        |  punch |
          |   _____|________|_____
          |  |     | sample |     |     die (pyrometer spot on the outer wall at z = 0)
          |  |_____|________|_____|
          |        |  punch |
          |   _____|________|______
          |  |       spacer        |   <- ram contact, 0 V
          +---------------------------> r

Governing equations, solved on a cell-centred grid with harmonic-mean face conductances:

  electric   div( sigma_e grad phi ) = 0            (quasi-static, DC)
  thermal    rho c_p dT/dt = div( k grad T ) + q     q = sigma_e |grad phi|^2

Boundary conditions: fixed potential on the two ram faces; convective contact (h_c) to
300 K cooling water on the same faces; grey-body radiation eps*sigma_SB*(T^4 - T_amb^4)
from every surface facing the vacuum chamber. Contact resistances between parts are
neglected (they can be added as interface conductances). A PID controller adjusts the
current so that the pyrometer temperature follows the heating schedule.

Because the electric problem is linear for a given temperature field, it is solved once
for 1 V and the solution scaled to the controller's current.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .materials import Graphite, Material

SIGMA_SB = 5.670374419e-8
VOID, GRAPHITE, SAMPLE = 0, 1, 2


@dataclass
class Geometry:
    """Dimensions in metres (defaults: a typical 20 mm SPS set-up)."""
    r_sample: float = 0.010
    h_sample: float = 0.005
    r_die_out: float = 0.025
    h_die: float = 0.040
    l_punch: float = 0.030
    r_spacer: float = 0.040
    h_spacer: float = 0.015
    dr: float = 0.0005
    dz: float = 0.0005

    def build(self):
        nr = int(round(self.r_spacer / self.dr))
        z_top = self.h_sample / 2 + self.l_punch + self.h_spacer
        nz = int(round(2 * z_top / self.dz))
        r = (np.arange(nr) + 0.5) * self.dr
        z = -z_top + (np.arange(nz) + 0.5) * self.dz
        R, Z = np.meshgrid(r, z, indexing="ij")
        mat = np.full((nr, nz), VOID, dtype=int)
        az = np.abs(Z)
        mat[(R < self.r_sample) & (az < self.h_sample / 2)] = SAMPLE
        mat[(R < self.r_sample) & (az >= self.h_sample / 2) & (az < self.h_sample / 2 + self.l_punch)] = GRAPHITE
        mat[(R >= self.r_sample) & (R < self.r_die_out) & (az < self.h_die / 2)] = GRAPHITE
        mat[az >= self.h_sample / 2 + self.l_punch] = GRAPHITE  # spacers fill r < r_spacer
        return r, z, mat


@dataclass
class Schedule:
    """Pyrometer set-point: ramp from T0 at `rate` K/min to T_hold, then hold."""
    T0: float = 573.0  # pyrometers usually start reading around 570 K
    rate_K_per_min: float = 100.0
    T_hold: float = 1273.0
    t_hold_s: float = 600.0

    def setpoint(self, t):
        return min(self.T0 + self.rate_K_per_min / 60.0 * t, self.T_hold)

    @property
    def duration(self):
        return (self.T_hold - self.T0) / (self.rate_K_per_min / 60.0) + self.t_hold_s


@dataclass
class SPSModel:
    sample: Material
    geom: Geometry = field(default_factory=Geometry)
    graphite: Material = field(default_factory=Graphite)
    h_contact: float = 3000.0  # W/m^2/K to cooling water at the ram faces
    T_water: float = 300.0
    T_amb: float = 300.0
    Kp: float = 40.0  # A/K
    Ki: float = 1.5  # A/(K s)
    I_max: float = 6000.0

    def __post_init__(self):
        self.r, self.z, self.mat = self.geom.build()
        self.nr, self.nz = self.mat.shape
        self.N = self.nr * self.nz
        dr, dz = self.geom.dr, self.geom.dz
        r = self.r
        self.vol = (2 * np.pi * r * dr * dz)[:, None] * np.ones((1, self.nz))
        self.A_r = (2 * np.pi * (r + dr / 2) * dz)  # outer radial face area of column i
        self.A_z = 2 * np.pi * r * dr  # axial face area of column i
        self.solid = self.mat != VOID
        self.idx = np.arange(self.N).reshape(self.nr, self.nz)
        # pyrometer: outermost die cell at mid-height; sample centre
        i_die = int(np.searchsorted(r, self.geom.r_die_out) - 1)
        self.pyro = (i_die, self.nz // 2)
        self.centre = (0, self.nz // 2)

    # ---------------------------------------------------------------- properties
    def props(self, T):
        g, s = self.graphite, self.sample
        is_s = self.mat == SAMPLE
        sig = np.where(is_s, 1.0 / s.rho_e(T), 1.0 / g.rho_e(T))
        k = np.where(is_s, s.k(T), g.k(T))
        rc = np.where(is_s, s.rho_cp(T), g.rho_cp(T))
        eps = np.where(is_s, s.emissivity, g.emissivity)
        sig[~self.solid] = 0.0
        k[~self.solid] = 0.0
        return sig, k, rc, eps

    # ---------------------------------------------------------------- face conductances
    def _faces(self, c):
        """Harmonic conductances between neighbouring cells for a cell property c."""
        dr, dz = self.geom.dr, self.geom.dz
        with np.errstate(divide="ignore", invalid="ignore"):
            cr = np.where((c[:-1] > 0) & (c[1:] > 0), self.A_r[:-1, None] / (dr / (2 * c[:-1]) + dr / (2 * c[1:])), 0.0)
            cz = np.where((c[:, :-1] > 0) & (c[:, 1:] > 0),
                          self.A_z[:, None] / (dz / (2 * c[:, :-1]) + dz / (2 * c[:, 1:])), 0.0)
        return np.nan_to_num(cr), np.nan_to_num(cz)

    def _laplacian(self, Gr, Gz):
        I, J = self.idx, self.idx
        rows, cols, vals = [], [], []
        a, b = I[:-1, :].ravel(), I[1:, :].ravel()
        g = Gr.ravel()
        a2, b2 = J[:, :-1].ravel(), J[:, 1:].ravel()
        g2 = Gz.ravel()
        for (p, q, w) in ((a, b, g), (a2, b2, g2)):
            rows += [p, q, p, q]
            cols += [q, p, p, q]
            vals += [-w, -w, w, w]
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(self.N, self.N))

    # ---------------------------------------------------------------- electric problem
    def solve_potential(self, sig):
        """phi for +1 V at the top ram face and 0 V at the bottom. Returns (phi, current A, Gr, Gz)."""
        Gr, Gz = self._faces(sig)
        L = self._laplacian(Gr, Gz)
        rhs = np.zeros(self.N)
        dz = self.geom.dz
        top, bot = self.idx[:, -1], self.idx[:, 0]
        g_top = np.where(sig[:, -1] > 0, self.A_z * 2 * sig[:, -1] / dz, 0.0)
        g_bot = np.where(sig[:, 0] > 0, self.A_z * 2 * sig[:, 0] / dz, 0.0)
        diag = np.zeros(self.N)
        diag[top] += g_top
        diag[bot] += g_bot
        diag[~self.solid.ravel()] = 1.0  # void cells: decoupled rows (phi = 0)
        rhs[top] = g_top * 1.0
        L = (L + sp.diags(diag)).tocsr()
        phi = spla.spsolve(L, rhs).reshape(self.nr, self.nz)
        current = float(np.sum(g_top * (1.0 - phi[:, -1])))
        return phi, current, Gr, Gz

    def joule_power(self, phi, Gr, Gz):
        """Cell heat sources (W) from face dissipation G (dphi)^2, split half/half."""
        q = np.zeros_like(phi)
        pr = Gr * (phi[1:] - phi[:-1]) ** 2
        pz = Gz * (phi[:, 1:] - phi[:, :-1]) ** 2
        q[:-1] += pr / 2
        q[1:] += pr / 2
        q[:, :-1] += pz / 2
        q[:, 1:] += pz / 2
        dz = self.geom.dz
        # dissipation in the half cells next to the electrodes
        q[:, -1] += np.where(self.solid[:, -1], self.A_z * 2 * self._sig[:, -1] / dz * (1 - phi[:, -1]) ** 2, 0)
        q[:, 0] += np.where(self.solid[:, 0], self.A_z * 2 * self._sig[:, 0] / dz * phi[:, 0] ** 2, 0)
        return q

    # ---------------------------------------------------------------- radiation faces
    def _radiating_area(self):
        """Area of each solid cell's faces that see the vacuum chamber (m^2)."""
        A = np.zeros((self.nr, self.nz))
        s = self.solid
        # radial neighbours
        A[:-1] += np.where(s[:-1] & ~s[1:], self.A_r[:-1, None], 0)
        A[1:] += np.where(s[1:] & ~s[:-1], self.A_r[:-1, None], 0)
        A[-1] += np.where(s[-1], self.A_r[-1], 0)  # outer boundary
        # axial neighbours
        A[:, :-1] += np.where(s[:, :-1] & ~s[:, 1:], self.A_z[:, None], 0)
        A[:, 1:] += np.where(s[:, 1:] & ~s[:, :-1], self.A_z[:, None], 0)
        return A

    # ---------------------------------------------------------------- time integration
    def run(self, schedule: Schedule, dt: float = 1.0, T_init: float | None = None, electric_every: int = 2,
            record_every: int = 10):
        T = np.full((self.nr, self.nz), T_init or schedule.T0 - 30.0)
        A_rad = self._radiating_area()
        A_cool = np.zeros((self.nr, self.nz))
        A_cool[:, 0] = np.where(self.solid[:, 0], self.A_z, 0)
        A_cool[:, -1] = np.where(self.solid[:, -1], self.A_z, 0)
        integ, I = 0.0, 0.0
        hist = {k: [] for k in ("t", "T_set", "T_pyro", "T_centre", "T_sample_max", "T_sample_min", "I", "V", "P",
                                "frac_current_sample")}
        nsteps = int(schedule.duration / dt)
        for n in range(nsteps + 1):
            t = n * dt
            sig, k, rc, eps = self.props(T)
            self._sig = sig
            if n % electric_every == 0:
                phi1, I1, Gr_e, Gz_e = self.solve_potential(sig)
                q1 = self.joule_power(phi1, Gr_e, Gz_e)  # W at 1 V
                R_total = 1.0 / I1
                # current through the sample = axial current crossing z = 0 inside r_sample
                jz0 = Gz_e[:, self.nz // 2 - 1] * (phi1[:, self.nz // 2] - phi1[:, self.nz // 2 - 1])
                frac = float(np.abs(jz0[self.r < self.geom.r_sample]).sum() / np.abs(jz0).sum())
            # PID (PI) control on the pyrometer temperature
            e = schedule.setpoint(t) - T[self.pyro]
            integ = np.clip(integ + e * dt, -1e5, self.I_max / max(self.Ki, 1e-9))
            I = float(np.clip(self.Kp * e + self.Ki * integ, 0.0, self.I_max))
            V = I * R_total
            q = q1 * V**2
            # implicit Euler step: (rc V/dt + L + h_rad + h_cool) T_new = rc V/dt T + q + ...
            Gr, Gz = self._faces(k)
            L = self._laplacian(Gr, Gz)
            h_rad = eps * SIGMA_SB * (T**2 + self.T_amb**2) * (T + self.T_amb) * A_rad
            diag = (rc * self.vol / dt + h_rad + self.h_contact * A_cool).ravel()
            rhs = (rc * self.vol / dt * T + q + h_rad * self.T_amb + self.h_contact * A_cool * self.T_water).ravel()
            void = ~self.solid.ravel()
            diag[void] = 1.0
            rhs[void] = self.T_amb
            M = (L + sp.diags(diag)).tocsr()  # void rows of L are empty -> identity rows
            T = spla.spsolve(M, rhs).reshape(self.nr, self.nz)
            if n % record_every == 0 or n == nsteps:
                s = self.mat == SAMPLE
                for key, val in (("t", t), ("T_set", schedule.setpoint(t)), ("T_pyro", T[self.pyro]),
                                 ("T_centre", T[self.centre]), ("T_sample_max", T[s].max()), ("T_sample_min", T[s].min()),
                                 ("I", I), ("V", V), ("P", I * V), ("frac_current_sample", frac)):
                    hist[key].append(float(val))
        self.T, self.phi, self.q = T, phi1 * V, q
        return {k: np.array(v) for k, v in hist.items()}
