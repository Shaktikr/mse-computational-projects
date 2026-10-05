import numpy as np
import pytest

from spsmodel.densification import (DensificationModel, fit_msc_activation_energy, heating_program, mu_eff,
                                    sigma_eff, stress_exponent)
from spsmodel.electrothermal import Geometry, Schedule, SPSModel
from spsmodel.materials import AluminaCompact, NickelCompact


def test_effective_stress_and_modulus_limits():
    assert sigma_eff(50e6, 1.0, 0.6) == pytest.approx(50e6)  # dense body carries the applied stress
    assert mu_eff(200e9, 1.0, 0.6) == pytest.approx(200e9 / 2.6)


def test_stack_resistance_and_heating_coarse_grid():
    g = Geometry(dr=0.001, dz=0.001)
    m = SPSModel(NickelCompact(), geom=g)
    sig = m.props(np.full((m.nr, m.nz), 600.0))[0]
    _, I1, _, _ = m.solve_potential(sig)
    assert 0.5e-3 < 1.0 / I1 < 10e-3  # a few milliohm, typical of SPS tool stacks
    h = m.run(Schedule(rate_K_per_min=200, T_hold=900, t_hold_s=10), dt=5.0, record_every=1)
    assert h["T_pyro"][-1] == pytest.approx(900, abs=40)  # controller tracks the set-point
    assert np.all(h["I"] >= 0)


def test_insulating_sample_carries_no_current():
    m = SPSModel(AluminaCompact(), geom=Geometry(dr=0.001, dz=0.001))
    h = m.run(Schedule(rate_K_per_min=200, T_hold=700, t_hold_s=0), dt=5.0, record_every=1)
    assert h["frac_current_sample"].max() < 1e-6


def test_densification_analysis_recovers_truth():
    m = DensificationModel()
    Tf, dur = heating_program(100.0, 1098.0, 1200.0)
    t = np.linspace(0, dur, 3000)
    D = m.simulate(t, Tf, 50e6)
    k = t >= dur - 1200
    n, _, _ = stress_exponent(t[k], D[k], 1098.0, 50e6, m.E(1098.0), m.D_green, (D[k][0] + 0.01, 0.97))
    assert n == pytest.approx(m.n, abs=0.05)
    curves = []
    for rate in (50, 100, 200):
        Tf, dur = heating_program(rate, 1273.0, 0.0)
        tt = np.linspace(0, dur, 1500)
        curves.append((tt, Tf(tt), m.simulate(tt, Tf, 50e6)))
    assert fit_msc_activation_energy(curves) == pytest.approx(m.Q_kJ, rel=0.05)
