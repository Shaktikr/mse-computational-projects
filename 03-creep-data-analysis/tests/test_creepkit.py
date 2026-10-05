import numpy as np
import pytest

from creepkit import constitutive as cs
from creepkit.curves import CreepTest, creep_stages, fit_theta_projection, minimum_creep_rate, theta_strain
from creepkit.io import load_test, save_test
from creepkit.synthetic import COMPOSITE, COMPOSITE_MATRIX, SyntheticAlloy, generate_dataset


def _table(alloy, matrix=None):
    tests = generate_dataset(alloy) if matrix is None else generate_dataset(alloy, matrix)
    s = np.array([t.stress_MPa for t in tests])
    T = np.array([t.temperature_K for t in tests])
    r = np.array([minimum_creep_rate(t).rate_per_s for t in tests])
    tr = np.array([t.rupture_time_h for t in tests])
    return s, T, r, tr


def test_global_fit_recovers_n_and_Q():
    alloy = SyntheticAlloy()
    s, T, r, _ = _table(alloy)
    E = cs.youngs_modulus_linear(alloy.E_RT_GPa, alloy.dEdT_GPa_per_K)
    fit = cs.fit_power_law(s, T, r, E)
    assert fit.n == pytest.approx(alloy.n, abs=0.3)
    assert fit.Q_kJ == pytest.approx(alloy.Q_kJ, rel=0.06)


def test_threshold_stress_recovered():
    s, T, r, _ = _table(COMPOSITE, COMPOSITE_MATRIX)
    n, sig, _ = cs.threshold_stress_by_temperature(s, T, r)
    assert n == pytest.approx(4.4)
    for Tk, v in sig.items():
        assert v == pytest.approx(COMPOSITE.sigma_th(Tk), abs=6.0)


def test_monkman_grant_and_lmp():
    alloy = SyntheticAlloy()
    s, T, r, tr = _table(alloy)
    m, C, f = cs.fit_monkman_grant(r, tr)
    assert m == pytest.approx(alloy.m_MG, abs=0.08)
    Cl, poly, rms = cs.fit_larson_miller_constant(s, T, tr)
    assert rms < 0.05
    pred = cs.predict_rupture_life_lmp(s[0], T[0], Cl, poly)
    assert np.log10(pred) == pytest.approx(np.log10(tr[0]), abs=0.3)


def test_sinh_law_recovery():
    rng = np.random.default_rng(0)
    alpha, n, Q, lnA = 0.012, 4.5, 280.0, 30.0
    T = np.repeat([1173.0, 1273.0, 1373.0], 6)
    s = np.tile(np.linspace(30, 250, 6), 3)
    rate = np.exp(lnA + n * np.log(np.sinh(alpha * s)) - Q * 1e3 / (cs.R * T)) * np.exp(rng.normal(0, 0.02, s.size))
    f = cs.fit_sinh_law(s, T, rate)
    assert f.Q_kJ == pytest.approx(Q, rel=0.05)
    assert f.n == pytest.approx(n, rel=0.1)


def test_theta_projection_and_stages():
    t = np.linspace(0, 100, 400)
    eps = theta_strain(t, 0.002, 0.01, 0.2, 0.0005, 0.05)
    test = CreepTest(t, eps, 100.0, 700.0)
    p, _ = fit_theta_projection(test)
    assert p["theta1"] == pytest.approx(0.01, rel=0.05)
    st = creep_stages(test)
    assert st["primary_end_h"] < st["minimum_rate_time_h"] < st["tertiary_start_h"]


def test_io_round_trip(tmp_path):
    t = generate_dataset(SyntheticAlloy(), {700: [150]})[0]
    save_test(t, tmp_path / "x.csv")
    t2 = load_test(tmp_path / "x.csv")
    assert t2.stress_MPa == 150 and t2.temperature_C == 700
    assert np.allclose(t2.strain, t.strain, rtol=1e-5)
