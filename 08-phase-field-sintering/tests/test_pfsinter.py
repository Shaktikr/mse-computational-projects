import numpy as np

from pfsinter import Params, SinteringPF
from pfsinter.analysis import neck_radius
from pfsinter.geometry import random_packing, two_particles


def test_mass_conservation_stability_and_neck_growth():
    rho, etas, info = two_particles(96, 48, R=12.0, overlap=2.0)
    sim = SinteringPF(rho, etas, Params(dt=0.05))
    m0 = sim.rho.sum()
    x0 = neck_radius(sim.rho, 48)
    F0 = sim.free_energy()
    sim.step(300)
    assert abs(sim.rho.sum() / m0 - 1) < 1e-8  # conserved field
    assert np.isfinite(sim.rho).all() and sim.rho.max() < 1.2
    assert neck_radius(sim.rho, 48) > x0  # neck grows
    assert sim.free_energy() < F0  # total free energy decreases


def test_packing_is_connected():
    rho, etas, info = random_packing(128, 128, n=6, r_mean=12, r_std=2, seed=0)
    c = info["circles"]
    assert len(c) == 6
    # every particle touches at least one other particle
    for i, (x, y, r) in enumerate(c):
        assert any(np.hypot(x - a, y - b) <= r + rr + 0.5 for j, (a, b, rr) in enumerate(c) if j != i)
