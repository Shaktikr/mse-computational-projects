from pathlib import Path

import numpy as np

from defmap import MECHANISMS, Material, dominant, rates

NI = Material.from_json(Path(__file__).resolve().parents[1] / "materials" / "nickel.json")


def mech_at(s, TTm, d):
    r = rates(NI, np.array([s]), np.array([TTm * NI.Tm_K]), d)
    return MECHANISMS[int(dominant(r, NI, np.array([s]))[0])]


def test_expected_fields_nickel():
    assert mech_at(1e-2, 0.1, 1e-4) == "plasticity (glide)"  # cold, high stress
    assert mech_at(1e-6, 0.95, 1e-4) == "diffusional flow (Nabarro-Herring)"  # hot, low stress, coarse grains
    assert mech_at(1e-6, 0.5, 1e-6) == "diffusional flow (Coble)"  # fine grains
    assert mech_at(3e-4, 0.8, 1e-4) == "power-law creep (lattice diffusion)"


def test_rates_increase_with_stress_and_temperature():
    s = np.logspace(-6, -2, 30)
    r1 = rates(NI, s, 0.6 * NI.Tm_K, 1e-4)["total"]
    r2 = rates(NI, s, 0.8 * NI.Tm_K, 1e-4)["total"]
    assert np.all(np.diff(r1) > 0) and np.all(r2 > r1)


def test_power_law_slope():
    s = np.array([1e-4, 2e-4])
    r = rates(NI, s, 0.8 * NI.Tm_K, 1e-2)["power-law creep (lattice diffusion)"]
    assert abs(np.log(r[1] / r[0]) / np.log(2) - NI.n) < 1e-6


def test_diffusional_flow_is_linear_in_stress():
    s = np.array([1e-6, 2e-6])
    r = rates(NI, s, 0.9 * NI.Tm_K, 1e-4)["diffusional flow (Nabarro-Herring)"]
    assert abs(r[1] / r[0] - 2.0) < 1e-9
