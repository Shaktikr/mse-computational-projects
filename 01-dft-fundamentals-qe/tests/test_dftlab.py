import numpy as np
import pytest
from ase.calculators.emt import EMT

from dftlab.calculators import kgrid_from_spacing
from dftlab.config import EV_PER_A3_TO_GPA
from dftlab.convergence import first_converged
from dftlab.elastic import cubic_constants_from_fits, quadratic_coefficient, run_strain_series, strained
from dftlab.eos import birch_murnaghan, fit_birch_murnaghan, scan_volumes
from dftlab.structures import b2, build, fcc, lattice_parameter


def test_birch_murnaghan_fit_recovers_parameters():
    V = np.linspace(14, 19, 9)
    E = birch_murnaghan(V, -3.7, 16.5, 76 / EV_PER_A3_TO_GPA, 4.5)
    r = fit_birch_murnaghan(V, E)
    assert r.V0 == pytest.approx(16.5, rel=1e-6)
    assert r.B0_GPa == pytest.approx(76, rel=1e-4)
    assert r.B0_prime == pytest.approx(4.5, rel=1e-3)


def test_lattice_parameter_round_trip():
    for s, a in [("Al", 4.05), ("Ni", 3.52), ("NiAl", 2.887)]:
        at = build(s, a)
        assert lattice_parameter(s, at.get_volume()) == pytest.approx(a)


def test_kgrid_density_scales_with_cell():
    prim = fcc("Al", 4.05)
    sc = fcc("Al", 4.05, cubic=True).repeat((2, 2, 2))
    k1 = kgrid_from_spacing(prim, 0.15)
    k2 = kgrid_from_spacing(sc, 0.15)
    assert k1[0] > k2[0] and k2 == (6, 6, 6)


def test_first_converged():
    assert first_converged([20, 30, 40, 50], [0.010, 0.0012, 0.0004, 0.0], 1e-3) == 40


def test_energy_strain_method_matches_stress_strain_method():
    """Mehl's energy method and a finite-difference stress method must agree (EMT Ni)."""
    f = lambda at, tag: EMT()  # noqa: E731
    at = fcc("Ni", 3.52)
    V, E = scan_volumes(at, f, np.linspace(-0.02, 0.02, 9))
    eos = fit_birch_murnaghan(V, E)
    at = fcc("Ni", lattice_parameter("Ni", eos.V0))
    d, eo = run_strain_series(at, f, "ortho", np.linspace(-0.02, 0.02, 9))
    d, em = run_strain_series(at, f, "mono", np.linspace(-0.02, 0.02, 9))
    C = cubic_constants_from_fits(quadratic_coefficient(d, eo), quadratic_coefficient(d, em), eos.V0, eos.B0_GPa)
    h = 1e-4
    Cs = np.zeros((6, 6))
    for j in range(6):
        e = np.zeros(6)
        e[j] = h
        a1, a2 = strained(at, e), strained(at, -e)
        a1.calc, a2.calc = EMT(), EMT()
        Cs[:, j] = (a1.get_stress(voigt=True) - a2.get_stress(voigt=True)) / (2 * h) * EV_PER_A3_TO_GPA
    assert C.C11 == pytest.approx(Cs[0, 0], rel=0.01)
    assert C.C12 == pytest.approx(Cs[0, 1], rel=0.01)
    assert C.C44 == pytest.approx(Cs[5, 5], rel=0.01)
    assert C.born_stable()


def test_b2_structure():
    at = b2("Ni", "Al", 2.887)
    assert len(at) == 2 and sorted(at.get_chemical_symbols()) == ["Al", "Ni"]
