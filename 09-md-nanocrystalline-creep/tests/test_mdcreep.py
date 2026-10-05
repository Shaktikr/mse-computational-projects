import numpy as np
import pytest
from scipy.spatial import cKDTree

from mdcreep.analysis import stress_exponent
from mdcreep.polycrystal import random_rotation, voronoi_polycrystal


def test_random_rotation_is_orthonormal():
    R = random_rotation(np.random.default_rng(0))
    assert np.allclose(R @ R.T, np.eye(3)) and np.linalg.det(R) == pytest.approx(1.0)


@pytest.mark.parametrize("lattice,symbols,a", [("fcc", ["Ni"], 3.52), ("b2", ["Co", "Al"], 2.86)])
def test_polycrystal_density_and_spacing(lattice, symbols, a):
    at, gid = voronoi_polycrystal(symbols, a, 30.0, 3, lattice, seed=2)
    n_perfect = (4 if lattice == "fcc" else 2) / a**3 * 30.0**3
    assert 0.9 < len(at) / n_perfect < 1.01  # a few % excess volume at boundaries
    d, _ = cKDTree(at.positions, boxsize=30.0).query(at.positions, k=2)
    nn = a / np.sqrt(2) if lattice == "fcc" else a * np.sqrt(3) / 2
    assert d[:, 1].min() > 0.69 * nn  # no overlapping atoms
    assert len(np.unique(gid)) == 3


def test_stress_exponent():
    s = np.array([0.2, 0.4, 0.8])
    assert stress_exponent(s, 3e7 * s**2.0)[0] == pytest.approx(2.0)


def test_lammps_available_or_skip():
    pytest.importorskip("lammps")
