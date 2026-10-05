import numpy as np
import pytest

import vacdiff  # noqa: F401  (adds project 01 to the path)
from vacdiff import creep as cr
from vacdiff.defects import Settings, fcc_supercell, nearest_neighbour, vacancy_formation
from vacdiff.diffusion import F_FCC, arrhenius_parameters, debye_frequency, self_diffusion_fcc
from vacdiff.neb import final_state, run_neb


def test_vacancy_formation_emt_ni():
    s = Settings(calc="emt", fmax=0.01, workdir="/tmp")
    perfect = fcc_supercell("Ni", 3.4868, 2)
    res, relaxed = vacancy_formation(perfect, s)
    assert res.n_sites == 32
    assert 1.5 < res.Ef_relaxed_eV < 2.2  # EMT gives ~1.9 eV
    assert res.Ef_relaxed_eV <= res.Ef_unrelaxed_eV  # relaxation lowers the energy


def test_neb_symmetric_barrier_emt():
    s = Settings(calc="emt", fmax=0.01, workdir="/tmp")
    from vacdiff.defects import relax_positions, remove_atom
    perfect = fcc_supercell("Ni", 3.4868, 2)
    init, _ = relax_positions(remove_atom(perfect, 0), s, "i")
    j = nearest_neighbour(perfect, 0)
    fin, _ = relax_positions(final_state(perfect, j), s, "f")
    res = run_neb(init, fin, s, n_images=3, fmax=0.05)
    e = np.array(res.energies_eV)
    assert res.saddle_index == 2  # saddle in the middle for the symmetric fcc jump
    assert abs(e[-1]) < 0.02  # initial and final states are equivalent
    assert 0.8 < res.Em_eV < 1.5


def test_diffusion_formula():
    nu = debye_frequency(400.0)
    D0, Q, QkJ = arrhenius_parameters(4.05, 0.7, 0.6, nu)
    assert D0 == pytest.approx(F_FCC * (4.05e-10) ** 2 * nu)
    assert Q == pytest.approx(1.3) and QkJ == pytest.approx(1.3 * 96.485)
    assert self_diffusion_fcc(800.0, 4.05, 0.7, 0.6, nu) == pytest.approx(D0 * np.exp(-1.3 / (8.617333262e-5 * 800.0)))


def test_coble_beats_nabarro_herring_below_crossover():
    T, Om = 700.0, 1.66e-29
    DL = 1e-16
    dDb = 1e-23
    d_star = (cr.A_COBLE / cr.A_NH) * dDb / DL
    for d, coble_wins in [(0.1 * d_star, True), (10 * d_star, False)]:
        assert (cr.coble(10, T, d, dDb, Om) > cr.nabarro_herring(10, T, d, DL, Om)) == coble_wins
