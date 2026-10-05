"""Tests for project 11 that run without Quantum ESPRESSO.

Geometry: the shear frame picks the twinning sense of <112>; linear elasticity: the rotated
stiffness gives the textbook {111}<112> moduli; algorithm: one quasi-Newton step of the stress
relaxation solves the linear-elastic problem exactly; results: the committed DFT data reproduce
the central finding of Ogata, Li & Yip (2002).
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parents[1]


def _load(name, alias):
    # loaded by path: project 10 also has a module called qe.py
    spec = importlib.util.spec_from_file_location(alias, ROOT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sg = _load("shear_geometry", "shear_geometry_project11")
qe = _load("qe", "qe_project11")

ELASTIC = {"Al": (113.6, 60.0, 33.0), "Cu": (170.7, 116.8, 80.2)}   # this work, PBE (GPa)


def _json(name):
    return json.loads((ROOT / name).read_text())


# ---------------------------------------------------------------- geometry
def test_frame_is_orthonormal_with_z_along_111():
    R, h0 = sg.twin_frame(4.0)
    assert np.allclose(R @ R.T, np.eye(3))
    assert np.allclose(R[2], np.ones(3) / np.sqrt(3))
    assert abs(abs(np.linalg.det(h0)) - 4.0 ** 3 / 4) < 1e-10


def test_twinning_sense_is_selected():
    a0 = 4.04
    R, _ = sg.twin_frame(a0)
    prim = sg.fcc_primitive_cubic(a0)
    assert sg.nn_spread(prim, R, 0.0) < 1e-9
    assert sg.nn_spread(prim, R, sg.GAMMA_TWIN) < 1e-9          # twin: fcc again
    assert sg.nn_spread(prim, R, -sg.GAMMA_TWIN) > 0.1           # anti-twinning sense: not fcc


# ---------------------------------------------------------------- linear elasticity
@pytest.mark.parametrize("C", [ELASTIC["Al"], ELASTIC["Cu"], (250.0, 150.0, 120.0)])
def test_shear_moduli_closed_forms(C):
    C11, C12, C44 = C
    g_unrel, g_rel = sg.shear_moduli(*C)
    assert abs(g_unrel - (C11 - C12 + C44) / 3) < 1e-9
    assert abs(g_rel - 3 * C44 * (C11 - C12) / (C11 - C12 + 4 * C44)) < 1e-9
    assert g_rel <= g_unrel + 1e-12


def test_isotropic_limit():
    lam, mu = 60.0, 30.0
    g_unrel, g_rel = sg.shear_moduli(lam + 2 * mu, lam, mu)
    assert abs(g_unrel - mu) < 1e-9 and abs(g_rel - mu) < 1e-9


def test_one_relaxation_step_solves_the_linear_problem():
    """Same update as 02_ideal_shear.py (D <- D - K^-1 sigma_other), with a linear-elastic 'DFT'."""
    C = ELASTIC["Cu"]
    R, _ = sg.twin_frame(3.63)
    Cr = sg.rotate(sg.cubic_stiffness(*C), R)
    K = sg.relaxation_stiffness(Cr)
    g = 1e-3

    def stress(d):
        F = np.eye(3)
        F[0, 2] = g
        for (i, j), v in zip(sg.COMPS, d):
            F[i, j] += v
            if i != j:
                F[j, i] += v
        eps = 0.5 * (F + F.T) - np.eye(3)
        return np.einsum("ijkl,kl->ij", Cr, eps)

    s0 = stress(np.zeros(5))
    d = -np.linalg.solve(K, np.array([s0[i, j] for i, j in sg.COMPS]))
    s1 = stress(d)
    assert max(abs(s1[i, j]) for i, j in sg.COMPS) < 1e-12
    assert abs(s1[0, 2] / g - sg.shear_moduli(*C)[1]) < 1e-6


def test_materials_have_pseudopotentials():
    for name, m in qe.MATERIALS.items():
        assert (ROOT / "pseudo" / m["pseudo"]).exists(), name


# ---------------------------------------------------------------- committed DFT results
FILES = {("Al", "relaxed"): "results_shear_Al_relaxed_k24_d0.04.json",
         ("Al", "unrelaxed"): "results_shear_Al_unrelaxed_k24_d0.04.json",
         ("Cu", "relaxed"): "results_shear_CuPAW_relaxed_k24.json",
         ("Cu", "unrelaxed"): "results_shear_CuPAW_unrelaxed_k24.json"}


def _tau_max(fn):
    d = _json(fn)
    g, t = np.array(d["gamma"]), np.array(d["tau"])
    o = np.argsort(g)
    g, t = g[o], t[o]
    if g[0] > 0:
        g, t = np.r_[0.0, g], np.r_[0.0, t]
    gf = np.linspace(0, g.max(), 2000)
    tf = CubicSpline(g, t)(gf)
    return tf.max(), gf[np.argmax(tf)], t.max()


def test_summary_matches_raw_curves():
    s = _json("results_summary.json")
    for (el, mode), fn in FILES.items():
        tmax, gm, tmax_points = _tau_max(fn)
        assert abs(tmax - s[f"{el}_{mode}"]["tau_max_GPa"]) < 1e-6
        assert abs(gm - s[f"{el}_{mode}"]["gamma_m"]) < 1e-6
        assert tmax_points <= tmax + 1e-9 and tmax - tmax_points < 0.05 * tmax


def test_relaxed_points_are_converged():
    for el in ("Al", "Cu"):
        d = _json(FILES[(el, "relaxed")])
        assert max(d["resid"]) < 0.05                              # GPa, the tolerance in 02_ideal_shear.py
        assert max(d["n_scf"]) < 12


def test_initial_slope_matches_linear_elasticity():
    s = _json("results_summary.json")
    for el in ("Al", "Cu"):
        g_unrel, g_rel = sg.shear_moduli(*ELASTIC[el])
        assert abs(s[f"{el}_relaxed"]["G_GPa"] - g_rel) / g_rel < 0.08
        assert abs(s[f"{el}_unrelaxed"]["G_GPa"] - g_unrel) / g_unrel < 0.08


def test_central_finding_of_ogata_li_yip():
    """Cu is stiffer than Al, yet Al is stronger: it sustains a larger strain before softening."""
    s = _json("results_summary.json")
    al, cu = s["Al_relaxed"], s["Cu_relaxed"]
    assert cu["G_GPa"] > al["G_GPa"]
    assert al["tau_max_GPa"] > cu["tau_max_GPa"]
    assert al["gamma_m"] > cu["gamma_m"]
    for el in ("Al", "Cu"):
        assert s[f"{el}_relaxed"]["tau_max_GPa"] < s[f"{el}_unrelaxed"]["tau_max_GPa"]


def test_stacking_fault_energies():
    sfe = _json("results_summary.json")["sfe"]
    assert sfe["Al"]["gamma_isf_direct"] > 100 > 60 > sfe["Cu"]["gamma_isf_direct_k20"]
    assert abs(sfe["Cu"]["gamma_isf_direct_k20"] - sfe["Cu"]["gamma_isf_annni_k32"]) < 5
    for el in ("Al", "Cu"):
        assert sfe[el]["gamma_us"] > 150
