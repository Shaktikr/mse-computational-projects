"""Tests for project 10 that run without Quantum ESPRESSO.

They check the pw.x input writer and output parser, the fcc cell builders, and that the committed
DFT results are internally consistent (EOS refit, elastic-constant identities, k-point convergence)
and agree with experiment within the expected PBE error.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]


def _load(name, alias):
    # loaded by path: project 11 also has a module called qe.py
    spec = importlib.util.spec_from_file_location(alias, ROOT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


qe = _load("qe", "qe_project10")


def _json(name):
    return json.loads((ROOT / name).read_text())


# ---------------------------------------------------------------- helper code
def test_fcc_primitive_cell():
    a = 4.04
    cell = qe.fcc_primitive(a)
    assert abs(abs(np.linalg.det(cell)) - a ** 3 / 4) < 1e-10
    nn = [np.linalg.norm(v) for v in cell]
    assert np.allclose(nn, a / np.sqrt(2))


def test_conventional_supercell_has_32_distinct_sites():
    a = 4.04
    cell, frac = qe.fcc_conventional_supercell(a, 2)
    assert frac.shape == (32, 3)
    assert np.allclose(cell, 2 * a * np.eye(3))
    assert len({tuple(np.round(p, 6)) for p in frac}) == 32
    cart = frac @ cell
    d = np.linalg.norm(cart[1:] - cart[0], axis=1)
    assert abs(d.min() - a / np.sqrt(2)) < 1e-9


def test_input_writer(tmp_path):
    cell, frac = qe.fcc_conventional_supercell(4.04, 2)
    fixed = [(0, 0, 0)] + [(1, 1, 1)] * 31
    inp = tmp_path / "vac.in"
    qe.write_pw_input(str(inp), cell, ["Al"] * 32, frac, calc="relax", ecut=40, kpts=(4, 4, 4),
                      fixed=fixed, nbnd=80)
    txt = inp.read_text()
    assert "calculation = 'relax'" in txt
    assert "nat = 32" in txt
    assert "ecutrho = 320" in txt                      # 8 x ecutwfc for ultrasoft pseudopotentials
    assert "nbnd = 80" in txt
    assert "&IONS" in txt
    assert "K_POINTS automatic\n  4 4 4 0 0 0" in txt
    pos = txt.split("ATOMIC_POSITIONS crystal\n")[1].split("K_POINTS")[0].strip().splitlines()
    assert len(pos) == 32
    assert pos[0].split()[-3:] == ["0", "0", "0"] and pos[1].split()[-3:] == ["1", "1", "1"]


def test_input_writer_omits_nbnd_by_default(tmp_path):
    inp = tmp_path / "bulk.in"
    qe.write_pw_input(str(inp), qe.fcc_primitive(4.04), ["Al"], [[0, 0, 0]])
    assert "nbnd" not in inp.read_text()


def test_finished_run_keeps_its_input(tmp_path):
    """Re-running a script must not rewrite the input of a finished calculation."""
    inp, out = tmp_path / "x.in", tmp_path / "x.out"
    qe.write_pw_input(str(inp), qe.fcc_primitive(4.04), ["Al"], [[0, 0, 0]], ecut=40)
    original = inp.read_text()
    out.write_text("... JOB DONE.\n")
    qe.write_pw_input(str(inp), qe.fcc_primitive(4.10), ["Al"], [[0, 0, 0]], ecut=60)
    assert inp.read_text() == original
    out.write_text("crashed\n")                        # unfinished: the input is rewritten
    qe.write_pw_input(str(inp), qe.fcc_primitive(4.10), ["Al"], [[0, 0, 0]], ecut=60)
    assert "ecutwfc = 60" in inp.read_text()


def test_parser_on_committed_output():
    out = ROOT / "runs" / "01_conv" / "ecut_40.out"
    if not out.exists():
        pytest.skip("pw.x outputs not present")
    r = qe.parse(str(out))
    assert r["done"] and r["converged_scf"]
    ref = {row[0]: row for row in _json("results_convergence.json")["ecut"]}[40]
    assert abs(r["energy_eV"] - ref[1]) < 1e-8
    assert abs(r["pressure_kbar"] - ref[2]) < 1e-6
    assert r["stress_kbar"].shape == (3, 3)


# ---------------------------------------------------------------- committed results
def _bm3(V, E0, V0, B0, Bp):
    eta = (V0 / V) ** (2 / 3)
    return E0 + 9 * V0 * B0 / 16 * ((eta - 1) ** 3 * Bp + (eta - 1) ** 2 * (6 - 4 * eta))


def test_eos_refit_reproduces_committed_values():
    b = _json("results_bulk.json")["k40"]
    V, E = np.array(b["ev_curve"]["V"]), np.array(b["ev_curve"]["E"])
    (E0, V0, B0, Bp), _ = curve_fit(_bm3, V, E, p0=[E.min(), V[np.argmin(E)], 0.5, 4.0])
    assert abs((4 * V0) ** (1 / 3) - b["a0_A"]) < 1e-4
    assert abs(B0 * 160.21766 - b["B0_GPa"]) < 0.1


def test_elastic_constant_identities_and_convergence():
    bulk = _json("results_bulk.json")
    c = bulk["k40"]
    assert abs((c["C11"] + 2 * c["C12"]) / 3 - c["B0_GPa"]) < 0.5      # B = (C11 + 2 C12)/3
    assert c["G_Reuss"] <= c["G_Hill"] <= c["G_Voigt"]
    # C44 converges slowly with k (Fermi surface of Al): k32 -> k40 still changes it by ~4 GPa,
    # but B and a0 are converged.
    assert abs(bulk["k40"]["B0_GPa"] - bulk["k32"]["B0_GPa"]) < 0.5
    assert abs(bulk["k40"]["a0_A"] - bulk["k32"]["a0_A"]) < 1e-3


def test_bulk_properties_against_experiment():
    c = _json("results_bulk.json")["k40"]
    assert abs(c["a0_A"] - 4.0496) / 4.0496 < 0.01        # experiment, 298 K
    for key, exp in (("C11", 114.3), ("C12", 61.9), ("C44", 31.6)):   # Kamm & Alers, 0 K
        assert abs(c[key] - exp) / exp < 0.15, key


def test_vacancy_energies_converged_and_physical():
    k = _json("results_kcheck.json")
    assert abs(k["k8"]["Ef_eV"] - k["k6"]["Ef_eV"]) < 0.03
    assert abs(k["k8"]["Em_eV"] - k["k6"]["Em_eV"]) < 0.01
    cr = _json("results_creep.json")
    assert abs(cr["Q_eV"] - (cr["Ef_eV"] + cr["Em_eV"])) < 1e-9
    assert 0.60 < cr["Ef_eV"] < 0.72                       # experiment 0.67 +/- 0.03 eV
    assert abs(cr["Q_eV"] - 1.28) / 1.28 < 0.10            # tracer-diffusion activation energy


def test_stacking_fault_energies():
    g = _json("results_gsfe_final_k16.json")
    ann = _json("results_isf_annni.json")
    assert g["relaxed_f0.60"] > g["relaxed_f1.00"] > 100   # gamma_us > gamma_isf
    assert abs(ann["k40"]["gamma_isf_mJm2"] - ann["k32"]["gamma_isf_mJm2"]) < 2
    # direct supercell value and the independent ANNNI estimate agree
    assert abs(g["rigid_f1.00"] - ann["k40"]["gamma_isf_mJm2"]) < 10
