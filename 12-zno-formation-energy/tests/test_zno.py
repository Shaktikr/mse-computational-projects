"""Tests for project 12 that run without Quantum ESPRESSO.

Structures: the three cells sent to pw.x have the right geometry; inputs: the writer produces
an insulator, a metal and a spin-polarised molecule correctly; parser: energy, pressure and the
relaxed structure are read from pw.x-style output; results: the committed numbers are consistent
with each other; hand-written inputs: no "!" comment inside a card (pw.x 7.5 rejects it).
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load(name, alias):
    # loaded by path: projects 10 and 11 also have a module called qe.py
    spec = importlib.util.spec_from_file_location(alias, ROOT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


qe = _load("qe", "qe_project12")


def _distances(cell, frac, i, others):
    """Sorted distances (Angstrom) from atom i to the atoms in `others`, periodic images included."""
    cell, frac = np.array(cell, float), np.array(frac, float)
    d = []
    for j in others:
        for n in np.ndindex(3, 3, 3):
            r = np.linalg.norm((frac[j] + np.array(n) - 1 - frac[i]) @ cell)
            if r > 1e-8:
                d.append(r)
    return np.sort(d)


def test_wurtzite_zno_geometry():
    cell, sym, frac = qe.wurtzite_zno()
    assert sym == ["Zn", "Zn", "O", "O"]
    assert np.isclose(abs(np.linalg.det(cell)), np.sqrt(3) / 2 * 3.250**2 * 5.207)
    d = _distances(cell, frac, 0, [2, 3])           # Zn-O distances
    assert np.isclose(d[3], 0.382 * 5.207)          # the bond along c is the longest of the four
    assert np.all((d[:4] > 1.96) & (d[:4] < 2.00))  # tetrahedral coordination, bonds of about 1.98 A
    assert d[4] > 3.0                               # and nothing else nearby


def test_hcp_zn_geometry():
    cell, sym, frac = qe.hcp_zn()
    assert sym == ["Zn", "Zn"]
    assert np.isclose(np.linalg.norm(cell[2]) / np.linalg.norm(cell[0]), 4.947 / 2.665)
    d = _distances(cell, frac, 0, [0, 1])
    assert np.allclose(d[:6], 2.665)                                    # six neighbours in the plane
    assert np.allclose(d[6:12], np.sqrt(2.665**2 / 3 + 4.947**2 / 4))   # six in the planes above and below


def test_o2_molecule_is_centred_with_the_right_bond():
    for box in (8, 10, 12):
        cell, sym, frac = qe.o2_molecule(box)
        pos = np.array(frac) @ np.array(cell)
        assert sym == ["O", "O"]
        assert np.isclose(np.linalg.norm(pos[1] - pos[0]), 1.21)
        assert np.allclose(pos.mean(axis=0), np.diag(cell) / 2)


def test_input_writer(tmp_path):
    f = str(tmp_path / "zno.in")
    qe.write_pw_input(f, *qe.wurtzite_zno(), ecut=50, kpts=(6, 6, 4))
    txt = Path(f).read_text()
    assert "calculation = 'scf'" in txt and "occupations = 'fixed'" in txt
    assert "nat = 4" in txt and "ntyp = 2" in txt
    assert "ecutwfc = 50" in txt and "ecutrho = 400" in txt       # 8 x ecutwfc for PAW
    assert "K_POINTS automatic" in txt and "6 6 4 0 0 0" in txt
    assert "Zn.pbe-dn-kjpaw_psl.0.2.2.UPF" in txt and "O.pbe-n-kjpaw_psl.0.1.UPF" in txt
    assert "nspin" not in txt and "smearing" not in txt and "&IONS" not in txt

    f = str(tmp_path / "zn.in")
    qe.write_pw_input(f, *qe.hcp_zn(), kpts=(16, 16, 8), metal=True, calc="vc-relax")
    txt = Path(f).read_text()
    assert "occupations = 'smearing'" in txt and "smearing = 'mv'" in txt and "degauss = 0.02" in txt
    assert "&IONS" in txt and "&CELL" in txt and "press_conv_thr" in txt

    f = str(tmp_path / "o2.in")
    qe.write_pw_input(f, *qe.o2_molecule(10), kpts=None, tot_mag=2, calc="relax")
    txt = Path(f).read_text()
    assert "nspin = 2" in txt and "tot_magnetization = 2" in txt
    assert "starting_magnetization(1) = 0.5" in txt
    assert "K_POINTS gamma" in txt and "&IONS" in txt and "&CELL" not in txt


def test_finished_run_is_not_overwritten(tmp_path):
    (tmp_path / "done.out").write_text("...\n   JOB DONE.\n")
    qe.write_pw_input(str(tmp_path / "done.in"), *qe.hcp_zn(), metal=True)
    assert not (tmp_path / "done.in").exists()      # the input of a finished run stays its record


def test_parse_output(tmp_path):
    out = tmp_path / "x.out"
    out.write_text("""
!    total energy              =    -571.13979829 Ry
     total magnetization       =     2.00 Bohr mag/cell
     Total force =     0.000123     Total SCF correction =     0.000001
          total   stress  (Ry/bohr**3)                   (kbar)     P=        2.58
!    total energy              =    -571.14165872 Ry
Begin final coordinates
CELL_PARAMETERS (angstrom)
   3.278580741   0.000000000   0.000000000
  -1.639290370   2.839334210   0.000000000
  -0.000000000  -0.000000000   5.286280828

ATOMIC_POSITIONS (crystal)
Zn               0.3333333333        0.6666666667        0.0013029674
Zn               0.6666666667        0.3333333333        0.5013029674
O                0.3333333333        0.6666666667        0.3806970326
O                0.6666666667        0.3333333333        0.8806970326
End final coordinates
   JOB DONE.
""")
    r = qe.parse(str(out))
    assert r["done"]
    assert np.isclose(r["energy_eV"], -571.14165872 * qe.RY)     # the LAST "!" line
    assert np.isclose(r["pressure_kbar"], 2.58) and np.isclose(r["magnetization"], 2.0)
    assert np.isclose(r["cell"][0][0], 3.278580741) and np.isclose(r["cell"][2][2], 5.286280828)
    assert len(r["frac"]) == 4 and np.isclose(r["frac"][2][2] - r["frac"][0][2], 0.3793940652)


def test_committed_results_are_consistent():
    relax = json.loads((ROOT / "results_relax.json").read_text())
    final = json.loads((ROOT / "results_formation_energy.json").read_text())
    conv = json.loads((ROOT / "results_convergence.json").read_text())
    dh = (relax["ZnO"]["E_cell_eV"] / relax["ZnO"]["n_formula_units"]
          - relax["Zn"]["E_cell_eV"] / relax["Zn"]["n_atoms"]
          - relax["O2"]["E_molecule_eV"] / 2)
    assert np.isclose(dh, final["dHf_eV_per_fu"], atol=1e-6)
    assert -2.95 < dh < -2.80                                    # the PBE range for ZnO
    assert abs(relax["ZnO"]["a"] / 3.250 - 1) < 0.02 and abs(relax["ZnO"]["c"] / 5.207 - 1) < 0.02
    assert np.isclose(relax["O2"]["magnetization"], 2.0)         # the triplet state
    e = [conv["ecut"][k] for k in ("50", "60", "70")]
    assert max(e) - min(e) < 0.02                                # converged to about 0.01 eV


def test_hand_written_inputs_follow_the_comment_rule():
    files = sorted((ROOT / "inputs").glob("*.in"))
    assert len(files) == 8
    for f in files:
        in_cards = False
        for line in f.read_text().splitlines():
            in_cards = in_cards or line.startswith("ATOMIC_SPECIES")
            assert not (in_cards and line.startswith("!")), f"{f.name}: '!' comment inside a card"
