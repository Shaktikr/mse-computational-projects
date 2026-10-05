import pytest

from heaml.composition import normalized_key, parse_formula
from heaml.data import phase_class
from heaml.descriptors import compute
from heaml.rules import guo_vec_rule


def test_parse_formula():
    c = parse_formula("Al0.5 Co1 Cr1 Fe1 Ni1")
    assert c["Al"] == pytest.approx(0.5 / 4.5)
    assert parse_formula("AlCoCrFeNi") == pytest.approx({e: 0.2 for e in ["Al", "Co", "Cr", "Fe", "Ni"]})
    assert normalized_key("Co1 Cr1 Ni1") == normalized_key("CrNiCo")


def test_descriptors_cantor_family():
    d = compute(parse_formula("CoCrFeNi"))
    assert d["dHmix"] == pytest.approx(-3.75)  # Takeuchi-Inoue value
    assert d["VEC"] == pytest.approx(8.25)
    assert d["dSmix"] == pytest.approx(8.314462618 * 1.3862944, rel=1e-6)  # R ln 4
    d5 = compute(parse_formula("AlCoCrFeNi"))
    assert d5["dHmix"] == pytest.approx(-12.32)
    assert d5["VEC"] == pytest.approx(7.2)


def test_phase_labels_and_rules():
    assert phase_class("FCC") == "FCC"
    assert phase_class("BCC+B2") == "BCC"
    assert phase_class("FCC+B2") == "FCC+BCC"
    assert phase_class("BCC+Laves") == "IM"
    assert phase_class("Other") is None
    assert list(guo_vec_rule([8.3, 7.2, 5.0])) == ["FCC", "FCC+BCC", "BCC"]
