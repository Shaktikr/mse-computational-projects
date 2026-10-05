"""Physically motivated descriptors of multi-principal-element alloys.

For atomic fractions c_i:

  delta   atomic size mismatch  100 * sqrt( sum c_i (1 - r_i / r_mean)^2 )   [%]
            (Zhang et al., Adv. Eng. Mater. 10, 534 (2008))
  dHmix   mixing enthalpy  sum_{i<j} 4 dH_ij c_i c_j   [kJ/mol]
            dH_ij = binary liquid mixing enthalpies, Miedema model as tabulated by
            Takeuchi & Inoue, Mater. Trans. 46, 2817 (2005)  (via matminer)
  dSmix   ideal configurational entropy  -R sum c_i ln c_i   [J/mol K]
  Tm      rule-of-mixtures melting point   [K]
  Omega   Tm dSmix / |dHmix|   (Yang & Zhang, Mater. Chem. Phys. 132, 233 (2012))
  VEC     valence-electron concentration  sum c_i VEC_i  (Guo et al., J. Appl. Phys. 109, 103505 (2011))
  dchi    Pauling electronegativity difference  sqrt( sum c_i (chi_i - chi_mean)^2 )
  Lambda  dSmix / delta^2   (Singh et al., Acta Mater. 62, 105 (2014))
  gamma   solid-angle packing parameter (Wang et al., Scripta Mater. 94, 28 (2015))

Elemental radii are 12-coordinate metallic radii (pymatgen); for the few non-metals
without one the Takeuchi & Inoue / Goldschmidt values are used (B 0.82, C 0.77, Si 1.15,
Sn 1.58 A).
"""

from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np
import pandas as pd

from .composition import parse_formula

R = 8.314462618
RADIUS_FALLBACK = {"B": 0.82, "C": 0.77, "Si": 1.15, "Sn": 1.58}

DESCRIPTORS = ["delta", "dHmix", "dSmix", "Tm", "Omega", "VEC", "dchi", "Lambda", "gamma", "n_elements"]


@lru_cache(maxsize=None)
def element_data(symbol: str) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from pymatgen.core import Element

        e = Element(symbol)
        r = float(e.metallic_radius) if e.metallic_radius else RADIUS_FALLBACK[symbol]
        group = int(e.group)
        vec = group if group <= 12 else group - 10
        return {"r": r, "chi": float(e.X), "Tm": float(e.melting_point), "VEC": vec}


@lru_cache(maxsize=None)
def mixing_enthalpy(a: str, b: str) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from matminer.utils.data import MixingEnthalpy
        from pymatgen.core import Element

        v = _MIX().get_mixing_enthalpy(Element(a), Element(b))
        return float(v) if v is not None and v == v else 0.0


@lru_cache(maxsize=1)
def _MIX():
    from matminer.utils.data import MixingEnthalpy

    return MixingEnthalpy()


def compute(composition: dict[str, float]) -> dict[str, float]:
    els = list(composition)
    c = np.array([composition[e] for e in els], float)
    c = c / c.sum()
    d = [element_data(e) for e in els]
    r = np.array([x["r"] for x in d])
    chi = np.array([x["chi"] for x in d])
    Tm_i = np.array([x["Tm"] for x in d])
    vec_i = np.array([x["VEC"] for x in d])

    r_mean = np.sum(c * r)
    delta = 100.0 * np.sqrt(np.sum(c * (1 - r / r_mean) ** 2))
    dH = sum(4.0 * mixing_enthalpy(els[i], els[j]) * c[i] * c[j]
             for i in range(len(els)) for j in range(i + 1, len(els)))
    dS = -R * np.sum(c * np.log(c))
    Tm = np.sum(c * Tm_i)
    omega = Tm * dS / max(abs(dH) * 1000.0, 1e-3)
    chi_mean = np.sum(c * chi)
    dchi = np.sqrt(np.sum(c * (chi - chi_mean) ** 2))

    def omega_solid(rx):
        return 1.0 - np.sqrt(((rx + r_mean) ** 2 - r_mean**2) / (rx + r_mean) ** 2)

    gamma = omega_solid(r.min()) / omega_solid(r.max())
    return {
        "delta": delta, "dHmix": dH, "dSmix": dS, "Tm": Tm, "Omega": min(omega, 100.0),
        "VEC": np.sum(c * vec_i), "dchi": dchi, "Lambda": dS / max(delta, 1e-3) ** 2 if delta > 1e-3 else 100.0,
        "gamma": gamma, "n_elements": len(els),
    }


def featurize(formulas, elements_as_features: list[str] | None = None) -> pd.DataFrame:
    """Descriptor table for a list of formulas (optionally + atomic fractions of given elements)."""
    rows = []
    for f in formulas:
        comp = parse_formula(f)
        row = compute(comp)
        for el in elements_as_features or []:
            row[f"x_{el}"] = comp.get(el, 0.0)
        rows.append(row)
    return pd.DataFrame(rows)
