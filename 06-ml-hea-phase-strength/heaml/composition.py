"""Composition parsing."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)")


def parse_formula(formula: str) -> dict[str, float]:
    """'Al0.5 Co1 Cr1 Fe1 Ni1' or 'Al0.5CoCrFeNi' -> {'Al': 0.111, 'Co': 0.222, ...} (atomic fractions)."""
    amounts: dict[str, float] = {}
    for el, num in _TOKEN.findall(formula.replace(" ", "")):
        amounts[el] = amounts.get(el, 0.0) + (float(num) if num else 1.0)
    total = sum(amounts.values())
    return {el: v / total for el, v in amounts.items() if v > 0}


def normalized_key(formula: str, ndigits: int = 3) -> str:
    """Canonical string for grouping identical compositions written differently."""
    c = parse_formula(formula)
    return " ".join(f"{el}{round(c[el], ndigits)}" for el in sorted(c))


def alloy_formula(**amounts) -> str:
    """alloy_formula(Al=0.5, Co=1, Cr=1, Fe=1, Ni=1) -> 'Al0.5 Co1 Cr1 Fe1 Ni1'."""
    return " ".join(f"{k}{v:g}" for k, v in amounts.items() if v > 0)
