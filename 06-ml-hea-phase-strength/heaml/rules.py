"""Empirical phase-selection rules - the baselines any ML model must beat.

  Solid solution vs intermetallic (Yang & Zhang 2012):   Omega >= 1.1 and delta <= 6.6 %
  FCC vs BCC (Guo et al. 2011):  VEC >= 8.0 -> FCC;  VEC < 6.87 -> BCC;  in between FCC+BCC
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def guo_vec_rule(vec):
    vec = np.asarray(vec)
    return np.where(vec >= 8.0, "FCC", np.where(vec < 6.87, "BCC", "FCC+BCC"))


def yang_zhang_ss(omega, delta):
    return (np.asarray(omega) >= 1.1) & (np.asarray(delta) <= 6.6)


def combined_rule(df: pd.DataFrame):
    """4-class prediction: IM if Yang-Zhang says not solid solution, else Guo VEC rule."""
    ss = yang_zhang_ss(df["Omega"], df["delta"])
    return np.where(ss, guo_vec_rule(df["VEC"]), "IM")
