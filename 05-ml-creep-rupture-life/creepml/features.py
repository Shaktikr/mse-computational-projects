"""Physics-informed features for creep-rupture life.

Raw inputs (composition, processing, T, sigma) are augmented with the variables that
creep physics says matter, which makes models more data-efficient and extrapolate better:

  invT        1000/T[K]            - Arrhenius / Larson-Miller temperature dependence
  log_sigma   ln(sigma)            - power-law stress dependence
  log_sigma2  ln(sigma)^2          - curvature of the Larson-Miller master curve
  sig_invT    ln(sigma) * 1000/T   - stress-dependent activation energy
  gp_index    Al + 0.5Ti + 0.25Ta + 0.3Nb      gamma' former content (wt.%)
  refractory  W + Mo + Re + Ta                  solid-solution strengtheners
  Al_Ti       Al / (Ti + 0.1)                   gamma' chemistry
  dT_solution solution_T - aging_T

A composition-blind Larson-Miller fit (one master curve for all alloys) is used as a
physics BASELINE; ML can then learn only the residual (hybrid / "physics + ML" model).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import COMPOSITION, PROCESSING, TEST

DERIVED = ["invT", "log_sigma", "log_sigma2", "sig_invT", "gp_index", "refractory", "Al_Ti", "dT_solution"]


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    T = out["T_C"] + 273.15
    out["invT"] = 1000.0 / T
    out["log_sigma"] = np.log(out["stress_MPa"])
    out["log_sigma2"] = out["log_sigma"] ** 2
    out["sig_invT"] = out["log_sigma"] * out["invT"]
    out["gp_index"] = out["Al"] + 0.5 * out["Ti"] + 0.25 * out["Ta"] + 0.3 * out["Nb"]
    out["refractory"] = out["W"] + out["Mo"] + out["Re"] + out["Ta"]
    out["Al_Ti"] = out["Al"] / (out["Ti"] + 0.1)
    out["dT_solution"] = out["solution_T_C"] - out["aging_T_C"]
    return out


def feature_columns(physics: bool = True) -> list[str]:
    raw = COMPOSITION + PROCESSING + TEST
    return raw + DERIVED if physics else raw


class LarsonMillerBaseline:
    """One master curve for all alloys: log10 t_r = LMP(sigma)*1000/T - C, LMP quadratic in ln sigma.

    Fitted by linear least squares on [1/T, ln(s)/T, ln(s)^2/T, 1]:
        log10 t_r = (a0 + a1 ln s + a2 ln^2 s) * 1000/T - C
    """

    def fit(self, df: pd.DataFrame, y):
        X = self._design(df)
        self.coef_, *_ = np.linalg.lstsq(X, np.asarray(y, float), rcond=None)
        return self

    def predict(self, df: pd.DataFrame):
        return self._design(df) @ self.coef_

    @staticmethod
    def _design(df):
        invT = 1000.0 / (df["T_C"].to_numpy() + 273.15)
        ls = np.log(df["stress_MPa"].to_numpy())
        return np.column_stack([invT, ls * invT, ls**2 * invT, np.ones_like(invT)])

    @property
    def C(self):
        return -self.coef_[3]
