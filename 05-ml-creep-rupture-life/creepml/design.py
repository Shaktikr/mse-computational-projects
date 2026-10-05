"""Inverse alloy design with a Gaussian-process surrogate (Bayesian-optimisation style).

Goal: find compositions with the longest predicted rupture life at a target condition
(e.g. 1000 C / 200 MPa) while staying inside the domain the model was trained on.

Acquisition = upper confidence bound  UCB = mu + kappa * sigma   (kappa > 0 rewards
exploration of uncertain regions; kappa = 0 is pure exploitation). Candidates are sampled
inside the per-element ranges of the training data and rejected if the Ni balance falls
below 45 wt.%. The top candidates are what you would melt / sinter and test next.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import COMPOSITION
from .features import add_physics_features


def sample_candidates(df_train: pd.DataFrame, n: int, rng, single_crystal: int = 1):
    lo = df_train[COMPOSITION].min()
    hi = df_train[COMPOSITION].max()
    rows = []
    while len(rows) < n:
        c = {k: rng.uniform(lo[k], hi[k]) for k in COMPOSITION}
        if 100 - sum(c.values()) < 45:
            continue
        c["single_crystal"] = single_crystal
        c["solution_T_C"] = float(rng.uniform(df_train["solution_T_C"].min(), df_train["solution_T_C"].max()))
        c["aging_T_C"] = float(rng.uniform(df_train["aging_T_C"].min(), df_train["aging_T_C"].max()))
        rows.append(c)
    return pd.DataFrame(rows)


def ucb_search(gp_pipeline, columns, df_train, T_C: float, stress_MPa: float, n_candidates: int = 20000,
               kappa: float = 1.0, top: int = 10, seed: int = 0, single_crystal: int = 1):
    rng = np.random.default_rng(seed)
    cand = sample_candidates(df_train, n_candidates, rng, single_crystal)
    cand["T_C"] = T_C
    cand["stress_MPa"] = stress_MPa
    cand = add_physics_features(cand)
    X = cand[columns].to_numpy(float)
    scaler, gp = gp_pipeline[-1].steps[0][1], gp_pipeline[-1].steps[1][1]
    mu, sd = gp.predict(scaler.transform(X), return_std=True)
    cand["pred_log10_life"] = mu
    cand["pred_std"] = sd
    cand["UCB"] = mu + kappa * sd
    return cand.sort_values("UCB", ascending=False).head(top).reset_index(drop=True)
