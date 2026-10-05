"""Dataset schema, loading, and a SYNTHETIC Ni-base superalloy creep-rupture generator.

REAL DATA
---------
Put a CSV with the columns in `COMPOSITION + PROCESSING + TEST + [TARGET, "alloy_id"]` in
data/ and load it with `load_csv`. Good public sources of real creep-rupture data:
  * NIMS Creep Data Sheets (MatNavi, https://mits.nims.go.jp) - steels and superalloys
  * Supplementary data of e.g. Y. Liu et al., Acta Mater. 195 (2020) 454 (Ni-base single
    crystals) or J. Wang et al., Mater. Des. (2021) - check each licence.
Composition in wt.%, temperatures in C, stress in MPa, life in hours.

SYNTHETIC DATA
--------------
`generate_superalloy_dataset` creates a realistic-looking but ARTIFICIAL dataset so the
whole pipeline can run and be tested with known ground truth. The hidden "physics":

  gamma-prime former index  g  = Al + 0.5 Ti + 0.25 Ta + 0.3 Nb        (wt.%)
  gamma' fraction           f' = clip(0.075 g, 0.05, 0.75)
  solid-solution index      ss = 0.6 W + 0.7 Mo + 1.6 Re + 0.08 Co - 0.04 Cr
  TCP penalty                    if Re + 0.5 W + 0.3 Cr + 0.5 Mo > 15 (sigma / mu phase)
  gamma' solvus             Ts = 1150 + 25 (g - 5.5) + 8 Ta  (C)  - strength collapses near Ts
  grain-boundary elements        C, B, Hf only help polycrystals
  single-crystal bonus
  Larson-Miller master curve LMP = T[K] (20 + log10 t_r) / 1000
      LMP = 27.6 + dP(comp, T) - 1.8 ln(sigma/100) - 0.17 ln(sigma/100)^2
  heat-to-heat scatter (per alloy) + test scatter (per test)

The interaction between temperature and composition (solvus effect) and the non-monotonic
TCP penalty are deliberately included so that a simple Larson-Miller fit cannot capture
everything - exactly the situation in which ML helps.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COMPOSITION = ["Cr", "Co", "Mo", "W", "Re", "Al", "Ti", "Ta", "Nb", "Hf", "C", "B"]  # wt.%, Ni balance
PROCESSING = ["single_crystal", "solution_T_C", "aging_T_C"]
TEST = ["T_C", "stress_MPa"]
TARGET = "log10_rupture_h"
GROUP = "alloy_id"

RANGES = {  # wt.% ranges of the synthetic design space (typical of cast superalloys)
    "Cr": (2.0, 16.0), "Co": (0.0, 15.0), "Mo": (0.0, 5.0), "W": (0.0, 10.0), "Re": (0.0, 6.0),
    "Al": (3.0, 6.5), "Ti": (0.0, 4.5), "Ta": (0.0, 10.0), "Nb": (0.0, 2.0), "Hf": (0.0, 1.5),
    "C": (0.0, 0.17), "B": (0.0, 0.02),
}


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in COMPOSITION + PROCESSING + TEST + [TARGET, GROUP] if c not in df]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df


# --------------------------------------------------------------------------------------
# hidden synthetic "physics"
# --------------------------------------------------------------------------------------

def _gp_index(c):
    return c["Al"] + 0.5 * c["Ti"] + 0.25 * c["Ta"] + 0.3 * c["Nb"]


def solvus_C(c):
    return 1150.0 + 25.0 * (_gp_index(c) - 5.5) + 8.0 * c["Ta"]


def lmp_true(c, T_C, sigma):
    """Hidden Larson-Miller parameter (C = 20) for composition dict/row c."""
    g = _gp_index(c)
    fgp = np.clip(0.075 * g, 0.05, 0.75)
    ss = 0.6 * c["W"] + 0.7 * c["Mo"] + 1.6 * c["Re"] + 0.08 * c["Co"] - 0.04 * c["Cr"]
    tcp = np.maximum(c["Re"] + 0.5 * c["W"] + 0.3 * c["Cr"] + 0.5 * c["Mo"] - 15.0, 0.0)
    sx = c["single_crystal"]
    gb = (1 - sx) * (4.0 * c["C"] + 40.0 * c["B"] + 0.4 * c["Hf"])
    # under-solutioned alloys (solution T below solvus) do not dissolve coarse gamma'
    heat = -0.004 * np.maximum(solvus_C(c) + 20.0 - c["solution_T_C"], 0.0)
    aging = -2e-5 * (c["aging_T_C"] - 1000.0) ** 2
    dP = (2.4 * np.tanh((fgp - 0.45) / 0.18) + 0.22 * ss - 0.12 * tcp**2 + 0.9 * sx + gb + heat + aging - 0.6)
    # strength collapses as T approaches the gamma' solvus (T-composition interaction)
    dP = dP - 2.0 / (1.0 + np.exp(-(T_C - (solvus_C(c) - 60.0)) / 25.0))
    x = np.log(sigma / 100.0)
    return 27.6 + dP - 1.8 * x - 0.17 * x**2


def life_hours_true(c, T_C, sigma):
    return 10 ** (1000.0 * lmp_true(c, T_C, sigma) / (T_C + 273.15) - 20.0)


def _random_alloy(rng):
    while True:
        c = {k: rng.uniform(*v) for k, v in RANGES.items()}
        for k in ("Re", "Hf", "Nb", "Ta", "Ti", "Mo", "Co", "B", "C"):  # many alloys omit some elements
            if rng.random() < 0.3:
                c[k] = 0.0
        c["single_crystal"] = int(rng.random() < 0.45)
        if c["single_crystal"]:
            c["C"], c["B"], c["Hf"] = c["C"] * 0.1, 0.0, c["Hf"] * 0.1
        if 100 - sum(c[k] for k in COMPOSITION) < 45:  # Ni balance must stay > 45 wt.%
            continue
        c["solution_T_C"] = float(np.clip(solvus_C(c) + rng.normal(10, 30), 1150, 1340))
        c["aging_T_C"] = float(rng.uniform(850, 1100))
        return c


def generate_superalloy_dataset(n_alloys: int = 120, seed: int = 7) -> pd.DataFrame:
    """SYNTHETIC creep-rupture dataset (see module docstring)."""
    rng = np.random.default_rng(seed)
    rows = []
    for a in range(n_alloys):
        c = _random_alloy(rng)
        heat_effect = rng.normal(0.0, 0.08)  # heat-to-heat variation, in log10 t_r
        temps = rng.choice(np.arange(700, 1101, 50), size=rng.integers(4, 8), replace=False)
        for T in temps:
            for target_life in rng.choice([30, 100, 300, 1000, 3000, 10000], size=rng.integers(1, 4), replace=False):
                # choose the stress the experimentalist would pick to get ~target life
                s_grid = np.geomspace(40, 1300, 300)
                L = life_hours_true(c, T, s_grid)
                s = float(np.interp(np.log10(target_life), np.log10(L[::-1]), s_grid[::-1]))
                s = float(np.clip(round(s * rng.uniform(0.9, 1.1), -1), 40.0, 1300.0))
                logt = np.log10(life_hours_true(c, T, s)) + heat_effect + rng.normal(0.0, 0.12)
                if 0.5 <= logt <= 5.0:
                    rows.append({GROUP: f"SYN-{a:03d}", **{k: round(c[k], 3) for k in COMPOSITION},
                                 "single_crystal": c["single_crystal"], "solution_T_C": round(c["solution_T_C"]),
                                 "aging_T_C": round(c["aging_T_C"]), "T_C": int(T), "stress_MPa": s,
                                 TARGET: round(float(logt), 4)})
    df = pd.DataFrame(rows)
    df.insert(1, "Ni", (100 - df[COMPOSITION].sum(axis=1)).round(3))
    return df
