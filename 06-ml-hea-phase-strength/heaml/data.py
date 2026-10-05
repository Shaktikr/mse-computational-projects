"""Load and clean the MPEA dataset.

Source: C.K.H. Borg, C. Frey, J. Moh, T.M. Pollock, S. Gorsse, D.B. Miracle, O.N. Senkov,
B. Meredig, J.E. Saal, "Expanded dataset of mechanical properties and observed phases of
multi-principal element alloys", Scientific Data 7, 430 (2020).
Data: https://github.com/CitrineInformatics/MPEA_dataset (Apache-2.0),
      https://doi.org/10.6084/m9.figshare.12642953

Phase classes used here (from the 'Microstructure' column):
  FCC       single FCC solid solution
  BCC       BCC and/or ordered B2 (no FCC, no intermetallic)
  FCC+BCC   FCC together with BCC/B2
  IM        contains an intermetallic: Laves, L12, sigma/mu ('Sec.')
Rows labelled 'Other' or containing HCP are dropped from the classification task.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .composition import normalized_key

DATA = Path(__file__).resolve().parents[1] / "data" / "MPEA_dataset.csv"

RENAME = {
    "FORMULA": "formula",
    "PROPERTY: Microstructure": "microstructure",
    "PROPERTY: Processing method": "processing",
    "PROPERTY: Type of test": "test_type",
    "PROPERTY: Test temperature ($^\\circ$C)": "test_T_C",
    "PROPERTY: YS (MPa)": "YS_MPa",
    "PROPERTY: UTS (MPa)": "UTS_MPa",
    "PROPERTY: Elongation (%)": "elongation",
    "PROPERTY: HV": "HV",
    "PROPERTY: grain size ($\\mu$m)": "grain_size_um",
    "REFERENCE: doi": "doi",
}


def phase_class(micro: str) -> str | None:
    if not isinstance(micro, str) or micro == "Other" or "HCP" in micro:
        return None
    parts = micro.split("+")
    if any(p in ("Laves", "Sec.", "L12") for p in parts):
        return "IM"
    has_fcc = "FCC" in parts
    has_bcc = any(p in ("BCC", "B2") for p in parts)
    if has_fcc and has_bcc:
        return "FCC+BCC"
    if has_fcc:
        return "FCC"
    if has_bcc:
        return "BCC"
    return None


def load(path: str | Path = DATA) -> pd.DataFrame:
    df = pd.read_csv(path).rename(columns=RENAME)
    df = df[list(RENAME.values())].copy()
    df["phase"] = df["microstructure"].map(phase_class)
    df["key"] = df["formula"].map(normalized_key)
    return df


def phase_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (composition, processing) with an unambiguous phase label."""
    d = df.dropna(subset=["phase"]).drop_duplicates(["key", "processing", "phase"])
    # drop (composition, processing) pairs reported with conflicting phases
    n = d.groupby(["key", "processing"])["phase"].transform("nunique")
    return d[n == 1].reset_index(drop=True)


def strength_table(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with a yield strength and a test temperature."""
    d = df.dropna(subset=["YS_MPa", "test_T_C"]).copy()
    d = d[(d["YS_MPa"] > 0) & d["test_type"].isin(["C", "T"])]
    return d.reset_index(drop=True)
