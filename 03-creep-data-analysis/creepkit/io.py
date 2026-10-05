"""Read/write creep curves as CSV with a commented metadata header.

File format (see data/template_creep_test.csv):

    # specimen_id: S1
    # stress_MPa: 150
    # temperature_C: 700
    # ruptured: true
    # gauge_length_mm: 25          (optional - only needed for displacement data)
    time_h,strain                  (or time_h,displacement_mm)
    0.0,0.0021
    ...

Any extra '# key: value' lines are kept in CreepTest.metadata.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .curves import CreepTest


def load_test(path: str | Path) -> CreepTest:
    path = Path(path)
    meta = {}
    with path.open() as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            key, _, value = line[1:].partition(":")
            meta[key.strip()] = value.strip()
    df = pd.read_csv(path, comment="#")
    time_h = df["time_h"].to_numpy(float)
    if "strain" in df:
        strain = df["strain"].to_numpy(float)
    elif "displacement_mm" in df:
        strain = df["displacement_mm"].to_numpy(float) / float(meta["gauge_length_mm"])
    else:
        raise ValueError(f"{path}: need a 'strain' or 'displacement_mm' column")
    return CreepTest(
        time_h=time_h, strain=strain,
        stress_MPa=float(meta.pop("stress_MPa")),
        temperature_C=float(meta.pop("temperature_C")),
        specimen_id=meta.pop("specimen_id", path.stem),
        ruptured=meta.pop("ruptured", "true").lower() in ("true", "1", "yes"),
        metadata=meta,
    )


def save_test(test: CreepTest, path: str | Path) -> None:
    path = Path(path)
    lines = [
        f"# specimen_id: {test.specimen_id}",
        f"# stress_MPa: {test.stress_MPa}",
        f"# temperature_C: {test.temperature_C}",
        f"# ruptured: {str(test.ruptured).lower()}",
    ] + [f"# {k}: {v}" for k, v in test.metadata.items()]
    df = pd.DataFrame({"time_h": test.time_h, "strain": test.strain})
    path.write_text("\n".join(lines) + "\n" + df.to_csv(index=False, float_format="%.6g"))


def load_folder(folder: str | Path, pattern: str = "*.csv"):
    return [load_test(p) for p in sorted(Path(folder).glob(pattern)) if not p.name.startswith("template")]
