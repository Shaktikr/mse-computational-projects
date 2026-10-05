"""Shared helpers for the tutorial scripts (argument parsing, calculator factory, I/O)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dftlab import config  # noqa: E402
from dftlab.calculators import get_calculator, magnetic, set_initial_magnetization  # noqa: E402
from dftlab.structures import build  # noqa: E402

RESULTS = ROOT / "results"
RUNS = ROOT / "runs"  # raw pw.x input/output (git-ignored)


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--system", default="Al", choices=["Al", "Ni", "NiAl"])
    p.add_argument("--calc", default="qe", choices=["qe", "emt"],
                   help="'qe' = Quantum ESPRESSO DFT, 'emt' = fast toy potential for testing")
    p.add_argument("--ecutwfc", type=float, default=None, help="override the converged default (Ry)")
    p.add_argument("--kspacing", type=float, default=None, help="override k-point spacing (1/A)")
    p.add_argument("--reuse", action="store_true", help="reuse finished pw.x runs found in runs/")
    return p


def make_factory(args, step: str):
    """Return calc_factory(atoms, tag, **overrides) writing runs to runs/<system>/<step>/<tag>."""
    defaults = dict(config.DEFAULTS[args.system])
    if args.ecutwfc:
        defaults["ecutwfc"] = args.ecutwfc
    if args.kspacing:
        defaults["kspacing"] = args.kspacing

    def factory(atoms, tag, **overrides):
        kw = {**defaults, **overrides}
        spin = magnetic(atoms)
        if spin and args.calc == "qe":
            set_initial_magnetization(atoms)
        directory = RUNS / args.calc / args.system / step / tag
        if args.calc == "qe" and getattr(args, "reuse", False):
            cached = _reuse_finished_run(atoms, directory)
            if cached is not None:
                return cached
        return get_calculator(args.calc, atoms, directory, spin_polarized=spin, **kw)

    return factory


def _reuse_finished_run(atoms, directory: Path):
    """Return a SinglePointCalculator from a completed pw.x run in `directory`, if any.

    Lets an interrupted workflow restart without repeating finished calculations
    (use --reuse). The cell of the stored run must match the requested structure.
    """
    import numpy as np
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import read

    pwo = directory / "espresso.pwo"
    if not pwo.exists() or "JOB DONE" not in pwo.read_text(errors="ignore"):
        return None
    done = read(pwo, index=-1)
    if not np.allclose(done.cell[:], atoms.cell[:], atol=1e-4):
        return None
    print(f"  reusing {directory.relative_to(ROOT)}")
    return SinglePointCalculator(atoms, energy=done.get_potential_energy(), stress=done.get_stress())


def initial_structure(args):
    return build(args.system)


def out_dir(args) -> Path:
    sub = args.system if args.calc == "qe" else f"{args.system}_emt"
    d = RESULTS / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2))
    print(f"wrote {path.relative_to(ROOT)}")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())
