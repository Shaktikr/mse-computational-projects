"""Shared helpers for project 02 scripts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import vacdiff  # noqa: E402,F401  (puts project 01 on sys.path)
from vacdiff.defects import Settings  # noqa: E402

P01 = ROOT.parent / "01-dft-fundamentals-qe"


def parser(doc):
    p = argparse.ArgumentParser(description=doc)
    p.add_argument("--element", default="Al", choices=["Al", "Ni"])
    p.add_argument("--calc", default="qe", choices=["qe", "emt"])
    p.add_argument("--n", type=int, default=2, help="supercell repetitions of the cubic cell (n=2 -> 32 sites)")
    p.add_argument("--ecutwfc", type=float, default=None)
    p.add_argument("--kspacing", type=float, default=0.20)
    return p


def settings(args) -> Settings:
    ec = args.ecutwfc or {"Al": 35.0, "Ni": 45.0}[args.element]
    return Settings(calc=args.calc, ecutwfc=ec, kspacing=args.kspacing,
                    workdir=str(ROOT / "runs" / args.calc / f"{args.element}_{args.n}x{args.n}x{args.n}"))


def out_dir(args) -> Path:
    name = f"{args.element}_{args.n}x{args.n}x{args.n}" + ("" if args.calc == "qe" else "_emt")
    d = ROOT / "results" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def lattice_parameter(args) -> float:
    """DFT a0 from project 01 (falls back to an EMT relaxation when --calc emt)."""
    if args.calc == "qe":
        f = P01 / "results" / args.element / "eos.json"
        if not f.exists():
            sys.exit(f"run project 01 first: {f} not found")
        return json.loads(f.read_text())["a0_A"]
    from ase.build import bulk
    from ase.calculators.emt import EMT
    from ase.filters import ExpCellFilter
    from ase.optimize import BFGS
    at = bulk(args.element, "fcc", a={"Al": 4.05, "Ni": 3.52}[args.element], cubic=True)
    at.calc = EMT()
    BFGS(ExpCellFilter(at), logfile=None).run(fmax=1e-4)
    return float(at.cell[0, 0])


def debye_temperature(args) -> float | None:
    f = P01 / "results" / args.element / "elastic.json"
    if args.calc == "qe" and f.exists():
        return json.loads(f.read_text())["debye_temperature_K"]
    return {"Al": 428.0, "Ni": 450.0}[args.element]  # experimental fallback


def save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, default=float))
    print(f"wrote {path.relative_to(ROOT)}")
