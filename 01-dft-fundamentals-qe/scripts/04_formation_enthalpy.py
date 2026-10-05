"""Step 4 - formation enthalpy of B2 NiAl at 0 K (needs eos.json for Al, Ni and NiAl).

    dH_f(NiAl) = E(NiAl)/2 - [E(Ni) + E(Al)]/2        (eV per atom)

IMPORTANT: total energies from different calculations can only be subtracted if they use
the SAME plane-wave cut-off (and the same pseudopotentials / functional). Steps 2-3 use a
cut-off converged for each system separately, so here the three energies are recomputed
at a common cut-off (default 60 Ry - converged to 1 meV/atom for Ni and NiAl, see step 1)
at each system's own equilibrium lattice parameter.

A negative value means the compound is stable with respect to the pure elements. The very
negative dH_f of NiAl (strong Ni-Al bonding) underlies its high melting point (~1911 K)
and high-temperature strength - and also its low room-temperature ductility.

    python scripts/04_formation_enthalpy.py
"""

from types import SimpleNamespace

from common import RESULTS, load_json, make_factory, save_json  # noqa: E402
from dftlab import config  # noqa: E402
from dftlab.structures import build  # noqa: E402


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ecutwfc", type=float, default=60.0, help="common cut-off for all three systems (Ry)")
    ap.add_argument("--calc", default="qe", choices=["qe", "emt"])
    a = ap.parse_args()
    energies = {}
    for s in ("Al", "Ni", "NiAl"):
        sub = s if a.calc == "qe" else f"{s}_emt"
        a0 = load_json(RESULTS / sub / "eos.json")["a0_A"]
        atoms = build(s, a=a0)
        args = SimpleNamespace(system=s, calc=a.calc, ecutwfc=a.ecutwfc, kspacing=None)
        atoms.calc = make_factory(args, "formation")(atoms, f"common_ecut_{int(a.ecutwfc)}")
        energies[s] = atoms.get_potential_energy() / len(atoms)
        print(f"{s}: a0 = {a0:.4f} A, E = {energies[s]:.5f} eV/atom")
    dH = energies["NiAl"] - 0.5 * (energies["Ni"] + energies["Al"])
    out = RESULTS / ("NiAl" if a.calc == "qe" else "NiAl_emt")
    save_json(out / "formation_enthalpy.json", {
        "common_ecutwfc_Ry": a.ecutwfc,
        "kspacing_invA": {s: config.DEFAULTS[s]["kspacing"] for s in energies},
        "E_per_atom_eV": energies,
        "formation_enthalpy_eV_per_atom": dH,
        "formation_enthalpy_kJ_per_mol_atoms": dH * 96.485,
        "note": "Ni reference is ferromagnetic fcc Ni; each phase at its own DFT equilibrium volume; 0 K, no zero-point energy.",
    })
    print(f"dH_f(NiAl) = {dH:.3f} eV/atom = {dH * 96.485:.1f} kJ/mol-atoms")


if __name__ == "__main__":
    main()
