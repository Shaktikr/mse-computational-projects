"""Step 1 - vacancy formation energy in an fcc supercell.

    python scripts/01_vacancy_formation.py --element Al            # DFT (needs project 01 eos.json)
    python scripts/01_vacancy_formation.py --element Al --calc emt # quick test

Output: results/<element>_<n>x<n>x<n>/vacancy.json, perfect.extxyz, vacancy_relaxed.extxyz
"""

from ase.io import write

from common import lattice_parameter, out_dir, parser, save_json, settings  # noqa: E402
from vacdiff.defects import fcc_supercell, kpoints_used, vacancy_formation  # noqa: E402


def main():
    args = parser(__doc__).parse_args()
    s = settings(args)
    a0 = lattice_parameter(args)
    perfect = fcc_supercell(args.element, a0, args.n)
    print(f"{args.element}: a0 = {a0:.4f} A, {len(perfect)} sites, k-mesh {kpoints_used(perfect, s)}")
    res, relaxed = vacancy_formation(perfect, s)
    out = out_dir(args)
    write(out / "perfect.extxyz", perfect)
    write(out / "vacancy_relaxed.extxyz", relaxed)
    save_json(out / "vacancy.json", {"element": args.element, "a0_A": a0, "settings": s.to_dict(),
                                     "kmesh": kpoints_used(perfect, s), **res.to_dict()})
    print(f"E_f (unrelaxed) = {res.Ef_unrelaxed_eV:.3f} eV,  E_f (relaxed) = {res.Ef_relaxed_eV:.3f} eV")


if __name__ == "__main__":
    main()
