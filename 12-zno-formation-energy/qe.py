"""
qe.py - small, transparent helper for running Quantum ESPRESSO (pw.x) on
multi-species systems: an insulator (ZnO), a metal (Zn) and a magnetic molecule (O2).

It writes a plain-text pw.x input (so you can read exactly what is computed), runs it,
and parses energy / pressure / final cell / final positions / magnetisation.

Units: QE works in Rydberg and Bohr. 1 Ry = 13.605693 eV, 1 Bohr = 0.529177 Angstrom.
"""
import os, re, subprocess, time, shutil
import numpy as np

RY = 13.605693122994        # eV per Ry
HERE = os.path.dirname(os.path.abspath(__file__))
PSEUDO_DIR = os.path.join(HERE, "pseudo")
RUNS = os.path.join(HERE, "runs")
NPROC = int(os.environ.get("NPROC", "4"))   # MPI processes = physical cores
PW = os.environ.get("QE_PW") or shutil.which("pw.x") or "pw.x"

SPECIES = {   # element -> (mass, pseudopotential file)
    "Zn": (65.38, "Zn.pbe-dn-kjpaw_psl.0.2.2.UPF"),
    "O": (15.999, "O.pbe-n-kjpaw_psl.0.1.UPF"),
}


def _finished(out):
    return os.path.exists(out) and "JOB DONE" in open(out, errors="replace").read()


def write_pw_input(fname, cell, symbols, frac, *, calc="scf", ecut=50, ecutrho=None,
                   kpts=(6, 6, 4), metal=False, degauss=0.02, tot_mag=None,
                   prefix="x", conv_thr=1e-9, press_conv=0.1):
    """cell: 3x3 array in Angstrom (rows = lattice vectors); frac: fractional coordinates.
    metal=True  -> Marzari-Vanderbilt smearing (needed for a Fermi surface: hcp Zn)
    metal=False -> fixed occupations (insulator: ZnO, molecule: O2)
    tot_mag     -> spin-polarised run with this total moment in Bohr magnetons (O2: 2)
    kpts=None   -> Gamma point only (isolated molecule in a box)
    A finished run (same name, .out) is never overwritten: its input stays the record."""
    if _finished(fname[:-3] + ".out"):
        return
    ecutrho = ecutrho or 8 * ecut          # PAW/ultrasoft: charge density needs a denser grid
    species = sorted(set(symbols), key=symbols.index)
    L = ["&CONTROL",
         f"  calculation = '{calc}'",
         f"  prefix = '{prefix}'",
         f"  outdir = './tmp_{prefix}'",
         f"  pseudo_dir = '{PSEUDO_DIR}'",
         "  tstress = .true.",
         "  tprnfor = .true.",
         "  disk_io = 'low'",
         "  forc_conv_thr = 1.0d-4",
         "/",
         "&SYSTEM",
         "  ibrav = 0",
         f"  nat = {len(symbols)}",
         f"  ntyp = {len(species)}",
         f"  ecutwfc = {ecut}",
         f"  ecutrho = {ecutrho}"]
    if metal:
        L += ["  occupations = 'smearing'", "  smearing = 'mv'", f"  degauss = {degauss}"]
    else:
        L += ["  occupations = 'fixed'"]
    if tot_mag is not None:
        L += ["  nspin = 2", f"  tot_magnetization = {tot_mag}"]
        L += [f"  starting_magnetization({i + 1}) = 0.5" for i in range(len(species))]
    L += ["/", "&ELECTRONS", f"  conv_thr = {conv_thr:.1e}", "  mixing_beta = 0.5", "/"]
    if calc in ("relax", "vc-relax"):
        L += ["&IONS", "/"]
    if calc == "vc-relax":
        L += ["&CELL", f"  press_conv_thr = {press_conv}", "/"]
    L.append("ATOMIC_SPECIES")
    for s in species:
        L.append(f"  {s} {SPECIES[s][0]} {SPECIES[s][1]}")
    L.append("CELL_PARAMETERS angstrom")
    L += ["  " + "  ".join(f"{x:.10f}" for x in v) for v in cell]
    L.append("ATOMIC_POSITIONS crystal")
    L += [f"  {s} " + "  ".join(f"{x:.10f}" for x in p) for s, p in zip(symbols, frac)]
    if kpts is None:
        L.append("K_POINTS gamma")
    else:
        L += ["K_POINTS automatic", f"  {kpts[0]} {kpts[1]} {kpts[2]} 0 0 0"]
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, "w") as f:
        f.write("\n".join(L) + "\n")


def run_pw(inp, nproc=NPROC, npool=None):
    """Run pw.x with MPI inside the runs/ folder. A finished output is reused, not recomputed."""
    out = inp[:-3] + ".out"
    if not _finished(out):
        npool = npool or nproc
        cmd = (f"mpirun --oversubscribe -np {nproc} {PW} -nk {npool} "
               f"-in {os.path.basename(inp)} > {os.path.basename(out)} 2>&1")
        t0 = time.time()
        subprocess.run(cmd, shell=True, check=False, cwd=os.path.dirname(inp),
                       env={**os.environ, "OMP_NUM_THREADS": "1"})
        print(f"    ran {os.path.basename(inp)}  ({time.time() - t0:.0f} s)", flush=True)
    res = parse(out)
    if not res.get("done") or "energy_eV" not in res:
        raise RuntimeError(f"pw.x did not finish cleanly: {out}")
    return res


def parse(out):
    txt = open(out, errors="replace").read()
    res = {"done": "JOB DONE" in txt}
    e = re.findall(r"!\s+total energy\s+=\s+(-?\d+\.\d+)\s+Ry", txt)
    if e:
        res["energy_eV"] = float(e[-1]) * RY
    p = re.findall(r"P=\s*(-?\d+\.\d+)", txt)
    if p:
        res["pressure_kbar"] = float(p[-1])
    m = re.findall(r"total magnetization\s+=\s+(-?\d+\.\d+)", txt)
    if m:
        res["magnetization"] = float(m[-1])
    f = re.findall(r"Total force =\s+(\d+\.\d+)", txt)
    if f:
        res["total_force_Ry_bohr"] = float(f[-1])
    c = re.findall(r"CELL_PARAMETERS \(angstrom\)\n(.*)\n(.*)\n(.*)", txt)
    if c:
        res["cell"] = [[float(x) for x in r.split()] for r in c[-1]]
    if "ATOMIC_POSITIONS (crystal)" in txt:
        pos = []
        for line in txt.split("ATOMIC_POSITIONS (crystal)")[-1].strip().splitlines():
            q = line.split()
            if len(q) < 4 or q[0] not in SPECIES:
                break
            pos.append([float(x) for x in q[1:4]])
        res["frac"] = pos
    return res


# ---------------------------------------------------------------- structures
def wurtzite_zno(a=3.250, c=5.207, u=0.382):
    """Wurtzite ZnO (space group P6_3mc), 2 formula units. Defaults: experiment."""
    cell = np.array([[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]])
    frac = [[1/3, 2/3, 0.0], [2/3, 1/3, 0.5], [1/3, 2/3, u], [2/3, 1/3, 0.5 + u]]
    return cell, ["Zn", "Zn", "O", "O"], frac


def hcp_zn(a=2.665, c=4.947):
    """hcp Zn, 2 atoms. Defaults: experiment (note the unusually large c/a = 1.856)."""
    cell = np.array([[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]])
    return cell, ["Zn", "Zn"], [[1/3, 2/3, 0.25], [2/3, 1/3, 0.75]]


def o2_molecule(box=10.0, d=1.21):
    """O2 molecule along z in a slightly non-cubic empty box (Angstrom)."""
    cell = np.diag([box, box + 0.4, box + 0.8])
    z = d / (box + 0.8)
    return cell, ["O", "O"], [[0.5, 0.5, 0.5 - z / 2], [0.5, 0.5, 0.5 + z / 2]]
