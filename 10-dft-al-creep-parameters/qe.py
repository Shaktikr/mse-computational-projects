"""
qe.py - a tiny, transparent helper for running Quantum ESPRESSO (pw.x).

It writes a plain-text pw.x input file (so you can read exactly what is being
computed), runs it, and parses energy / pressure / stress / forces from the output.

Units: QE works in Rydberg (Ry) and Bohr internally. 1 Ry = 13.605693 eV,
1 Bohr = 0.529177 Angstrom.
"""
import os, re, subprocess, time, shutil
import numpy as np

RY = 13.605693122994        # eV per Ry
BOHR = 0.529177210903       # Angstrom per Bohr
PSEUDO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pseudo")
AL_PP = "Al.pbe-n-rrkjus_psl.0.1.UPF"
# number of MPI processes: set e.g.  export NPROC=4  (use your PHYSICAL core count)
NPROC = int(os.environ.get("NPROC", "4"))
# path to the pw.x executable (QE 6.6 compiled from source; change to "pw.x" if it is on your PATH)
PW = os.environ.get("QE_PW") or shutil.which("pw.x") or "pw.x"


def _finished(out):
    """True if a pw.x output file exists and reached the end of the run."""
    return os.path.exists(out) and "JOB DONE" in open(out, errors="replace").read()


def write_pw_input(fname, cell, symbols, frac, *, calc="scf", ecut=30, ecutrho=None,
                   kpts=(12, 12, 12), smearing="mv", degauss=0.02, prefix="al",
                   outdir="./tmp", conv_thr=1e-8, tstress=True, tprnfor=True,
                   fixed=None, extra_control="", extra_ions="", extra_cell="",
                   pseudo=AL_PP, mass=26.9815, nbnd=None):
    """cell: 3x3 array in Angstrom (rows = lattice vectors); frac: fractional coords.
    fixed: optional list of (fx,fy,fz) 0/1 flags per atom (1 = free to move).
    If the matching output (same name, .out) has already finished, the input is left untouched:
    it stays the exact record of that run, and run_pw() will reuse the output."""
    if fname.endswith(".in") and _finished(fname[:-3] + ".out"):
        return
    ecutrho = ecutrho or 8 * ecut          # ultrasoft PPs need a denser charge grid
    nat = len(symbols)
    lines = []
    lines.append("&CONTROL")
    lines.append(f"  calculation = '{calc}'")
    lines.append(f"  prefix = '{prefix}'")
    lines.append(f"  outdir = '{outdir}'")
    lines.append(f"  pseudo_dir = '{PSEUDO_DIR}'")
    lines.append(f"  tstress = {'.true.' if tstress else '.false.'}")
    lines.append(f"  tprnfor = {'.true.' if tprnfor else '.false.'}")
    lines.append("  disk_io = 'low'")
    if extra_control:
        lines.append(extra_control)
    lines.append("/")
    lines.append("&SYSTEM")
    lines.append("  ibrav = 0")
    lines.append(f"  nat = {nat}")
    lines.append("  ntyp = 1")
    lines.append(f"  ecutwfc = {ecut}")
    lines.append(f"  ecutrho = {ecutrho}")
    lines.append("  occupations = 'smearing'")
    lines.append(f"  smearing = '{smearing}'")
    lines.append(f"  degauss = {degauss}")
    if nbnd:
        lines.append(f"  nbnd = {nbnd}")
    lines.append("/")
    lines.append("&ELECTRONS")
    lines.append(f"  conv_thr = {conv_thr:.1e}")
    lines.append("  mixing_beta = 0.5")
    lines.append("/")
    if calc in ("relax", "vc-relax", "neb"):
        lines.append("&IONS")
        if extra_ions:
            lines.append(extra_ions)
        lines.append("/")
    if calc == "vc-relax":
        lines.append("&CELL")
        if extra_cell:
            lines.append(extra_cell)
        lines.append("/")
    lines.append("ATOMIC_SPECIES")
    lines.append(f"  Al {mass} {pseudo}")
    lines.append("CELL_PARAMETERS angstrom")
    for v in cell:
        lines.append("  " + "  ".join(f"{x:.10f}" for x in v))
    lines.append("ATOMIC_POSITIONS crystal")
    for i, (s, p) in enumerate(zip(symbols, frac)):
        tail = ""
        if fixed is not None:
            tail = "  " + " ".join(str(int(f)) for f in fixed[i])
        lines.append(f"  {s} " + "  ".join(f"{x:.10f}" for x in p) + tail)
    lines.append("K_POINTS automatic")
    lines.append(f"  {kpts[0]} {kpts[1]} {kpts[2]} 0 0 0")
    with open(fname, "w") as f:
        f.write("\n".join(lines) + "\n")


def run_pw(inp, out, nproc=NPROC, npool=None):
    """Run pw.x with MPI. Skips the run if a finished output already exists."""
    if _finished(out):
        return parse(out)
    npool = npool or nproc
    root = "--allow-run-as-root " if hasattr(os, "geteuid") and os.geteuid() == 0 else ""
    cmd = (f"mpirun {root}--oversubscribe -np {nproc} {PW} -nk {npool} "
           f"-in {inp} > {out} 2>&1")
    t0 = time.time()
    subprocess.run(cmd, shell=True, check=False,
                   env={**os.environ, "OMP_NUM_THREADS": "1"})
    res = parse(out)
    res["wall"] = time.time() - t0
    return res


def parse(out):
    txt = open(out).read()
    res = {}
    e = re.findall(r"!\s+total energy\s+=\s+(-?\d+\.\d+)\s+Ry", txt)
    if e:
        res["energy_Ry"] = float(e[-1])
        res["energy_eV"] = float(e[-1]) * RY
        res["energies_eV"] = [float(x) * RY for x in e]
    p = re.findall(r"P=\s*(-?\d+\.\d+)", txt)
    if p:
        res["pressure_kbar"] = float(p[-1])
    s = re.findall(r"total\s+stress.*\n(.*)\n(.*)\n(.*)", txt)
    if s:
        rows = s[-1]
        res["stress_kbar"] = np.array([[float(x) for x in r.split()[3:6]] for r in rows])
    f = re.findall(r"Total force =\s+(\d+\.\d+)", txt)
    if f:
        res["total_force_Ry_bohr"] = float(f[-1])
    res["done"] = "JOB DONE" in txt
    res["converged_scf"] = "convergence has been achieved" in txt
    return res


def fcc_primitive(a):
    return 0.5 * a * np.array([[-1, 0, 1], [0, 1, 1], [-1, 1, 0]], float)


def fcc_conventional_supercell(a, n):
    """n x n x n conventional fcc cubes -> (cell, fractional coords) with 4 n^3 atoms."""
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    frac = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for b in basis:
                    frac.append((b + [i, j, k]) / n)
    return a * n * np.eye(3), np.array(frac)


def final_positions(out):
    """Return the last ATOMIC_POSITIONS (crystal) block of a relax output as an array."""
    txt = open(out).read().split("ATOMIC_POSITIONS (crystal)")[-1].strip().splitlines()
    pos = []
    for line in txt:
        p = line.split()
        if len(p) < 4 or p[0] != "Al":
            break
        pos.append([float(x) for x in p[1:4]])
    return np.array(pos)
