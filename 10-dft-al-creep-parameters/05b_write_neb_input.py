"""Step 5b: write a ready-to-run climbing-image NEB (neb.x) input for the vacancy jump.

A nearest-neighbour atom jumps into the vacancy. Initial state = relaxed vacancy (step 3).
The final state is the same relaxed structure mirrored through the midpoint of the jump
(the midpoint of a nearest-neighbour bond is an inversion centre of the fcc lattice), so
initial and final states are symmetry-equivalent and have identical energy.
neb.x (Quantum ESPRESSO) relaxes a chain of images between them; the climbing image
converges exactly onto the saddle point, whose energy above the initial state is E_m.

Self-diffusion (vacancy mechanism) activation energy:  Q = E_f + E_m."""
import os, json
import numpy as np
from scipy.optimize import linear_sum_assignment
from qe import *

NEB = os.environ.get("QE_NEB", "neb.x")
n, k = 2, 6
vac = json.load(open("results_vacancy.json"))
a0 = vac["a0_A"]
cell, frac = fcc_conventional_supercell(a0, n)
os.makedirs("runs/05_neb", exist_ok=True)
os.chdir("runs/05_neb")

ini = np.load(f"../03_vac/vac_relaxed_frac_{n}.npy") % 1.0      # 31 atoms, vacancy at origin
mover = 0                                                         # unrelaxed site (0.25,0.25,0)
m = np.array([0.125, 0.125, 0.0]) / 1.0                           # midpoint of the jump (fractional)
inv = (2 * m - ini) % 1.0                                         # inverted structure
# re-order the inverted atoms so that every atom keeps its identity (shortest path)
def pdist(a, b):
    d = a[:, None, :] - b[None, :, :]; d -= np.round(d)
    return np.linalg.norm(d @ cell, axis=2)
D = pdist(ini, inv)
D[mover, :] = 1e3; D[:, :] += 0
# the mover must go to the image of itself (which sits on the old vacancy site)
row, col = linear_sum_assignment(np.delete(np.delete(D, mover, 0), mover, 1))
fin = ini.copy()
others = [i for i in range(len(ini)) if i != mover]
fin[others] = inv[[others[c] for c in col]]
fin[mover] = inv[mover]
# keep coordinates continuous with the initial ones (no jumps across the cell boundary)
dd = fin - ini; dd -= np.round(dd); fin = ini + dd
print("mover initial", ini[mover], "final", fin[mover])
print("max displacement of non-moving atoms (A):",
      np.max(np.linalg.norm((dd[others]) @ cell, axis=1)))

def block(pos):
    return "\n".join("Al " + " ".join(f"{x:.10f}" for x in p) for p in pos)

txt = f"""BEGIN
BEGIN_PATH_INPUT
&PATH
  string_method = 'neb'
  nstep_path    = 60
  num_of_images = 7
  opt_scheme    = 'broyden'
  CI_scheme     = 'auto'
  path_thr      = 0.05
  k_max = 0.3, k_min = 0.2
/
END_PATH_INPUT
BEGIN_ENGINE_INPUT
&CONTROL
  prefix = 'neb'
  outdir = './tmp'
  pseudo_dir = '{os.path.relpath(PSEUDO_DIR)}'
  disk_io = 'low'
/
&SYSTEM
  ibrav = 0, nat = {len(ini)}, ntyp = 1
  ecutwfc = 40, ecutrho = 320
  occupations = 'smearing', smearing = 'mv', degauss = 0.02
/
&ELECTRONS
  conv_thr = 1.0d-8
  mixing_beta = 0.5
/
ATOMIC_SPECIES
  Al 26.9815 {AL_PP}
BEGIN_POSITIONS
FIRST_IMAGE
ATOMIC_POSITIONS crystal
{block(ini)}
LAST_IMAGE
ATOMIC_POSITIONS crystal
{block(fin)}
END_POSITIONS
K_POINTS automatic
  {k} {k} {k} 0 0 0
CELL_PARAMETERS angstrom
""" + "\n".join("  " + "  ".join(f"{x:.10f}" for x in v) for v in cell) + """
END_ENGINE_INPUT
END
"""
open("neb.in", "w").write(txt)
print("wrote runs/05_neb/neb.in  ->  run on a cluster with:")
print("  mpirun -np 32 neb.x -nk 4 -inp neb.in > neb.out")
print("neb.x writes the converged energy profile to neb.dat (image, E - E_initial in eV, error);")
print("E_m is the maximum of that profile (the climbing image).")
