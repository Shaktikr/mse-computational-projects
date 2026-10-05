"""Constant-stress (creep) molecular dynamics through the LAMMPS Python module.

Protocol (metal units: A, ps, eV, bar):
  1. energy minimisation with box relaxation (removes overlaps at grain boundaries)
  2. NPT equilibration at temperature T and zero stress (anisotropic barostat)
  3. creep: NPT with sigma_xx = sigma (tension; LAMMPS pressure = -sigma), sigma_yy = sigma_zz = 0
     the box length L_x(t) gives the creep strain eps(t) = L_x(t)/L_x(0) - 1
Grain-boundary atoms are identified with common-neighbour analysis (CNA, fixed cut-off):
fcc structure type = 1 (bcc = 3); everything else is counted as boundary/defect.

MD time scales are nanoseconds, so MD creep needs stresses of ~0.3-2 GPa and
temperatures close to the melting point to give measurable strain - rates are 1e7-1e9 1/s,
many orders above laboratory creep. MD is therefore used for MECHANISMS (grain-boundary
diffusion, sliding, dislocation emission) and for trends of the stress exponent with grain
size, not for direct rate prediction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GPA_TO_BAR = 1.0e4


@dataclass
class CreepSettings:
    T: float = 1200.0
    dt_ps: float = 0.002
    t_equil_ps: float = 20.0
    t_creep_ps: float = 40.0
    sample_every: int = 250
    seed: int = 12345
    tdamp_ps: float = 0.1
    pdamp_ps: float = 1.0


def _lmp():
    import lammps

    return lammps.lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])


def setup(data_file: str, pair_style: str, pair_coeff: str, s: CreepSettings, masses: dict | None = None,
          cna_cutoff: float = 3.0):
    """cna_cutoff: between 1st and 2nd neighbour shells, 0.854 a (fcc) or 1.207 a (bcc/B2)."""
    L = _lmp()
    L.commands_string(f"""
        units metal
        atom_style atomic
        boundary p p p
        read_data {data_file}
        pair_style {pair_style}
        pair_coeff {pair_coeff}
        neighbor 1.0 bin
        neigh_modify every 1 delay 0 check yes
        compute cna all cna/atom {cna_cutoff}
        thermo_style custom step temp pxx lx
        thermo 0
    """)
    for t, m in (masses or {}).items():
        L.command(f"mass {t} {m}")
    L.commands_string("""
        fix relax all box/relax iso 0.0 vmax 0.001
        minimize 1e-8 1e-10 2000 20000
        unfix relax
        reset_timestep 0
    """)
    L._fcc_fraction_0K = fcc_fraction(L)  # structure analysis on the quenched (0 K) sample
    L.command(f"timestep {s.dt_ps}")
    L.command(f"velocity all create {s.T} {s.seed} mom yes rot yes dist gaussian")
    L.command(f"fix eq all npt temp {s.T} {s.T} {s.tdamp_ps} aniso 0.0 0.0 {s.pdamp_ps}")
    L.command(f"run {int(s.t_equil_ps / s.dt_ps)}")
    L.command("unfix eq")
    return L


def fcc_fraction(L) -> float:
    """Fraction of atoms in a perfect crystalline environment (CNA type 1 = fcc, 3 = bcc).

    Evaluate it on a minimised (0 K) configuration: thermal vibrations at high T make the
    fixed-cut-off CNA misclassify many crystalline atoms."""
    L.command("run 0")
    cna = np.array(L.numpy.extract_compute("cna", 1, 1))  # per-atom vector
    return float(np.mean((cna == 1) | (cna == 3)))


def creep(L, stress_GPa: float, s: CreepSettings):
    """Run the constant-stress stage. Returns dict of arrays: t_ps, strain, T, pxx_GPa."""
    p = -stress_GPa * GPA_TO_BAR
    L.command(f"fix creep all npt temp {s.T} {s.T} {s.tdamp_ps} x {p} {p} {s.pdamp_ps} "
              f"y 0.0 0.0 {s.pdamp_ps} z 0.0 0.0 {s.pdamp_ps} couple none")
    boxlo, boxhi, *_ = L.extract_box()
    Lx0 = boxhi[0] - boxlo[0]
    n_chunks = int(s.t_creep_ps / s.dt_ps / s.sample_every)
    out = {"t_ps": [0.0], "strain": [0.0], "T": [s.T], "pxx_GPa": [-stress_GPa]}
    for k in range(n_chunks):
        # 'pre no' skips the run set-up, valid only once the new fix has been set up
        L.command(f"run {s.sample_every} {'pre yes' if k == 0 else 'pre no'} post no")
        boxlo, boxhi, *_ = L.extract_box()
        out["t_ps"].append((k + 1) * s.sample_every * s.dt_ps)
        out["strain"].append((boxhi[0] - boxlo[0]) / Lx0 - 1.0)
        out["T"].append(L.get_thermo("temp"))
        out["pxx_GPa"].append(L.get_thermo("pxx") / GPA_TO_BAR)
    L.command("unfix creep")
    return {k: np.array(v) for k, v in out.items()}


def dump(L, path: str):
    L.command(f"write_dump all custom {path} id type x y z c_cna modify sort id")
