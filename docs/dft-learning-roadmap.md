# Learning DFT from scratch — a 16-week roadmap for a materials PhD student

Goal: go from "never run DFT" to independently setting up, converging and critically
interpreting DFT calculations for high-temperature structural alloys (B2 intermetallics,
superalloys, medium/high-entropy alloys). Each phase ends with a deliverable that is
already in this repository, so you can compare your numbers with mine.

---

## Phase 0 (week 0) — tools

* Linux shell, `ssh`, `scp`, `tmux`; Python with numpy/matplotlib; git.
* Install Quantum ESPRESSO + ASE: see [getting-started.md](getting-started.md).
* Learn the cluster's job scheduler (SLURM): `sbatch`, `squeue`, `scancel`, and the
  template `01-dft-fundamentals-qe/hpc/slurm_qe.sh`.

## Phase 1 (weeks 1–3) — what DFT actually computes

**Theory.** Many-electron Schrödinger equation → Born–Oppenheimer approximation →
Hohenberg–Kohn theorems (the ground-state density determines everything) → Kohn–Sham
equations (non-interacting electrons in an effective potential) → exchange–correlation
functionals (LDA, GGA-PBE, meta-GGA, hybrids; PBE is the workhorse for metals) →
plane-wave basis and the cut-off energy → pseudopotentials (norm-conserving, ultrasoft,
PAW) → Brillouin-zone sampling (Monkhorst–Pack k-points) → smearing for metals →
self-consistent field cycle.

**Practice** (project 01):
1. Run `inputs/01_al_scf.in` by hand. Find the total energy, Fermi energy, forces, stress
   in the output. Change `celldm(1)` and watch the stress change sign.
2. `scripts/01_convergence.py` — cut-off and k-point convergence for Al, Ni, NiAl.
   Understand why Al needs a dense k-mesh (free-electron metal) while Ni needs a high
   cut-off (localised 3d electrons).

**Read.** D. Sholl & J. Steckel, *Density Functional Theory: A Practical Introduction*
(Wiley), ch. 1–4. F. Giustino, *Materials Modelling using Density Functional Theory*
(OUP), ch. 1–7. The Quantum ESPRESSO input documentation for `pw.x` (INPUT_PW).

**Deliverable.** Converged settings with a plot — `results/*/convergence.png`.

## Phase 2 (weeks 4–6) — structural, elastic and magnetic properties

**Practice** (project 01):
* Equation of state (Birch–Murnaghan) → a₀, B₀ for Al, Ni, B2 NiAl.
* Spin polarisation: `inputs/02_ni_scf_spin.in`; compare non-magnetic vs ferromagnetic Ni.
* Elastic constants C11, C12, C44 from volume-conserving strains; Pugh ratio, Cauchy
  pressure, Zener anisotropy, Debye temperature.
* Formation enthalpy of NiAl — and why every energy in a difference must use the same
  cut-off and pseudopotentials.
* Variable-cell relaxation: `inputs/03_nial_b2_vcrelax.in`.
* Extra exercise: electronic density of states (`dos.x`, `projwfc.x`) of NiAl — find the
  pseudogap at the Fermi level that stabilises the B2 phase.

**Deliverable.** `results/SUMMARY.md` — your table vs experiment. Learn to explain the
typical PBE errors (lattice parameters ~1 % too large for 3d metals, bulk moduli
slightly too small, magnetic moments close to experiment).

## Phase 3 (weeks 7–9) — defects and kinetics (the bridge to creep)

**Practice** (project 02):
* Supercells, vacancy formation energy (unrelaxed and relaxed), finite-size checks.
* Nudged elastic band (climbing image) for the vacancy jump → migration barrier.
* Self-diffusion coefficient D = f a² ν exp(S_f/k) exp(−(E_f+E_m)/kT) and the
  Nabarro–Herring / Coble creep rates that follow from it.

**Read.** Mehrer, *Diffusion in Solids* (Springer), ch. 1–8; Henkelman et al.,
J. Chem. Phys. 113, 9901 (2000) (CI-NEB).

**Deliverable.** `02-.../results/Al_2x2x2/` — E_f, E_m, Q compared with tracer-diffusion
experiments.

## Phase 4 (weeks 10–12) — alloys, chemistry and finite temperature

* **Chemical disorder:** special quasirandom structures (SQS) with `icet` or ATAT `mcsqs`
  for CoCrFeNi, NiCoCr, AlCoCrFe; average over several SQS.
* **Planar faults:** generalised stacking-fault energy (GSFE) curves for fcc alloys —
  they control dislocation dissociation and hence creep resistance; antiphase-boundary
  (APB) energies in B2 and L1₂ (γ′).
* **Solute effects:** substitution energies and migration barriers of Re, W, Ta, Mo in Ni
  (the slow diffusers behind superalloy creep strength).
* **Phonons and temperature:** `phonopy` + QE → phonon DOS, vibrational free energy,
  quasi-harmonic thermal expansion and C_ij(T) — needed because creep happens at
  0.5–0.8 T_m, not at 0 K.

## Phase 5 (weeks 13–16) — DFT meets machine learning

* High-throughput workflows: `pymatgen`, `atomate2` or `AiiDA`; Materials Project / OQMD
  data for screening.
* Machine-learned interatomic potentials (MACE, NequIP, SNAP/ACE in LAMMPS) trained on
  your DFT data → MD of creep and sintering at DFT-like accuracy (connects to projects
  08 and 09). Learn active learning and validation on properties that were NOT in the
  training set (elastic constants, vacancy energies, stacking faults).
* Combine DFT descriptors with experimental data in ML models (projects 05 and 06).

---

## Habits that separate good DFT from bad DFT

1. Converge **the property you need** (energy differences, stresses, phonons), not just
   the total energy.
2. Always compare like with like: same cut-off, pseudopotentials, functional, smearing.
3. Check magnetism explicitly for Fe, Co, Ni, Cr, Mn-containing systems (try FM, AFM, NM).
4. Benchmark against experiment or high-quality references before trusting a new result.
5. Report settings completely (code version, pseudopotentials, cut-offs, k-mesh,
   smearing, supercell size) so others can reproduce your numbers.
6. Keep inputs, outputs and analysis scripts under version control — like this repository.

## First research-relevant DFT problems for a creep / high-temperature alloys PhD

* Vacancy formation and migration in B2 NiAl and CoAl (triple-defect mechanism,
  six-jump cycle) — the basis for diffusion-controlled creep of B2 aluminides.
* Elastic constants of the fcc and bcc/B2 phases of CoCrFeNi–AlCoCrFe MEA composites
  (needed for modulus-compensated creep analysis and load partitioning).
* GSFE / stacking-fault energy of NiCoCr and CoCrFeNi (SQS) vs composition.
* Solute–vacancy binding and diffusion barriers of refractory elements in Ni.
