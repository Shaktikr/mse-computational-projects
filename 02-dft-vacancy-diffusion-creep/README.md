# 02 · From vacancies to diffusional creep with DFT (`vacdiff`)

Diffusion-controlled creep (Nabarro–Herring, Coble, and the climb step of dislocation
creep) is limited by how fast vacancies form and move. This project computes those two
energies from first principles and carries them all the way to a creep rate.

$$E_f^v = E_{N-1}^{relaxed} - \tfrac{N-1}{N}E_N,\qquad E_m = E_{saddle}-E_{initial}\ \text{(CI-NEB)}$$

$$D = f\,a^2\,\nu\,e^{S_f/k}\,\exp\!\left(-\frac{E_f+E_m}{kT}\right)\qquad (f = 0.7815\ \text{for fcc})$$

$$\dot\varepsilon_{NH}=14\frac{\Omega D_L\sigma}{kTd^2},\qquad \dot\varepsilon_{Coble}=\frac{150}{\pi}\frac{\Omega\,\delta D_{gb}\,\sigma}{kTd^3}$$

## Workflow

1. `scripts/01_vacancy_formation.py` — 2×2×2 fcc supercell (32 sites) at the **DFT** lattice
   parameter from project 01; unrelaxed and relaxed vacancy formation energy.
2. `scripts/02_vacancy_migration_neb.py` — migration barrier E_m by two methods:
   * `--method midpoint` (default): the jumping atom is fixed half-way along the jump and all
     other atoms are relaxed. For the symmetric nearest-neighbour jump in a pure fcc metal
     the saddle lies exactly there, so one relaxation gives E_m.
   * `--method neb`: climbing-image NEB (ASE, IDPP initial path) between relaxed end
     states — the general method, needed for asymmetric paths (B2 compounds, solute–vacancy
     pairs, concentrated alloys) but one DFT calculation per image per optimiser step.
   A unit test checks that both give the same barrier (1.164 eV for EMT Ni).
3. `scripts/03_diffusion_and_creep.py` — D(T) with the attempt frequency ν = kθ_D/h (θ_D
   from the DFT elastic constants of project 01), comparison with tracer-diffusion
   experiments, Nabarro–Herring and Coble rates vs grain size and the crossover grain size.

```bash
bash run_all.sh 4 Al            # DFT (needs project 01 results)
python scripts/01_vacancy_formation.py --element Ni --calc emt   # seconds, toy potential
```

## Results (Al, PBE)

Results are written to `results/Al_2x2x2/` (`vacancy.json`, `neb.json`,
`diffusion_creep.json` and figures) by `bash run_all.sh`. Reference values for Al:
experiment E_f ≈ 0.67 eV, E_m ≈ 0.6 eV, Q ≈ 1.3–1.5 eV (142 kJ/mol tracer diffusion);
PBE typically gives E_f ≈ 0.6–0.7 eV and E_m ≈ 0.5–0.6 eV.

## Limitations — and how to go further

* A 32-site cell has finite-size errors of ~0.05 eV for E_f; check with 3×3×3 (108 sites).
* ν and S_f are estimated; a phonon calculation (harmonic transition-state theory,
  Vineyard 1957) gives both properly.
* GGA functionals underestimate vacancy formation energies because of the intrinsic
  surface error (Carling et al. 2000; Mattsson & Mattsson 2002).
* **B2 intermetallics** (NiAl, CoAl) diffuse by more complex mechanisms (six-jump cycle,
  triple defects, antisite-assisted jumps): the same scripts with `b2_supercell` are the
  starting point for a research-level study of diffusion-controlled creep in nanograined
  B2 aluminides.

## References

* G. Henkelman, B.P. Uberuaga & H. Jónsson, *J. Chem. Phys.* 113 (2000) 9901 — CI-NEB.
* S. Smidstrup et al., *J. Chem. Phys.* 140 (2014) 214106 — IDPP path.
* H. Mehrer, *Diffusion in Solids* (Springer, 2007).
* C. Herring, *J. Appl. Phys.* 21 (1950) 437; R.L. Coble, *J. Appl. Phys.* 34 (1963) 1679.
* K. Carling et al., *Phys. Rev. Lett.* 85 (2000) 3862 — vacancies in Al: first principles vs experiment.
* T.R. Mattsson & A.E. Mattsson, *Phys. Rev. B* 66 (2002) 214110 — surface-error correction of vacancy energies.
* Y. Mishin & D. Farkas, *Phil. Mag. A* 75 (1997) 169 — diffusion mechanisms in B2 NiAl.
