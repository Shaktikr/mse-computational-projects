# 08 · Phase-field model of solid-state sintering (`pfsinter`)

A compact, fully vectorised implementation of the Wang (2006) phase-field model of
solid-state sintering: neck formation between particles, pore rounding and grain growth,
with the dominant diffusion path (surface vs grain boundary) selected through the mobility.

## Model

Conserved density ρ (1 = solid, 0 = pore) and one non-conserved order parameter ηᵢ per
particle/grain:

$$f = A\rho^2(1-\rho)^2 + B\Big[\rho^2 + 6(1-\rho)\sum\eta_i^2 - 4(2-\rho)\sum\eta_i^3 + 3\big(\sum\eta_i^2\big)^2\Big]$$

$$\frac{\partial\rho}{\partial t}=\nabla\cdot\Big(M\nabla\frac{\delta F}{\delta\rho}\Big),\qquad
\frac{\partial\eta_i}{\partial t}=-L\frac{\delta F}{\delta\eta_i}$$

$$M = M_{vol}\,\phi(\rho)+M_{vap}\,[1-\phi(\rho)]+M_{surf}\,16\rho^2(1-\rho)^2+M_{gb}\sum_{i\ne j}\eta_i\eta_j$$

The surface term is active only where ρ ≈ ½ (free surface) and the grain-boundary term
only where two grains meet, so the ratio M_surf/M_gb decides which path carries the
matter. Time integration: semi-implicit Fourier-spectral scheme with linear stabilisation
(periodic boundaries), ~10–100× larger time steps than explicit Euler, mass conserved to
machine precision (unit-tested).

**Not included:** rigid-body motion of particles (Wang's advection term), so shrinkage
(centre-to-centre approach) is not captured — neck growth, surface smoothing and grain
growth are. Adding the advection velocity is a natural extension project.

## Run it

```bash
python scripts/two_particle_neck_growth.py      # neck growth, surface vs grain-boundary diffusion
python scripts/multi_particle_sintering.py      # random powder aggregate, snapshots + GIF
python scripts/two_particle_neck_growth.py --quick   # 20-second test
```

## Results

NECK_RESULTS_PLACEHOLDER

## Connecting to experiments

* Units are reduced (grid spacing and time scale set by the mobilities). To map to real
  time, calibrate M_surf and M_gb with δD_s and δD_gb (e.g. from project 02 / Frost–Ashby
  data) and the interface width with the surface energy.
* Compare simulated neck sizes with SEM fractographs of interrupted SPS runs, and grain
  growth with EBSD.
* For SPS, add a temperature field from project 07 (mobilities ∝ exp(−Q/RT)) and an
  applied-stress term.

## References

* Y.U. Wang, *Acta Mater.* 54 (2006) 953 — phase-field model of sintering with rigid-body motion.
* R.L. Coble, *J. Am. Ceram. Soc.* 41 (1958) 55 — initial-stage sintering kinetics.
* L.-Q. Chen & J. Shen, *Comput. Phys. Commun.* 108 (1998) 147 — semi-implicit Fourier-spectral method.
* G.C. Kuczynski, *Trans. AIME* 185 (1949) 169 — neck-growth laws.
