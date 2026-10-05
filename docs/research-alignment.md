# How the projects map onto the research themes of the group

The projects in this repository were chosen to match the research directions of the
**Mechanics and Materials Research Laboratory (MMRL), Department of MSE, IIT Kanpur**
(PI: Dr. Niraj Mohan Chawake). From the group's public pages and publication list, the
main themes are:

* creep and high-temperature deformation (time-dependent deformation, cryogenic to
  high-temperature mechanical behaviour);
* solid-state sintering and spark plasma sintering (SPS) of metals, intermetallics and
  medium/high-entropy alloys;
* ordered intermetallics (B2 aluminides), superalloys and medium/high-entropy alloys
  for high-temperature structural applications;
* machine learning for creep-life prediction.

> This is a personal learning and research-preparation portfolio. It is not an official
> repository of the group, and nothing here reproduces unpublished group data.

## Theme → project map

| Research theme | Related public work of the group (titles as listed on Google Scholar / IIT Kanpur pages) | Projects |
|---|---|---|
| Creep of nanograined B2 aluminides (PI's doctoral topic, IIT Madras) | Doctoral work on creep and high-temperature deformation of nanograined B2 aluminides | **01** (B2 NiAl DFT), **02** (vacancy diffusion → diffusional creep), **09** (MD creep of nanocrystalline Ni and B2 CoAl) |
| Creep testing and constitutive analysis | *Creep deformation of a composite of dual-phase medium entropy alloys*, JOM (2025) | **03** (creep-curve toolkit, threshold stress for composites), **04** (deformation-mechanism maps) |
| Joule heating in SPS | *On Joule heating during spark plasma sintering of metal powders*, Scripta Materialia 93 (2014) 52–55 | **07A** (electro-thermal finite-volume model, conductive vs insulating powders) |
| Diffusivity from SPS densification data | *Estimation of diffusivity from densification data obtained during spark plasma sintering* | **07B** (Bernard-Granger analysis, master sintering curve, D_eff) |
| Solid-state sintering of NiCoCr | Group project on solid-state sintering of NiCoCr | **08** (phase-field sintering), **06** (phase stability of NiCoCr / CoCrFeNi) |
| MEA composites made by milling + SPS | *Composite of medium entropy alloys synthesized using spark plasma sintering*; *How does powder metallurgy facilitate the preparation of intermetallics and high-entropy alloys?* (ASM, 2024) | **06** (phase prediction of CoCrFeNi (fcc) and AlCoCrFe (bcc) constituents), **07** |
| HEA design and refractory HEAs | *High entropy alloy design concept enabled emerging novel materials…*, Frontiers in Materials 10 (2023); *Heterogeneous microstructure in nonequiatomic MoNbTaW refractory HEA after high pressure torsion*, MSE A 864 (2023) | **06** (ML phase + high-temperature strength) |
| ML for creep-life prediction | Group project on ML for creep-life prediction | **05** (grouped CV, uncertainty, inverse design) |

## A suggested research path that ties the pieces together

1. **Experiment → constitutive law.** Analyse your creep tests with project 03
   (minimum creep rate, n, Q, threshold stress, Larson–Miller, Monkman–Grant).
2. **Mechanism identification.** Compare n and Q with diffusion data and place the test
   conditions on a deformation-mechanism map (project 04) built with your own alloy's
   parameters.
3. **First-principles inputs.** Where diffusion or elastic data are missing (new MEA
   phases, B2 intermetallics), compute elastic constants (project 01) and vacancy
   formation/migration energies (project 02) with DFT.
4. **Processing.** For SPS-made materials, check how far the sample temperature deviates
   from the pyrometer reading (project 07A) before extracting activation energies from
   densification data (project 07B); use phase-field (project 08) to visualise neck
   growth and grain growth.
5. **Data-driven design.** Use ML (projects 05 and 06) to screen compositions and to
   predict rupture life with honest uncertainty; feed the best candidates back to step 1.
