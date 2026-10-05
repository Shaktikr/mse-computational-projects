# Getting started

## 1. Get the code

```bash
git clone https://github.com/Shaktikr/mse-computational-projects.git
cd mse-computational-projects
```

## 2. Python environment (all projects)

**Option A — conda / mamba (recommended; also installs Quantum ESPRESSO and LAMMPS):**

```bash
conda env create -f environment.yml
conda activate msecomp
```

**Option B — pip only** (DFT and MD codes installed separately):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Check everything works (about 15 s):

```bash
pytest
```

## 3. Quantum ESPRESSO (projects 01–02)

| Platform | How |
|---|---|
| conda (any OS) | included in `environment.yml` (`qe` from conda-forge) |
| Ubuntu / WSL2 | `sudo apt install quantum-espresso` — on **Ubuntu 24.04** the packaged pw.x 6.7 aborts with `buffer overflow detected`; build the 10-line fix once: `gcc -shared -fPIC -O2 -U_FORTIFY_SOURCE -o 01-dft-fundamentals-qe/tools/libsnprintf_chk_fix.so 01-dft-fundamentals-qe/tools/snprintf_chk_fix.c` — `tools/env.sh` then preloads it automatically |
| Windows | use WSL2 (Ubuntu) and follow the line above |
| Cluster | load the site's QE module and edit `01-dft-fundamentals-qe/hpc/slurm_qe.sh` |

Then

```bash
cd 01-dft-fundamentals-qe
source tools/env.sh 4      # 4 MPI processes
bash run_all.sh 4
```

## 4. LAMMPS (project 09)

`pip install lammps` (or conda-forge `lammps`) provides the Python module with the
MANYBODY package (EAM potentials). If the import fails with `libmpi.so.12` missing,
install the MPICH runtime: `sudo apt install libmpich12`.

## 5. Running each project

| Project | Command | Time on a laptop |
|---|---|---|
| 01 DFT fundamentals | `bash 01-dft-fundamentals-qe/run_all.sh 4` | 1–2 h (minutes on a cluster node) |
| 02 Vacancy → creep | `bash 02-dft-vacancy-diffusion-creep/run_all.sh 4` | 1–2 h |
| 03 Creep analysis | `python 03-creep-data-analysis/scripts/analyze_creep.py <folder>` | seconds |
| 04 Mechanism maps | `python 04-deformation-mechanism-maps/scripts/make_maps.py` | seconds |
| 05 ML creep life | `python 05-ml-creep-rupture-life/scripts/train_evaluate.py` | ~10 min |
| 06 ML HEA | `python 06-ml-hea-phase-strength/scripts/phase_and_strength.py` | ~2 min |
| 07 SPS | `python 07-sps-joule-heating-densification/scripts/run_electrothermal.py` then `run_densification.py` | ~3 min |
| 08 Phase field | `python 08-phase-field-sintering/scripts/two_particle_neck_growth.py` | 15–30 min |
| 09 MD creep | `python 09-md-nanocrystalline-creep/scripts/run_md_creep.py` | ~1 h |

Every script prints where its figures and JSON summaries are written (`results/`).

## 6. Using your own data

* Creep curves → CSV files in the format of `03-creep-data-analysis/data/template_creep_test.csv`.
* Creep-rupture database → CSV with the columns listed in `05-ml-creep-rupture-life/creepml/data.py`.
* New alloy for deformation maps → copy `04-deformation-mechanism-maps/materials/TEMPLATE_my_alloy.json`.
* SPS run logs → replace the simulated (t, T, D) arrays in `07-.../scripts/run_densification.py`.
