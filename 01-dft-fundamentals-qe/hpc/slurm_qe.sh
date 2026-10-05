#!/bin/bash
# ------------------------------------------------------------------------------------
# Generic SLURM job script for Quantum ESPRESSO runs (adapt partition/module names to the
# cluster you use, e.g. the IIT Kanpur HPC facility - check its user guide for the exact
# partition names, QE module and scratch paths).
#   sbatch hpc/slurm_qe.sh
# ------------------------------------------------------------------------------------
#SBATCH --job-name=qe-elastic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40          # one MPI rank per physical core
#SBATCH --time=12:00:00
#SBATCH --partition=standard          # <-- change
#SBATCH --output=%x-%j.out

module purge
module load quantum-espresso          # <-- change to the cluster's module name
# module load python/3.11             # if Python is also provided as a module

cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1
# -nk = number of k-point pools: k-point parallelisation is the most efficient for small
# metallic cells; choose nk so that ntasks/nk ranks share each pool's plane waves.
export DFTLAB_PW_COMMAND="srun pw.x -nk 8"

python3 scripts/02_eos.py     --system NiAl
python3 scripts/03_elastic.py --system NiAl
