# Source this before running the scripts:   source tools/env.sh [nprocs]
# Sets how pw.x is launched. Edit for your machine / cluster.

NP=${1:-2}
export OMP_NUM_THREADS=1
export DFTLAB_PW_COMMAND="mpirun -np ${NP} pw.x"

# --- Ubuntu 24.04 (incl. WSL) + apt quantum-espresso 6.7 only ----------------------------
# The packaged pw.x aborts with "*** buffer overflow detected ***" while reading the
# pseudopotential (a false positive of glibc _FORTIFY_SOURCE=3 in QE's get_md5()).
# Build the tiny preload library once and it is used automatically:
#     gcc -shared -fPIC -O2 -U_FORTIFY_SOURCE -o tools/libsnprintf_chk_fix.so tools/snprintf_chk_fix.c
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${_here}/libsnprintf_chk_fix.so" ]; then
    export LD_PRELOAD="${_here}/libsnprintf_chk_fix.so${LD_PRELOAD:+:$LD_PRELOAD}"
fi

# Running as root inside a container (not needed on a normal account)
if [ "$(id -u)" = "0" ]; then
    export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
fi
echo "pw.x command: ${DFTLAB_PW_COMMAND}"
