#!/usr/bin/env bash
# Generate the O and Zn PAW pseudopotentials (PBE, scalar-relativistic) with ld1.x.
# Recipes: pslibrary (A. Dal Corso), O-n-kjpaw_psl.0.1 and Zn-dn-kjpaw_psl.0.2.2.
#   conda activate dft && bash make_pseudo.sh
set -e
cd "$(dirname "$0")"
command -v ld1.x >/dev/null || { echo "ld1.x not found (conda activate dft)"; exit 1; }
for el in O Zn; do
  ld1.x < $el.in > $el.ld1.out 2>&1
  echo "$el: ld1.x exit $?"
done
rm -f ld1.wfc* ld1ps.wfc ld1.test vx.pot 2>/dev/null || true
ls -1 *.UPF
