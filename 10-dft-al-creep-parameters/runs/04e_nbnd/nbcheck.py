import sys,json; sys.path.insert(0,'../..')
from qe import *
a0=json.load(open('../../results_vacancy.json'))['a0_A']
cell,frac=fcc_conventional_supercell(a0,2)
pos=final_positions('../05_mig/mig_e30_k4_x0.00.out')
for nb in [None, 90]:
    n=f"perf_k6_nb{nb}"; write_pw_input(n+".in",cell,["Al"]*32,frac,ecut=30,kpts=(6,6,6),degauss=0.02,outdir="./t",nbnd=nb)
    ep=run_pw(n+".in",n+".out")['energy_eV']
    n=f"vac_k6_nb{nb}"; write_pw_input(n+".in",cell,["Al"]*31,pos,ecut=30,kpts=(6,6,6),degauss=0.02,outdir="./t",nbnd=nb)
    ev=run_pw(n+".in",n+".out")['energy_eV']
    print(nb, 'perf/atom',ep/32,'Ef',ev-31/32*ep, flush=True)
