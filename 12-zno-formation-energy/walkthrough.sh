#!/usr/bin/env bash
# =====================================================================================
#  Guided, hands-on run of the ZnO formation-energy calculation.
#  It runs the real commands on this machine one at a time: it explains the next command,
#  shows it, waits for Enter, runs it, and then explains what came out.
#
#      bash walkthrough.sh            interactive (press Enter before each command)
#      bash walkthrough.sh --quick    skip the 4-minute relaxation near the end
#      bash walkthrough.sh --auto     no pauses (with --quick: a 3-minute self-test)
#
#  Everything happens in the practice folder ~/zno-by-hand; the repository is not touched.
#  The lines printed after "$" are exactly what you would type by hand.
# =====================================================================================
AUTO=0; QUICK=0
for a in "$@"; do
  case "$a" in
    --auto)  AUTO=1 ;;
    --quick) QUICK=1 ;;
    *) echo "unknown option: $a   (use --auto and/or --quick)"; exit 1 ;;
  esac
done
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK=~/zno-by-hand
NP=${NP:-4}

C=$'\033[1;36m'; Y=$'\033[1;33m'; G=$'\033[0;32m'; N=$'\033[0m'
[ -t 1 ] || { C=; Y=; G=; N=; }            # no colours when the output is not a terminal

step()  { printf '\n%s================ %s ================%s\n' "$C" "$1" "$N"; }
say()   { printf '%s\n' "$@"; }
pause() { [ "$AUTO" = 1 ] || read -r -p "${G}[Enter = run it]${N} " _; }
run()   {                                   # show a command, wait, then run it in THIS shell
  printf '\n%s$ %s%s\n' "$Y" "$1" "$N"
  pause
  local t0=$SECONDS
  eval "$1"
  LAST=$((SECONDS - t0))
}
runpw() {                                   # run pw.x on $1.in and make sure it finished
  say "" "(nothing appears on screen while pw.x works; this one takes about $2)"
  run "mpirun -np $NP pw.x -in $1.in > $1.out"
  if ! grep -q 'JOB DONE' "$1.out"; then
    say "" "pw.x did not finish normally. Its error message, from $1.out:"
    grep -a -B1 -A3 'Error in' "$1.out" | head -8
    exit 1
  fi
  say "(done in $LAST s)"
}

# make "conda activate" usable inside a script (an interactive Ubuntu window already has it)
CONDA_SH="$HOME/miniforge3/etc/profile.d/conda.sh"
[ -f "$CONDA_SH" ] || CONDA_SH="$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
if [ ! -f "$CONDA_SH" ]; then echo "conda not found - see section 4 of README.md"; exit 1; fi
source "$CONDA_SH"

step "What is about to happen"
say "Goal: the energy released by   Zn (metal) + 1/2 O2 (gas) -> ZnO (crystal)." \
    "DFT gives the total energy of ONE arrangement of atoms, so we run three calculations" \
    "(ZnO crystal, Zn crystal, O2 molecule) and subtract:" \
    "" \
    "    dH_f = E(ZnO cell)/2 - E(Zn cell)/2 - E(O2)/2" \
    "" \
    "Each calculation is: write a text input file -> run pw.x on it -> read the text output." \
    "Lines starting with \$ are the commands, exactly as you would type them yourself."

step "1. Switch on the DFT software"
say "Quantum ESPRESSO was installed with conda into an 'environment' called dft." \
    "'conda activate dft' puts its programs (pw.x, ld1.x, mpirun) on the command path."
run "conda activate dft"
say "" "'which' prints where a program lives. No output would mean it is not installed."
run "which pw.x mpirun ld1.x"
say "" "pw.x can also use several threads per process. Together with mpirun that only slows" \
    "it down (on this laptop: 268 s instead of 23 s), so tell it to use one thread:"
run "export OMP_NUM_THREADS=1"

step "2. A practice folder"
say "mkdir makes a folder (-p: no complaint if it already exists). cp -r copies folders." \
    "cd moves into a folder. ls lists what is in it."
run "mkdir -p $WORK"
run "cp -r $PROJ/pseudo $PROJ/inputs $WORK/"
run "cd $WORK/inputs"
run "ls"

step "3. Pseudopotentials: one file per element"
say "A pseudopotential stands in for the nucleus and the inner-shell electrons of an atom, so" \
    "that only the outer (valence) electrons have to be computed. The program ld1.x builds" \
    "one from a short recipe. This is the recipe for oxygen (cat prints a file):"
run "cat ../pseudo/O.in"
say "" "How to read it: zed=8 is oxygen; dft='PBE' is the approximation used for the electrons;" \
    "config lists the electrons ([He] inner shell, 2s2 2p4 valence); lpaw=.true. asks for a" \
    "PAW-type pseudopotential; file_pseudopw is the file that will be written. The table at" \
    "the bottom sets, for each valence state, the radius inside which the atom is smoothed." \
    "" \
    "'<' feeds a file to a program as its input; '>' saves what the program prints to a file."
run "cd ../pseudo"
run "ld1.x < O.in > O.ld1.out"
run "ld1.x < Zn.in > Zn.ld1.out"
run "ls -lh *.UPF"
say "" "The .UPF files are plain text. head -12 shows the first 12 lines of one:"
run "head -12 O.pbe-n-kjpaw_psl.0.1.UPF"
say "" "'Suggested minimum cutoff for wavefunctions' is the starting value for the setting" \
    "ecutwfc: 47 Ry for O and 39 Ry for Zn, so we start with 50 Ry and test it in step 10."
run "cd ../inputs"

step "4. The input file for wurtzite ZnO"
run "cat 01_zno_scf.in"
say "" "How to read it:" \
    " &CONTROL          what to do. 'scf' = compute the energy of this fixed structure." \
    "                   pseudo_dir = where the .UPF files are; outdir = scratch folder." \
    " &SYSTEM           the material and the accuracy. nat = 4 atoms, ntyp = 2 elements;" \
    "                   ecutwfc / ecutrho = plane-wave cut-offs (how finely the electron" \
    "                   waves are described); occupations = 'fixed' because ZnO is an insulator." \
    " &ELECTRONS        when to stop the self-consistency loop (conv_thr)." \
    " ATOMIC_SPECIES    element, mass, pseudopotential file." \
    " CELL_PARAMETERS   the three edge vectors of the repeating cell, in Angstrom" \
    "                   (measured ZnO: a = 3.250, c = 5.207)." \
    " ATOMIC_POSITIONS  where the 4 atoms sit, as fractions of those three vectors." \
    " K_POINTS          the 6 x 6 x 4 sampling grid for the electron states." \
    "Text after ! (upper part) and lines starting with # (lower part) are comments."

step "5. Run pw.x"
say "mpirun -np $NP    start $NP copies of the program that share the work, one per CPU core" \
    "pw.x             the DFT program" \
    "-in FILE         the input file" \
    "> FILE           save everything it prints into this output file"
runpw 01_zno_scf "25 seconds"

step "6. Read the output"
say "tail -4 shows the last 4 lines of a file. 'JOB DONE.' means the run ended normally."
run "tail -4 01_zno_scf.out"
say "" "grep prints the lines of a file that contain a given text. Did the loop converge?"
run "grep 'convergence has been achieved' 01_zno_scf.out"
say "" "The result. pw.x marks the final total energy with '!' at the start of the line" \
    "('^!' means: a line that begins with !):"
run "grep '^!' 01_zno_scf.out"
say "" "It is in Rydberg (1 Ry = 13.6057 eV) and belongs to the whole 4-atom cell." \
    "The self-consistency loop at work: each pass prints an energy and an error estimate" \
    "that shrinks (head -8 keeps the first 8 lines):"
run "grep -E 'total energy  |estimated scf accuracy' 01_zno_scf.out | head -8"
say "" "Sanity check - did pw.x understand the structure? Expect 4 atoms, 36 electrons" \
    "(12 per Zn + 6 per O) and 21 k-points (the 6x6x4 grid reduced by symmetry):"
run "grep -E 'number of atoms/cell|number of electrons|number of k points' 01_zno_scf.out"
say "" "The pressure, in kbar. Positive means this cell is slightly too small for PBE, which" \
    "would like it about 1 % larger - a normal error of the method, not a mistake:"
run "grep 'P=' 01_zno_scf.out"

step "7. The metal: hcp zinc"
say "Same recipe with three differences (grep -n also prints the line numbers):"
run "grep -n -E 'occupations|smearing|degauss|^   16' 02_zn_scf.in"
say "" "A metal has no energy gap, so its electron levels are 'smeared' a little around the" \
    "Fermi level (occupations / smearing / degauss), and it needs a much denser k-point" \
    "grid (16 16 8). The cut-offs are the SAME as for ZnO: energies may only be subtracted" \
    "when they were computed with identical settings."
runpw 02_zn_scf "35 seconds"
run "grep '^!' 02_zn_scf.out"
run "grep 'Fermi' 02_zn_scf.out"

step "8. The molecule: O2"
run "grep -n -E 'nspin|_magnetization' 03_o2_scf.in"
say "" "O2 has two unpaired electrons. nspin = 2 treats spin-up and spin-down electrons" \
    "separately, and tot_magnetization = 2 says there are 2 more 'up' than 'down'." \
    "Forgetting this is the classic mistake in oxide formation energies." \
    "" \
    "The molecule sits alone in an empty 10 A box and needs a single k-point ('gamma')." \
    "(sed -n '/TEXT/,\$p' prints a file from the first line containing TEXT to the end)"
run "sed -n '/^CELL_PARAMETERS/,\$p' 03_o2_scf.in"
runpw 03_o2_scf "35 seconds"
run "grep '^!' 03_o2_scf.out"
run "grep 'total magnetization' 03_o2_scf.out | tail -1"

step "9. The answer: subtract"
run "grep '^!' 01_zno_scf.out 02_zn_scf.out 03_o2_scf.out"
E1=$(grep '^!' 01_zno_scf.out | awk '{print $5}')
E2=$(grep '^!' 02_zn_scf.out  | awk '{print $5}')
E3=$(grep '^!' 03_o2_scf.out  | awk '{print $5}')
say "" "dH_f = [ E(ZnO cell) - E(Zn cell) - E(O2) ] / 2, then x 13.605693 to turn Ry into eV." \
    "Why /2: the ZnO cell holds 2 formula units, the Zn cell 2 atoms, and one ZnO needs" \
    "half an O2. python3 -c \"...\" runs one line of Python, used here as a calculator:"
run "python3 -c \"print(($E1 - ($E2) - ($E3)) / 2 * 13.605693)\""
say "" "About -2.85 eV per ZnO. A published PBE value is about -2.8 eV; experiment is -3.63 eV." \
    "Agreeing with other PBE work means the set-up is right. The 0.8 eV gap to experiment" \
    "is the known error of the PBE approximation for oxides."

step "10. Can the number be trusted? A convergence test"
say "A numerical setting is 'converged' when making it more accurate no longer changes the" \
    "answer. Test the cut-off: make a copy of the input with 50 -> 60 Ry and run it." \
    "sed 's/OLD/NEW/' FILE prints FILE with OLD replaced by NEW; '>' saves it as a new file."
run "sed 's/ecutwfc     = 50/ecutwfc     = 60/; s/ecutrho     = 400/ecutrho     = 480/' 01_zno_scf.in > zno_e60.in"
say "" "diff shows the lines in which two files differ:"
run "diff 01_zno_scf.in zno_e60.in"
runpw zno_e60 "30 seconds"
run "grep '^!' 01_zno_scf.out zno_e60.out"
say "" "The total energy moved by a few thousandths of a Ry. Almost the same shift happens in" \
    "Zn and O2 and cancels in the subtraction: dH_f changes by only about 0.01 eV between" \
    "50 and 70 Ry (table in section 6 of README.md). Converge the DIFFERENCE you want."

step "11. Let the crystal relax (about 4 minutes)"
say "So far the atoms sat at the measured positions. 'vc-relax' lets pw.x move the atoms" \
    "and reshape the cell until forces and pressure vanish: the structure PBE itself prefers." \
    "What is different in this input:"
run "grep -n -E \"calculation|forc_conv|&IONS|&CELL|press_conv|ecut\" 04_zno_vcrelax.in"
if [ "$QUICK" = 1 ]; then
  say "" "(--quick: not running it now. To do it yourself, type" \
      "    mpirun -np $NP pw.x -in 04_zno_vcrelax.in > 04_zno_vcrelax.out )"
else
  runpw 04_zno_vcrelax "4 minutes"
  say "" "Every geometry step has its own converged energy. Watch it go down:"
  run "grep '^!' 04_zno_vcrelax.out"
  say "" "The relaxed cell and positions come after 'Begin final coordinates'" \
      "(-A 13 means: also print the 13 lines After the matching line):"
  run "grep -A 13 'Begin final coordinates' 04_zno_vcrelax.out"
  say "" "First number of the first row = a; last number of the third row = c." \
      "PBE gives a = 3.279 A and c = 5.286 A; measured: 3.250 and 5.207. About 1 % too" \
      "large, the usual error of PBE."
fi
say "" "Inputs 05 (Zn) and 06 (O2) are relaxed the same way, and 07 / 08 recompute the" \
    "energies at the relaxed structures (section 7 of README.md)."

step "12. The final numbers"
say "With all three structures relaxed and identical settings (60 Ry), the result is:"
if [ -f "$PROJ/SUMMARY.md" ]; then run "cat $PROJ/SUMMARY.md"; fi
say "" "That is the whole calculation. Your files are in $WORK/inputs." \
    "To repeat any step by hand: open Ubuntu, type   conda activate dft   and" \
    "   cd $WORK/inputs   and then the \$ lines above."
