# 12 · Formation energy of ZnO from first principles — a from-scratch walkthrough

How much energy is released when zinc metal reacts with oxygen gas to form zinc oxide?

    Zn (metal)  +  ½ O₂ (gas)  →  ZnO (wurtzite crystal)

This project computes that number with density-functional theory (DFT) and checks it against
published calculations and against experiment. It is written as a tutorial for someone who has
never run DFT: it starts from a Windows laptop with nothing installed, gives every command,
explains every line of every input file, and can be done entirely by hand in a terminal with
free software (Quantum ESPRESSO). No cluster and no AI assistant are needed.

## Result

| | ΔH_f (eV per ZnO formula unit) |
|---|---|
| **This work — PBE, Quantum ESPRESSO 7.5, relaxed structures** | **−2.88** |
| This work — same, at the measured (unrelaxed) structures | −2.85 |
| Published GGA (PBE) value ¹ | −2.8 |
| Published hybrid / screened-exchange values ¹ | −3.1 to −3.3 |
| Published HSE value with 38 % exact exchange ² | −3.25 |
| Published diffusion Monte Carlo values ² | −3.94 and −4.15 |
| Experiment (298 K) | −3.63 |

¹ Clark et al. (2010); ² Santana et al. (2015); both in section 12. These values were read from
the text of the papers. The experimental value is the CODATA one listed in the NIST Chemistry
WebBook, −350.46 ± 0.27 kJ/mol.

Two different comparisons are made here, and they answer different questions.

* **Against other PBE calculations: "did I set the calculation up correctly?"** The approximation
  is the same, so the numbers should agree within a few hundredths of an eV. They do.
* **Against experiment: "how good is the approximation?"** PBE misses by about 0.7–0.8 eV. This is a
  known, systematic error of PBE for oxides (section 8), not a mistake in the calculation.

## Contents

1. [The question in plain words](#1-the-question-in-plain-words)
2. [What the program actually does](#2-what-the-program-actually-does)
3. [The tools](#3-the-tools)
4. [Setting up a PC from zero](#4-setting-up-a-pc-from-zero)
5. [The calculation by hand](#5-the-calculation-by-hand)
6. [Convergence: how do I know the numbers are right?](#6-convergence-how-do-i-know-the-numbers-are-right)
7. [Relaxing the structures and the final answer](#7-relaxing-the-structures-and-the-final-answer)
8. [Comparison with papers and experiment](#8-comparison-with-papers-and-experiment)
9. [The same thing automated](#9-the-same-thing-automated)
10. [When something goes wrong](#10-when-something-goes-wrong)
11. [Doing this for another material](#11-doing-this-for-another-material)
12. [Files, limitations, references](#12-files-limitations-references)

---

## 1. The question in plain words

A reaction releases energy when the products are more strongly bound than the reactants. The
**formation energy** of ZnO (strictly, the enthalpy of formation, ΔH_f) is the energy change when
one formula unit of ZnO is made from its elements in their normal forms: solid zinc metal and O₂
gas. It is negative because energy is released. The measured value is −350.5 kJ per mole of ZnO,
which is −3.63 eV per ZnO formula unit (1 eV per formula unit = 96.485 kJ/mol).

DFT does not simulate the reaction. What it computes is the **total energy of one given
arrangement of atoms** at zero temperature. So the job is three separate calculations — a crystal
of ZnO, a crystal of Zn, one O₂ molecule — followed by a subtraction:

    ΔH_f = E(ZnO) − E(Zn) − ½ E(O₂)

The bookkeeping matters. The wurtzite cell that is simulated holds 2 Zn + 2 O (two formula units)
and the hcp zinc cell holds 2 Zn atoms, so in terms of what the program prints:

    ΔH_f = E_cell(ZnO)/2 − E_cell(Zn)/2 − E(O₂)/2

Each E is a large negative number (hundreds of Ry) and the answer is a small difference between
them (0.2 Ry). That is why the three calculations must use *exactly* the same approximation and
the same numerical settings: errors that are identical in all three cancel in the difference.

## 2. What the program actually does

**The problem.** A material is nuclei plus electrons. For fixed nuclei, quantum mechanics fixes the
lowest possible energy of the electrons, but the exact equation (the many-electron Schrödinger
equation) cannot be solved for more than a handful of electrons.

**The idea of DFT.** Two results make the problem tractable. First, the lowest energy is
completely determined by the electron density n(r) — how many electrons there are at each point in
space — which is a far simpler object than the full wavefunction (Hohenberg and Kohn, 1964).
Second, that density can be found by solving equations for *independent* electrons that move in
an effective potential: the attraction of the nuclei, the average repulsion of all the other
electrons, and a correction term called exchange–correlation (Kohn and Sham, 1965).

**The self-consistent loop.** The effective potential depends on the density, and the density
depends on the potential. So the program guesses a density, builds the potential, solves for the
electron states, gets a new density, mixes it with the old one, and repeats until nothing changes.
This is the SCF (self-consistent field) cycle; each pass is one `iteration #` in the output.

**The one real approximation.** The exact exchange–correlation term is not known. This project uses
PBE (Perdew, Burke and Ernzerhof, 1996), the standard choice for solids. Everything else below is
numerics that can be made as accurate as you like; PBE cannot.

**The numerical settings.** Each one is a line in the input file, and each has to be tested:

| Idea | In plain words | Where it is in the input |
|---|---|---|
| Plane-wave cut-off | Electron states are written as sums of waves. The cut-off sets how many waves, i.e. how fine a detail can be described. Higher is more accurate and slower. | `ecutwfc`, `ecutrho` |
| k-points | A crystal repeats for ever, so its electron states are labelled by a wave-vector **k** and every property is an average over a grid of k values. Metals need dense grids. | `K_POINTS` |
| Pseudopotential | Inner-shell electrons take no part in bonding. They are replaced by an effective potential stored in one file per element. | the `.UPF` file named under `ATOMIC_SPECIES` |
| Smearing | In a metal the filled states end at a sharp edge (the Fermi level). Blurring that edge slightly makes the k-point average converge. | `occupations = 'smearing'`, `smearing`, `degauss` |
| Spin | Normally spin-up and spin-down electrons pair up. O₂ has two unpaired electrons, so the two spins have to be treated separately. | `nspin = 2`, `tot_magnetization` |
| Vacuum box | The program always assumes a repeating cell. A molecule is put alone in a big empty box so that its copies do not feel each other. | `CELL_PARAMETERS` |
| Relaxation | Moving the atoms (and reshaping the cell) until the forces (and the pressure) vanish: finding the structure the theory itself prefers. | `calculation = 'relax'` or `'vc-relax'` |

## 3. The tools

| Layer | What it is | Why it is needed |
|---|---|---|
| Windows 10/11 | the laptop's operating system | — |
| WSL 2 | a Windows feature that runs a real Linux system inside Windows | Quantum ESPRESSO is Linux software |
| Ubuntu | the Linux system; you work in its terminal window | every command below is typed there |
| Miniforge (`conda`) | a package manager that installs scientific software into separate "environments" | installs Quantum ESPRESSO with one command |
| Quantum ESPRESSO | the DFT code. `pw.x` is the main program (it does the SCF cycle and the relaxations); `ld1.x` builds pseudopotentials | does the physics |
| Open MPI (`mpirun`) | starts several copies of `pw.x` that share the work, one per CPU core | makes it 3–4 times faster on a 4-core laptop |
| A text editor (`nano`, or VS Code) | for writing input files | inputs are plain text |
| Python (optional) | automates the repetitive runs | section 9 |

Every calculation, however advanced, is the same three steps:
**write a text input file → run `pw.x` on it → read the text output file.**

## 4. Setting up a PC from zero

Done once. The results here were produced on Windows 10 with WSL 2, Ubuntu 26.04, Miniforge and
Quantum ESPRESSO 7.5 from conda-forge, on a 4-core laptop with 16 GB of RAM.

### 4.1 Install WSL 2 and Ubuntu (in Windows)

1. Check that virtualization is on: Task Manager → Performance → CPU → "Virtualization: Enabled".
   If it says Disabled, switch it on in the BIOS/UEFI settings of the laptop.
2. Open **PowerShell as Administrator** (Start menu → type "PowerShell" → right-click → Run as
   administrator) and type:

   ```powershell
   wsl --install -d Ubuntu
   ```

3. Restart Windows when it asks. Ubuntu then opens (or start "Ubuntu" from the Start menu) and
   asks you to invent a Linux username and password. Nothing appears on screen while you type the
   password; that is normal. This is the password `sudo` asks for later.
4. Check, in PowerShell, that the VERSION column says 2:

   ```powershell
   wsl -l -v
   ```

From here on, everything is typed in the **Ubuntu** window.

### 4.2 Update Ubuntu

```bash
sudo apt update && sudo apt upgrade -y
```

### 4.3 Install Miniforge (conda)

```bash
cd ~
```

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
```

```bash
bash Miniforge3-Linux-x86_64.sh
```

Press Enter to page through the licence, type `yes`, press Enter to accept the default location
(`~/miniforge3`), and answer `yes` when asked whether to initialise conda. Then close the Ubuntu
window and open it again. The prompt now starts with `(base)`.

### 4.4 Install Quantum ESPRESSO

```bash
conda create -n dft -c conda-forge qe openmpi numpy scipy matplotlib -y
```

This creates an environment called `dft` that contains `pw.x`, `ld1.x` and the other Quantum
ESPRESSO programs, `mpirun`, and Python with numpy/scipy/matplotlib.

### 4.5 Tell the programs to use one thread each (important)

```bash
echo 'export OMP_NUM_THREADS=1' >> ~/.bashrc
```

Close and reopen the Ubuntu window once more so the setting takes effect. Without it, each of the
4 `pw.x` processes started by `mpirun` tries to use 2 threads on a 4-core CPU and they get in each
other's way: on this laptop the first calculation below took **268 s instead of 23 s**.

### 4.6 Check the installation

```bash
conda activate dft
```

```bash
which pw.x mpirun ld1.x
```

Three paths inside `~/miniforge3/envs/dft/bin/` should be printed.

```bash
echo $OMP_NUM_THREADS
```

should print `1`.

### 4.7 At the start of every working session

Open Ubuntu and type `conda activate dft`. The prompt changes from `(base)` to `(dft)`. If you
forget, the shell answers `pw.x: command not found`.

### 4.8 Finding and editing your files

* Your Linux home folder is `~` (that is `/home/<your-username>`). Keep calculations there, not
  under `/mnt/c/...`, which is the Windows disk seen from Linux and is much slower.
* `explorer.exe .` opens the current Linux folder in Windows File Explorer.
* `nano file.in` edits a file in the terminal: type, then Ctrl+O and Enter to save, Ctrl+X to
  leave. If Visual Studio Code is installed on Windows (with its WSL extension), `code .` opens the
  current folder in it.
* Useful shell commands: `ls` (list files), `cd folder` (go into a folder), `cd ..` (go up),
  `pwd` (where am I), `cp a b` (copy), `mkdir name` (new folder), `less file` (read a file; `q`
  quits), `grep 'text' file` (print the lines of a file that contain a text).

## 5. The calculation by hand

**A guided version of this section runs on your own machine.** `walkthrough.sh` performs the
steps below one at a time in the terminal: it explains what the next command does, shows it,
waits for Enter, runs it, and explains the output.

```bash
bash ~/mse-computational-projects/12-zno-formation-energy/walkthrough.sh
```

It takes about ten minutes including one relaxation. `--quick` skips the relaxation and `--auto`
runs without pausing.

### Step 1 — make a practice folder

Work in a folder of your own, outside the repository, so that experiments never disturb it:

```bash
mkdir ~/zno-by-hand
```

```bash
cp -r ~/mse-computational-projects/12-zno-formation-energy/pseudo ~/mse-computational-projects/12-zno-formation-energy/inputs ~/zno-by-hand/
```

```bash
cd ~/zno-by-hand/inputs
```

`pseudo/` holds the pseudopotentials and `inputs/` holds the hand-written input files
`01_…in` to `08_…in`. Typing the inputs yourself instead of copying them is the best way to learn
them.

### Step 2 — pseudopotentials

One file per element, ending in `.UPF`. There are two ways to get them.

**(a) Generate them** with `ld1.x`, the atomic program that ships with Quantum ESPRESSO. The
recipes `O.in` and `Zn.in` in `pseudo/` come from the public *pslibrary* collection of A. Dal Corso:

```bash
cd ~/zno-by-hand/pseudo
```

```bash
ld1.x < O.in > O.ld1.out
```

```bash
ld1.x < Zn.in > Zn.ld1.out
```

Each takes about 3 seconds and writes `O.pbe-n-kjpaw_psl.0.1.UPF` and
`Zn.pbe-dn-kjpaw_psl.0.2.2.UPF` (plus scratch files `ld1.wfc`, `ld1ps.wfc`, `ld1.test`, `vx.pot`
that can be deleted). `bash make_pseudo.sh` does the same in one go.

**(b) Download tested files.** Ready-made `.UPF` files for every element are listed on the
Quantum ESPRESSO pseudopotential page (https://www.quantum-espresso.org/pseudopotentials/), and
the SSSP library on Materials Cloud (https://www.materialscloud.org/discover/sssp) is a tested
selection with a recommended cut-off for each element. Put the files in `pseudo/` and write their
exact names under `ATOMIC_SPECIES`.

How to read a pseudopotential's name, e.g. `Zn.pbe-dn-kjpaw_psl.0.2.2.UPF`: element `Zn`;
functional `pbe`; `d` = the 3d electrons are kept as valence electrons (12 valence electrons for
Zn); `n` = non-linear core correction; `kjpaw` = a PAW (projector-augmented-wave) dataset;
`psl.0.2.2` = pslibrary version. **The functional in the file name must be the functional you
want to use**, and the same files must be used in all three calculations.

Look inside one (it is a text file):

```bash
head -12 O.pbe-n-kjpaw_psl.0.1.UPF
```

The lines `Suggested minimum cutoff for wavefunctions: 47. Ry` (O) and `39. Ry` (Zn) give the
starting point for `ecutwfc`: take the larger, round up to 50 Ry, then test it (section 6).

Go back to the inputs folder:

```bash
cd ~/zno-by-hand/inputs
```

### Step 3 — the first input file: wurtzite ZnO

`inputs/01_zno_scf.in`, in full:

```text
&CONTROL
   calculation = 'scf'          ! single-point energy: atoms and cell stay where you put them
   prefix      = 'zno'          ! name used for the temporary files of this calculation
   outdir      = './tmp'        ! scratch folder (wavefunctions, charge density) - can be deleted later
   pseudo_dir  = '../pseudo'    ! folder that holds the .UPF pseudopotential files
   tprnfor     = .true.         ! also print the forces on the atoms
   tstress     = .true.         ! also print the stress tensor and the pressure
/
&SYSTEM
   ibrav       = 0              ! 0 = "I give the three lattice vectors myself" (CELL_PARAMETERS below)
   nat         = 4              ! atoms in the cell: 2 Zn + 2 O  =  2 formula units of ZnO
   ntyp        = 2              ! number of different elements
   ecutwfc     = 50             ! plane-wave cut-off for the wavefunctions (Ry) - from the convergence test
   ecutrho     = 400            ! cut-off for the charge density (Ry): 8 x ecutwfc for PAW pseudopotentials
   occupations = 'fixed'        ! ZnO has a band gap: every band is either completely full or empty
/
&ELECTRONS
   conv_thr    = 1.0d-9         ! stop the SCF loop when the estimated energy error is below this (Ry)
   mixing_beta = 0.5            ! fraction of the new charge density mixed in at each iteration
/
ATOMIC_SPECIES
#  symbol  mass (amu)  pseudopotential file
   Zn  65.38   Zn.pbe-dn-kjpaw_psl.0.2.2.UPF
   O   15.999  O.pbe-n-kjpaw_psl.0.1.UPF
CELL_PARAMETERS angstrom
#  a1 = (a, 0, 0)    a2 = (-a/2, a*sqrt(3)/2, 0)    a3 = (0, 0, c)    with a = 3.250 A, c = 5.207 A
   3.2500000000   0.0000000000   0.0000000000
  -1.6250000000   2.8145825623   0.0000000000
   0.0000000000   0.0000000000   5.2070000000
ATOMIC_POSITIONS crystal
   Zn  0.3333333333  0.6666666667  0.0000000000
   Zn  0.6666666667  0.3333333333  0.5000000000
   O   0.3333333333  0.6666666667  0.3820000000
   O   0.6666666667  0.3333333333  0.8820000000
K_POINTS automatic
   6 6 4  0 0 0
```

The file has two kinds of blocks.

* **Namelists** start with `&NAME` and end with `/`. They hold `keyword = value` settings.
  `&CONTROL` says what to do, `&SYSTEM` describes the material and the accuracy, `&ELECTRONS`
  steers the SCF loop. Text after `!` is a comment.
* **Cards** come after the namelists and hold tables: the elements and their pseudopotentials,
  the three lattice vectors, the atom positions, the k-point grid. **Inside a card a comment line
  must start with `#`.** A line starting with `!` there stops `pw.x` 7.5 with
  `cannot read atomic specie from: ! ...`.

Where the numbers come from:

* **The structure** (lattice vectors and atom positions) is the measured one. Wurtzite ZnO is
  hexagonal with a = 3.250 Å, c = 5.207 Å; Zn sits at (1/3, 2/3, 0) and (2/3, 1/3, 1/2) in units of
  the lattice vectors and each O sits a fraction u = 0.382 of c above a Zn. For any material these
  come from a crystallographic database or a paper (section 11).
* **The cut-offs** start from the suggestion inside the pseudopotential files and are then tested.
* **The k-point grid** is tested (section 6).
* **The masses** are not used in an energy calculation, but the format requires them.

### Step 4 — run it

```bash
mpirun -np 4 pw.x -in 01_zno_scf.in > 01_zno_scf.out
```

* `mpirun -np 4` starts 4 copies of the program that share the work. Use the number of
  **physical cores** of the CPU (4 here), not the number of threads Windows shows (8).
* `pw.x` is the program; `-in 01_zno_scf.in` is the input file.
* `> 01_zno_scf.out` sends everything the program prints into a file instead of the screen.

Nothing appears on screen while it runs; the prompt comes back after about 25 seconds. To watch
a long run from a second Ubuntu window, use `tail -f 01_zno_scf.out` (Ctrl+C stops watching).

### Step 5 — read the output

The output is a text file of a few hundred lines. Check four things, in this order.

```bash
tail -4 01_zno_scf.out
```

must show `JOB DONE.` — the run ended normally.

```bash
grep 'convergence has been achieved' 01_zno_scf.out
```

must print `convergence has been achieved in  13 iterations` — the SCF loop converged.

```bash
grep '^!' 01_zno_scf.out
```

prints the result, the converged total energy (the line is marked with `!` so it is easy to find):

    !    total energy              =    -571.13431252 Ry

```bash
grep 'P=' 01_zno_scf.out
```

prints the pressure, `P=       36.33` (in kbar; 10 kbar = 1 GPa).

Reading the whole file once (`less 01_zno_scf.out`) is worth it. Things to recognise:

| In the output | What it tells you |
|---|---|
| `number of atoms/cell = 4`, `number of electrons = 36.00` | pw.x understood the structure. 36 = 2 × 12 (Zn) + 2 × 6 (O) valence electrons |
| `number of k points= 21` | the 6×6×4 = 144 grid points reduce to 21 different ones by symmetry |
| `iteration # 1 … total energy = -570.81267997 Ry`, `estimated scf accuracy < 0.84334722 Ry` | first pass of the SCF loop; the accuracy estimate shrinks at every pass |
| `!    total energy = -571.13431252 Ry` | the converged energy of the 4-atom cell |
| `Total force = 0.002615` (Ry/bohr) | forces on the atoms are tiny: the measured positions are almost PBE's own |
| `P= 36.33` (kbar) | positive pressure: the cell is slightly too small for PBE, which would like it larger |

Messages that look alarming and are not: `using ibrav=0 with symmetry is DISCOURAGED` (a
reminder that the cell was typed in by hand; harmless when the vectors are given to 10 digits),
`wavefunction(s) 4S 3D renormalized`, `Check: negative core charge=`, and, for O₂,
`negative rho (up, down)` with values around 10⁻⁴.

**Units.** `pw.x` prints energies in rydberg (Ry) and lengths in bohr: 1 Ry = 13.605693 eV,
1 bohr = 0.529177 Å.

**A total energy means nothing on its own.** Its zero depends on the pseudopotentials. Only
differences between calculations done with the same pseudopotentials and settings are physical.

### Step 6 — the metal: hcp zinc

`inputs/02_zn_scf.in` differs from the ZnO input in three places:

```text
   occupations = 'smearing'     ! metal: bands cross the Fermi level, so occupations are fractional
   smearing    = 'mv'           ! Marzari-Vanderbilt "cold" smearing
   degauss     = 0.02           ! smearing width (Ry)
```

a different cell and atom list (2 Zn atoms, measured a = 2.665 Å, c = 4.947 Å), and a much denser
k-point grid (`16 16 8`). The cut-offs are the same as for ZnO, on purpose.

```bash
mpirun -np 4 pw.x -in 02_zn_scf.in > 02_zn_scf.out
```

```bash
grep '^!' 02_zn_scf.out
```

    !    total energy              =    -487.67034605 Ry

A metal's output has a line `the Fermi energy is    11.1057 ev` where an insulator's has
`highest occupied level`.

If a metal is run with `occupations = 'fixed'` by mistake, `pw.x` does not complain when the
number of electrons is even: it silently treats the metal as an insulator and returns a wrong
energy. Decide metal or insulator yourself, before running.

### Step 7 — the molecule: O₂

`inputs/03_o2_scf.in` has three new things.

```text
   nspin       = 2              ! spin-polarised: spin-up and spin-down electrons treated separately
   tot_magnetization = 2        ! O2 has 2 unpaired electrons (triplet ground state): N_up - N_down = 2
   starting_magnetization(1) = 0.5   ! starting guess for the spin polarisation of species 1 (O)
```

The cell is an empty box about 10 Å wide with the molecule in the middle, and the k-point card is
`K_POINTS gamma` (a single k-point, which is all an isolated molecule needs).

```bash
mpirun -np 4 pw.x -in 03_o2_scf.in > 03_o2_scf.out
```

```bash
grep '^!' 03_o2_scf.out
```

    !    total energy              =     -83.04473183 Ry

```bash
grep 'total magnetization' 03_o2_scf.out | tail -1
```

must print `total magnetization       =     2.00 Bohr mag/cell`. Leaving the spin out is the
classic mistake in oxide formation energies: with the three spin lines removed, `pw.x` here did not converge at all
(`convergence NOT achieved after 100 iterations: stopping`), and with other settings such a run
can settle into a non-magnetic state whose energy is too high, which silently corrupts ΔH_f.

### Step 8 — the first answer

All three energies in one command:

```bash
grep '^!' 01_zno_scf.out 02_zn_scf.out 03_o2_scf.out
```

Then the subtraction, with Python as a calculator:

```bash
python3 -c "print((-571.13431252 - (-487.67034605) - (-83.04473183)) / 2 * 13.605693)"
```

    -2.8519889...

So at the measured structures, **ΔH_f = −2.85 eV per ZnO**. The division by 2 is the bookkeeping
of section 1; the factor 13.605693 converts Ry to eV. Sections 6 and 7 check how far this number
can be trusted and refine it.

## 6. Convergence: how do I know the numbers are right?

The cut-off, the k-point grids and the size of the O₂ box are numerical settings. A setting is
*converged* when making it more accurate no longer changes the answer. The test is always the same:
change one setting, rerun, compare.

By hand, for the cut-off of ZnO: copy the input, change the two cut-off numbers, run, read.

```bash
cp 01_zno_scf.in zno_e60.in
```

```bash
nano zno_e60.in
```

(set `ecutwfc = 60` and `ecutrho = 480`, save with Ctrl+O, Enter, leave with Ctrl+X)

```bash
mpirun -np 4 pw.x -in zno_e60.in > zno_e60.out
```

```bash
grep '^!' zno_e60.out
```

Once this is routine, a loop does the copying and editing:

```bash
for e in 40 50 60 70; do sed "s/ecutwfc     = 50/ecutwfc     = $e/; s/ecutrho     = 400/ecutrho     = $((8*e))/" 01_zno_scf.in > zno_e$e.in; mpirun -np 4 pw.x -in zno_e$e.in > zno_e$e.out; done
```

```bash
grep '^!' zno_e*.out
```

**Converge the quantity you want, not the total energies.** The total energies below still change
by several mRy between 50 and 70 Ry, but the same change happens in all three systems and cancels
in ΔH_f. The numbers (all at the measured structures; `01_convergence.py` produces them):

| `ecutwfc` (Ry) | E(ZnO cell) (Ry) | E(Zn cell) (Ry) | E(O₂) (Ry) | ΔH_f (eV) |
|---|---|---|---|---|
| 40 | −571.13232 | −487.66842 | −83.03901 | −2.891 |
| 50 | −571.13431 | −487.67035 | −83.04473 | −2.852 |
| 60 | −571.13966 | −487.67285 | −83.04566 | −2.865 |
| 70 | −571.13807 | −487.67185 | −83.04693 | −2.852 |

ΔH_f moves by 0.04 eV from 40 to 50 Ry and then stays within about ±0.01 eV. That ±0.01 eV is the
numerical uncertainty of this calculation; it is far smaller than the 0.7 eV error of PBE itself.

| ZnO k-grid (50 Ry) | E per ZnO (eV) | | Zn k-grid (50 Ry) | E per Zn atom (eV) |
|---|---|---|---|---|
| 4×4×2 | −3885.3260 | | 12×12×6 | −3317.5336 |
| 6×6×4 | −3885.3391 | | 16×16×8 | −3317.5465 |
| 8×8×5 | −3885.3401 | | 20×20×10 | −3317.5385 |
| | | | 24×24×12 | −3317.5415 |

The insulator is converged to 1 meV with a 6×6×4 grid. The metal still wobbles by a few meV at
20×20×10: metals converge slowly with k-points, which is why they need smearing and dense grids.

| O₂ box edge (Å, 50 Ry) | E per O atom (eV) |
|---|---|
| 8 | −564.9404 |
| 10 | −564.9406 |
| 12 | −564.9406 |

A 10 Å box is ample.

Settings used for the final numbers: `ecutwfc = 60`, `ecutrho = 480`, ZnO grid 8×8×5, Zn grid
20×20×10, O₂ box 10 Å.

## 7. Relaxing the structures and the final answer

So far the atoms sat at the measured positions. PBE has its own idea of the best structure, and
the consistent thing is to let each system relax to it. Three more runs
(`inputs/04_zno_vcrelax.in`, `05_zn_vcrelax.in`, `06_o2_relax.in`):

```bash
mpirun -np 4 pw.x -in 04_zno_vcrelax.in > 04_zno_vcrelax.out
```

```bash
mpirun -np 4 pw.x -in 05_zn_vcrelax.in > 05_zn_vcrelax.out
```

```bash
mpirun -np 4 pw.x -in 06_o2_relax.in > 06_o2_relax.out
```

What is new in these inputs: `calculation = 'vc-relax'` (atoms and cell both free) for the two
crystals and `'relax'` (atoms only) for the molecule, plus the namelists `&IONS` and `&CELL` that
these modes require. `pw.x` now repeats "SCF → forces and stress → move atoms, reshape cell" until
forces and pressure are below the thresholds. The relaxed structure is printed near the end:

```bash
grep -A 13 'Begin final coordinates' 04_zno_vcrelax.out
```

For ZnO the block reads:

    CELL_PARAMETERS (angstrom)
       3.278580741   0.000000000   0.000000000
      -1.639290370   2.839334210   0.000000000
      -0.000000000  -0.000000000   5.286280828

    ATOMIC_POSITIONS (crystal)
    Zn               0.3333333333        0.6666666667        0.0013029674
    Zn               0.6666666667        0.3333333333        0.5013029674
    O                0.3333333333        0.6666666667        0.3806970326
    O                0.6666666667        0.3333333333        0.8806970326

The first number of the first row is a, the last number of the third row is c, and u is the z
coordinate of an O minus that of the Zn below it (0.3807 − 0.0013 = 0.3794). `grep '^!'` on a
relaxation output prints one energy per geometry step; it goes down from step to step.

| | This work (PBE) | Measured | Difference |
|---|---|---|---|
| ZnO a (Å) | 3.279 | 3.250 | +0.9 % |
| ZnO c (Å) | 5.286 | 5.207 | +1.5 % |
| ZnO u | 0.3794 | 0.382 | −0.7 % |
| Zn a (Å) | 2.628 | 2.665 | −1.4 % |
| Zn c (Å) | 5.148 | 4.947 | +4.1 % |
| O₂ bond length (Å) | 1.222 | 1.208 | +1.1 % |

ZnO and O₂ show the usual behaviour of PBE: bonds about 1 % too long (Clark et al. also report GGA
lattice parameters of ZnO 1 % too large). Zinc metal is the odd one out: c/a comes out as 1.96,
against 1.86 measured at room temperature (1.83 extrapolated to 0 K) and 1.89–1.90 in published
PBE calculations. The reason is that the energy of hcp Zn is an almost flat function of c/a. With
the settings used here, the measured cell lies only 3.2 meV per atom above the relaxed one, and
the published PBE cell (c/a = 1.902) only 1.5 meV per atom above it. The value of c/a is therefore
very sensitive to details such as the pseudopotential and the k-point grid, while its effect on
ΔH_f is about 0.002 eV.

**Final energies.** A variable-cell run changes the cell while it runs, so the clean way to get
the final energy is one more single-point calculation at the relaxed structure: copy the relaxed
cell and positions into a copy of the SCF input. `inputs/07_zno_final_scf.in` and
`08_zn_final_scf.in` are exactly that. For O₂ the box did not change, so the last `!` line of
`06_o2_relax.out` is already the final energy.

```bash
mpirun -np 4 pw.x -in 07_zno_final_scf.in > 07_zno_final_scf.out
```

```bash
mpirun -np 4 pw.x -in 08_zn_final_scf.in > 08_zn_final_scf.out
```

```bash
grep '^!' 07_zno_final_scf.out 08_zn_final_scf.out
```

```bash
grep '^!' 06_o2_relax.out | tail -1
```

| System | Final total energy (Ry) |
|---|---|
| ZnO cell (2 formula units) | −571.14165872 |
| Zn cell (2 atoms) | −487.67214154 |
| O₂ molecule | −83.04602393 |

```bash
python3 -c "print((-571.14165872 - (-487.67214154) - (-83.04602393)) / 2 * 13.605693)"
```

prints −2.88096, so **ΔH_f = −2.88 eV per ZnO** (−278 kJ/mol, or −1.44 eV per atom). The numerical
uncertainty from the settings is about ±0.01–0.02 eV (section 6). Relaxing the structures changed
the answer by less than 0.02 eV.

## 8. Comparison with papers and experiment

| Method | ΔH_f (eV per ZnO) | Source |
|---|---|---|
| PBE, this work | −2.88 | section 7 |
| GGA (PBE) | −2.8 | Clark et al. |
| Hybrid and screened-exchange functionals | −3.1 to −3.3 | Clark et al. |
| HSE with 38 % exact exchange | −3.25 | Santana et al. |
| Diffusion Monte Carlo (two variants) | −3.94 and −4.15 | Santana et al. |
| Experiment, 298 K | −3.63 | quoted by both |

**Is the calculation validated?** Against PBE, yes: −2.88 eV agrees with the published −2.8 eV
(which is quoted to one decimal) within 0.1 eV, and the error against experiment, 0.75 eV, is the
size others report for this kind of functional.

**Why PBE misses experiment.** Two systematic errors, both documented for oxides:

1. *The O₂ molecule.* Semilocal functionals such as PBE bind O₂ too strongly. E(O₂) comes out
   too low, and because ½ E(O₂) is subtracted, every oxide formation energy comes out too small
   in magnitude.
2. *Localised d electrons.* The Zn 3d electrons suffer from the self-interaction error of
   semilocal functionals, which adds a further error on the oxide side.

The usual remedies are a fitted correction to the O₂ energy or fitted elemental reference
energies (Stevanović et al.), DFT+U (Mutter et al.; Artrith et al.), and hybrid functionals, which
cost far more and still leave 0.3–0.5 eV here. Even diffusion Monte Carlo misses, by 0.3–0.5 eV in
the other direction: this energy difference is hard for every method because it combines an
insulator, a metal and a molecule, whose errors do not cancel (Santana et al.).

**Temperature.** The computed number is for motionless atoms at 0 K; the measured one is an
enthalpy at 298 K. For ZnO the zero-point and thermal corrections should amount to a few
hundredths of an eV (an estimate, not computed here), so they do not change the picture.

## 9. The same thing automated

Typing forty runs by hand is slow and error-prone, so the project also contains small Python
scripts that write the same input files, run `pw.x`, and collect the numbers.

| File | What it does |
|---|---|
| `qe.py` | writes a `pw.x` input from a structure and a few settings, runs it with `mpirun`, reads energy, pressure, relaxed cell and positions from the output |
| `01_convergence.py` | the convergence tables of section 6 → `results_convergence.json` |
| `02_relax.py` | the three relaxations and the final single-point energies → `results_relax.json` |
| `03_formation_energy.py` | ΔH_f and the comparison table → `SUMMARY.md`, `results_formation_energy.json` |
| `run_all.sh` | runs the three scripts in order |

```bash
conda activate dft
```

```bash
cd ~/mse-computational-projects/12-zno-formation-energy
```

```bash
./run_all.sh
```

Every input and output file lands in `runs/`. A run whose output already contains `JOB DONE` is
not repeated, so after an interruption (laptop asleep, Ctrl+C) the same command continues where it
stopped. The whole project takes about 40 minutes on 4 cores.

## 10. When something goes wrong

The messages below were produced on purpose with this installation.

| What you see | Cause | Fix |
|---|---|---|
| `pw.x: command not found` | the conda environment is not active | `conda activate dft` |
| The run takes minutes instead of seconds; the output says `Threads/MPI process: 2` | `OMP_NUM_THREADS` is not 1 | section 4.5 |
| `Error in routine readpp (1): file ../pseudo/….UPF not found` | `pseudo_dir` is wrong, or the file name under `ATOMIC_SPECIES` has a typo | fix the path or the name; `ls ../pseudo` |
| `Error in routine read_namelists (1): bad line in namelist &system: "…"` | a misspelt keyword, or a missing `/` at the end of the namelist before | compare with a working input |
| `Error in routine card_atomic_species (5010): cannot read atomic specie from: ! …` | a comment line starting with `!` inside a card | start card comments with `#` |
| `Error in routine read_cards (4): wrong number of columns in ATOMIC_POSITIONS` | fewer atom lines than `nat` says | make `nat` and the list agree |
| `Error in routine read_input (1): opening input file` | the file named after `-in` does not exist in this folder | `ls`, check the name |
| `convergence NOT achieved after 100 iterations: stopping` | the SCF loop did not settle. Seen here for O₂ with the spin lines left out | put the spin settings back; otherwise lower `mixing_beta` (0.3) and check the structure for atoms that are too close |
| The energy is far from the expected one, with no error | wrong structure, a metal run with `occupations = 'fixed'`, O₂ without spin, different pseudopotentials or cut-offs in the three runs | check each against the inputs here |

If the answer is wrong by a factor rather than by a little, it is almost always bookkeeping:
energy per cell instead of per formula unit, a forgotten ½ for O₂, Ry not converted to eV.

## 11. Doing this for another material

The same recipe works for any compound A_xB_y.

1. **Decide what to compare with.** Find a paper that reports the number, and read its
   "computational details": functional, pseudopotentials or code, cut-off, k-points.
2. **Get the structures** of the compound and of each element in its standard state.
   Free sources are the Materials Project (https://materialsproject.org, calculated structures)
   and the Crystallography Open Database (https://www.crystallography.net/cod/, measured
   structures); both give CIF files with the lattice parameters and the atomic positions.
3. **Get pseudopotentials** for every element, all for the same functional. See step 2 of
   section 5.
4. **Write one input per system**, starting from the closest example here: insulator
   (`01_zno_scf.in`), metal (`02_zn_scf.in`), molecule (`03_o2_scf.in`). Magnetic elements
   (Fe, Co, Ni, Mn, Cr) need `nspin = 2`.
5. **Converge** cut-off and k-points on the energy *difference*.
6. **Relax** each structure, then compute the final energies with identical settings.
7. **Subtract**, per formula unit, and compare.

## 12. Files, limitations, references

| Path | What it is |
|---|---|
| `inputs/01_…in` – `08_…in` | hand-written, fully commented `pw.x` inputs for sections 5–7 |
| `pseudo/O.in`, `pseudo/Zn.in`, `pseudo/make_pseudo.sh` | pseudopotential recipes (pslibrary) and the command that builds the `.UPF` files |
| `walkthrough.sh` | guided, step-by-step run of section 5 in the terminal |
| `qe.py`, `01_convergence.py`, `02_relax.py`, `03_formation_energy.py`, `run_all.sh` | the automated version |
| `results_*.json`, `SUMMARY.md` | the numbers |
| `tests/` | unit tests that run without `pw.x` (`pytest 12-zno-formation-energy/tests` from the repository root) |
| `runs/` | every input and output of the automated runs (not under version control) |

### Limitations

* The published ZnO values in the Result table and in section 8 were read from the text of the two
  papers. The statement that PBE binds O₂ too strongly rests on the abstracts of Mutter et al. and
  Artrith et al.; no number for it is quoted here.
* The numerical uncertainty is about ±0.01–0.02 eV: ΔH_f wobbles by ±0.01 eV with the cut-off
  (`ecutrho` was kept at 8 × `ecutwfc` and not converged separately) and the zinc energy by a few
  meV with the k-point grid.
* One pair of pseudopotentials was used. Repeating the calculation with an independent set (for
  example from SSSP) would show how much the answer depends on that choice.
* The pslibrary recipe used for Zn (version 0.2.2) is commented out in the current pslibrary
  collection, so it is not that library's present default for Zn.
* The c/a of zinc found here (1.96) is larger than in published PBE calculations (1.89–1.90).
  The two cells differ by 1.5 meV per atom with the present settings, so ΔH_f is not affected,
  but the structure of zinc itself should not be quoted from this project without a check with
  a second pseudopotential and a denser k-point grid.
* Static 0 K energies: no zero-point or thermal contributions.

### References

* S. J. Clark, J. Robertson, S. Lany and A. Zunger, "Intrinsic defects in ZnO calculated by
  screened exchange and hybrid density functionals", *Phys. Rev. B* **81**, 115311 (2010),
  doi:10.1103/PhysRevB.81.115311 — heat of formation of ZnO with GGA (PBE), hybrid and
  screened-exchange functionals.
* J. A. Santana, J. T. Krogel, J. Kim, P. R. C. Kent and F. A. Reboredo, "Structural stability and
  defect energetics of ZnO from diffusion quantum Monte Carlo", *J. Chem. Phys.* **142**, 164705
  (2015), doi:10.1063/1.4919242, arXiv:1406.3169.
* NIST Chemistry WebBook, zinc oxide, condensed-phase thermochemistry: ΔfH°(solid) =
  −350.46 ± 0.27 kJ/mol, from J. D. Cox, D. D. Wagman and V. A. Medvedev, *CODATA Key Values for
  Thermodynamics* (Hemisphere, 1984). The O₂ bond length, 1.20752 Å, is from K. P. Huber and
  G. Herzberg, *Constants of Diatomic Molecules* (1979), also via the WebBook.
* "Implementation of self-consistent MGGA functionals in augmented plane wave based methods",
  arXiv:2112.13145, Table VII; and E. T. Marcial, L. Chaudhary, O. Gorbunova and A. N. Kolmogorov,
  "Performance of universal machine learning potentials in global optimization of inorganic
  crystal structures", arXiv:2602.23515 — the c/a ratio of hcp Zn with PBE and from experiment.
* V. Stevanović, S. Lany, X. Zhang and A. Zunger, *Phys. Rev. B* **85**, 115104 (2012) — fitted
  elemental-phase reference energies.
* D. Mutter et al., *Materials* (2020) — formation energies of transition-metal oxides with DFT+U.
* N. Artrith et al., *Phys. Rev. Materials* (2022) — corrections for oxide thermochemistry.
* J. P. Perdew, K. Burke and M. Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996) — the PBE functional.
* P. Giannozzi et al., *J. Phys.: Condens. Matter* **21**, 395502 (2009) and **29**, 465901 (2017) —
  Quantum ESPRESSO.
* A. Dal Corso, pslibrary, https://github.com/dalcorso/pslibrary — the pseudopotential recipes.
* D. S. Sholl and J. A. Steckel, *Density Functional Theory: A Practical Introduction* (Wiley);
  F. Giustino, *Materials Modelling using Density Functional Theory* (Oxford University Press).
* Quantum ESPRESSO input reference: https://www.quantum-espresso.org/Doc/INPUT_PW.html
