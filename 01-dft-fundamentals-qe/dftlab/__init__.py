"""dftlab - a small, readable toolkit for learning plane-wave DFT with Quantum ESPRESSO + ASE.

Modules
-------
config        paths to pseudopotentials and the pw.x command
calculators   factory functions that build a Quantum ESPRESSO (or EMT test) calculator
structures    fcc / bcc / B2 cells used throughout the tutorials
convergence   cut-off energy and k-point convergence tests
eos           Birch-Murnaghan equation of state fitting
elastic       cubic elastic constants (C11, C12, C44) from energy-strain curves
"""

__version__ = "0.1.0"
