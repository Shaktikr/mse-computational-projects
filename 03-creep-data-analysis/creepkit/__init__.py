"""creepkit - analysis of constant-load / constant-stress creep tests.

    curves        strain-rate extraction, minimum creep rate, creep stages, theta projection
    constitutive  Norton / Arrhenius / global power-law fits, threshold stress,
                  Larson-Miller, Monkman-Grant, Zener-Hollomon (sinh law)
    io            read / write creep curves stored as CSV with a metadata header
    synthetic     physically consistent synthetic creep curves for testing and teaching
"""

from .curves import CreepTest, minimum_creep_rate, strain_rate, creep_stages, fit_theta_projection  # noqa: F401
from . import constitutive, io, synthetic  # noqa: F401

__version__ = "0.1.0"
