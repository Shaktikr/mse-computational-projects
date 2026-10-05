"""pfsinter - phase-field model of solid-state sintering (Wang 2006 formulation).

    model     free energy, mobilities, semi-implicit spectral time stepping
    geometry  initial particle arrangements (two particles, random packings)
    analysis  neck radius, surface area, grain areas, neck-growth exponent
"""

from .model import Params, SinteringPF  # noqa: F401

__version__ = "0.1.0"
