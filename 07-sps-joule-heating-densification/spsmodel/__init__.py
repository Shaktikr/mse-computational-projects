"""spsmodel - spark plasma sintering (SPS / FAST) modelling.

    materials        temperature-dependent properties of graphite tooling and powder compacts
    electrothermal   axisymmetric finite-volume model of Joule heating in the SPS stack with
                     PID temperature control (pyrometer on the die surface)
    densification    creep-based densification kinetics (Bernard-Granger & Guizard),
                     extraction of n, Q and an effective diffusivity, master sintering curve
"""

__version__ = "0.1.0"
