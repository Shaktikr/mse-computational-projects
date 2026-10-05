"""creepml - machine learning for creep-rupture life prediction.

    data      dataset schema, loader for real data, SYNTHETIC superalloy generator
    features  physics-informed features (1/T, log sigma, Larson-Miller baseline ...)
    models    model zoo, grouped cross-validation, extrapolation split, conformal intervals
    design    inverse alloy design with a Gaussian-process surrogate
"""

__version__ = "0.1.0"
