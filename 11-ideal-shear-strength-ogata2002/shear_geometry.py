"""Geometry and linear elasticity of {111}<11-2> shear in an fcc crystal.

Used by 02_ideal_shear.py (to build the sheared cells and to precondition the stress relaxation)
and by the unit tests. Nothing here calls Quantum ESPRESSO.

Shear frame: x = <11-2> shear direction, z = (111) plane normal, y = z cross x.
The sign of x is chosen so that an engineering shear gamma = 1/sqrt(2) maps the fcc lattice onto
its twin (the "easy", twinning sense of <112> shear studied by Ogata, Li & Yip 2002).
"""
import numpy as np

ZC = np.array([1, 1, 1]) / np.sqrt(3)
XC = np.array([1, 1, -2]) / np.sqrt(6)
GAMMA_TWIN = 1 / np.sqrt(2)
# the five strain/stress components that are relaxed to zero stress in "pure" shear
COMPS = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2)]
VOIGT = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def fcc_primitive_cubic(a0):
    """1-atom fcc primitive cell, rows = lattice vectors, cubic axes (Angstrom)."""
    return 0.5 * a0 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], float)


def frame(xc):
    """Rotation matrix whose rows are the shear-frame axes x, y, z in cubic coordinates."""
    yc = np.cross(ZC, xc)
    return np.array([xc, yc, ZC])


def nn_spread(prim, R, g):
    """Spread (max - min) of the 12 shortest lattice vectors after shear g in frame R.
    Zero means the sheared lattice is again a perfect fcc lattice."""
    F = np.eye(3)
    F[0, 2] = g
    h = (prim @ R.T) @ F.T
    pts = [i * h[0] + j * h[1] + k * h[2] for i in range(-2, 3) for j in range(-2, 3) for k in range(-2, 3)]
    d = np.sort([np.linalg.norm(p) for p in pts if np.linalg.norm(p) > 1e-6])[:12]
    return float(np.ptp(d))


def twin_frame(a0):
    """Return (R, h0): the twinning-sense shear frame and the primitive cell expressed in it."""
    prim = fcc_primitive_cubic(a0)
    R = frame(XC)
    if nn_spread(prim, R, GAMMA_TWIN) > 1e-6:
        R = frame(-XC)
    assert nn_spread(prim, R, GAMMA_TWIN) < 1e-6
    return R, prim @ R.T


def cubic_stiffness(C11, C12, C44):
    """Fourth-rank stiffness tensor C_ijkl of a cubic crystal in its cubic axes."""
    d = np.eye(3)
    C = (C12 * np.einsum("ij,kl->ijkl", d, d)
         + C44 * (np.einsum("ik,jl->ijkl", d, d) + np.einsum("il,jk->ijkl", d, d)))
    for i in range(3):
        C[i, i, i, i] += C11 - C12 - 2 * C44
    return C


def rotate(C, R):
    """Express a fourth-rank tensor in the frame whose axes are the rows of R."""
    return np.einsum("ia,jb,kc,ld,abcd->ijkl", R, R, R, R, C)


def relaxation_stiffness(Cr):
    """d sigma_p / d D_q for the five relaxed components (D symmetric, so off-diagonal D counts twice)."""
    return np.array([[Cr[p[0], p[1], q[0], q[1]] * (1 if q[0] == q[1] else 2) for q in COMPS] for p in COMPS])


def to_voigt(Cr):
    """6x6 Voigt matrix (engineering shear strains) of a fourth-rank stiffness tensor."""
    return np.array([[Cr[a[0], a[1], b[0], b[1]] for b in VOIGT] for a in VOIGT])


def shear_moduli(C11, C12, C44, a0=4.0):
    """Linear-elastic {111}<11-2> shear moduli: (unrelaxed, relaxed) in the units of C.
    unrelaxed = C'_xzxz (all other strains zero) = (C11 - C12 + C44) / 3
    relaxed   = 1 / S'_55 (all other stresses zero) = 3 C44 (C11 - C12) / (C11 - C12 + 4 C44)"""
    R, _ = twin_frame(a0)
    Cr = rotate(cubic_stiffness(C11, C12, C44), R)
    return float(Cr[0, 2, 0, 2]), float(1 / np.linalg.inv(to_voigt(Cr))[4, 4])
