"""Periodic Voronoi polycrystals - a small, dependency-free alternative to Atomsk.

Each grain is a randomly oriented single crystal filling the Voronoi cell of its seed
point (minimum-image distances, so grains wrap across the periodic boundaries). Atoms
that end up too close to atoms of a neighbouring grain at the boundaries are removed.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from scipy.spatial import cKDTree

LATTICES = {
    # basis in fractional coordinates of the cubic cell, species index per basis atom
    "fcc": ([(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)], [0, 0, 0, 0]),
    "bcc": ([(0, 0, 0), (0.5, 0.5, 0.5)], [0, 0]),
    "b2": ([(0, 0, 0), (0.5, 0.5, 0.5)], [0, 1]),
}


def random_rotation(rng) -> np.ndarray:
    """Uniformly distributed random rotation matrix (from a random unit quaternion)."""
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _lattice_block(a, lattice, half_extent):
    basis, species = LATTICES[lattice]
    n = int(np.ceil(half_extent / a)) + 1
    g = np.arange(-n, n)
    I, J, K = np.meshgrid(g, g, g, indexing="ij")
    cells = np.stack([I.ravel(), J.ravel(), K.ravel()], axis=1).astype(float)
    pos = (cells[:, None, :] + np.array(basis)[None, :, :]).reshape(-1, 3) * a
    spec = np.tile(species, len(cells))
    return pos, spec


def voronoi_polycrystal(symbols, a: float, box: float, n_grains: int, lattice: str = "fcc",
                        seed: int = 0, min_dist_frac: float = 0.7) -> tuple[Atoms, np.ndarray]:
    """Cubic periodic box of side `box` (A) with `n_grains` grains.

    symbols: list of chemical symbols per species index, e.g. ['Ni'] or ['Co', 'Al'] for B2.
    Returns (atoms, grain_id per atom).
    """
    rng = np.random.default_rng(seed)
    seeds = rng.uniform(0, box, size=(n_grains, 3))
    nn = a / np.sqrt(2) if lattice == "fcc" else a * np.sqrt(3) / 2
    # all 27 periodic images of all seeds; an (unwrapped) lattice point belongs to grain g
    # only if its nearest seed image is the ORIGINAL seed g -> grains tile the box exactly
    shifts = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], float) * box
    images = (seeds[None, :, :] + shifts[:, None, :]).reshape(-1, 3)  # index = img * n + g
    i0 = int(np.where((shifts == 0).all(axis=1))[0][0])
    tree_seeds = cKDTree(images)
    pos_all, spec_all, gid_all = [], [], []
    half = 0.9 * box  # covers any Voronoi cell (the box half-diagonal is 0.87 box)
    base_pos, base_spec = _lattice_block(a, lattice, half)
    for g, s0 in enumerate(seeds):
        Rm = random_rotation(rng)
        p = base_pos @ Rm.T + s0
        _, owner = tree_seeds.query(p)
        keep = owner == i0 * n_grains + g
        pos_all.append(np.mod(p[keep], box))
        spec_all.append(base_spec[keep])
        gid_all.append(np.full(keep.sum(), g))
    pos = np.concatenate(pos_all)
    spec = np.concatenate(spec_all)
    gid = np.concatenate(gid_all)
    # remove one atom of every too-close pair (they sit at grain boundaries)
    tree = cKDTree(pos, boxsize=box)
    pairs = tree.query_pairs(min_dist_frac * nn, output_type="ndarray")
    drop = np.zeros(len(pos), bool)
    for i, j in pairs:
        if not drop[i] and not drop[j]:
            drop[j] = True
    pos, spec, gid = pos[~drop], spec[~drop], gid[~drop]
    atoms = Atoms([symbols[k] for k in spec], positions=pos, cell=[box, box, box], pbc=True)
    return atoms, gid


def write_lammps_data(atoms: Atoms, path, species_order):
    """LAMMPS 'atomic' data file with atom types in the order of `species_order`."""
    from ase.io import write

    write(path, atoms, format="lammps-data", specorder=species_order, atom_style="atomic", masses=True)
