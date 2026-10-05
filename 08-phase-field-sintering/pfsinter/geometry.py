"""Initial microstructures."""

from __future__ import annotations

import numpy as np


def _smooth_disk(X, Y, xc, yc, R, width=2.0):
    d = np.sqrt((X - xc) ** 2 + (Y - yc) ** 2)
    return 0.5 * (1 - np.tanh((d - R) / width))


def two_particles(nx=256, ny=128, R=30.0, overlap=1.0):
    """Two equal circles touching on the x axis (slight overlap seeds the neck)."""
    y, x = np.mgrid[0:ny, 0:nx].astype(float)
    xc1 = nx / 2 - R + overlap / 2
    xc2 = nx / 2 + R - overlap / 2
    e1 = _smooth_disk(x, y, xc1, ny / 2, R)
    e2 = _smooth_disk(x, y, xc2, ny / 2, R)
    rho = np.clip(e1 + e2, 0, 1)
    return rho, np.stack([e1, e2]), {"centres": [(xc1, ny / 2), (xc2, ny / 2)], "R": R}


def random_packing(nx=256, ny=256, n=14, r_mean=22.0, r_std=4.0, seed=0, max_tries=20000, contact_overlap=2.0):
    """Connected random aggregate of circles (a 2D 'powder agglomerate').

    The first particle sits at the box centre; each new particle is placed touching a
    randomly chosen existing particle (overlap `contact_overlap` pixels, which seeds a
    small neck) and accepted only if it does not overlap any other particle and stays
    inside the box. Every particle therefore has at least one contact, as in a green
    compact.
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:ny, 0:nx].astype(float)
    r0 = max(rng.normal(r_mean, r_std), 0.6 * r_mean)
    circles = [(nx / 2, ny / 2, r0)]
    tries = 0
    while len(circles) < n and tries < max_tries:
        tries += 1
        r = max(rng.normal(r_mean, r_std), 0.6 * r_mean)
        a, b, ra = circles[rng.integers(len(circles))]
        ang = rng.uniform(0, 2 * np.pi)
        d = ra + r - contact_overlap
        xc, yc = a + d * np.cos(ang), b + d * np.sin(ang)
        if not (r + 2 < xc < nx - r - 2 and r + 2 < yc < ny - r - 2):
            continue
        if all(np.hypot(xc - c, yc - e) >= rr + r - contact_overlap - 1e-6 for c, e, rr in circles):
            circles.append((xc, yc, r))
    etas = np.stack([_smooth_disk(x, y, a, b, r) for a, b, r in circles])
    rho = np.clip(etas.sum(0), 0, 1)
    return rho, etas, {"circles": circles}
