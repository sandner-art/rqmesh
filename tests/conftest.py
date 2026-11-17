# SPDX-License-Identifier: Apache-2.0
"""Procedural test meshes (no data files)."""
import numpy as np
import pytest


def cube():
    V = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    F = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
    return V, F


def torus(n=32, m=16, R=1.0, r=0.4):
    V = np.array([[(R + r * np.cos(2 * np.pi * j / m)) * np.cos(2 * np.pi * i / n),
                   (R + r * np.cos(2 * np.pi * j / m)) * np.sin(2 * np.pi * i / n),
                   r * np.sin(2 * np.pi * j / m)] for i in range(n) for j in range(m)])
    F = [[i * m + j, ((i + 1) % n) * m + j, ((i + 1) % n) * m + (j + 1) % m, i * m + (j + 1) % m]
         for i in range(n) for j in range(m)]
    return V, F


def icosahedron():
    t = (1 + 5 ** 0.5) / 2
    V = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                  [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    T = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6],
         [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10],
         [8, 6, 7], [9, 8, 1]]
    return V / np.linalg.norm(V[0]), T


def plate_with_hole(n=16, hole=4):
    """n x n quad grid with a centred hole of hole x hole quads."""
    idx = {}
    V = []
    for i in range(n + 1):
        for j in range(n + 1):
            idx[i, j] = len(V); V.append([i / n, j / n, 0.0])
    lo, hi = (n - hole) // 2, (n + hole) // 2
    F = [[idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]] for i in range(n) for j in range(n)
         if not (lo <= i < hi and lo <= j < hi)]
    return np.array(V), F


def polar_annulus(na=24, nr=10, r0=0.3, r1=1.0):
    """Disk with a round hole, polar quads: every radial layer is a closed ring around the hole."""
    V = np.array([[(r0 + (r1 - r0) * k / nr) * np.cos(2 * np.pi * i / na), (r0 + (r1 - r0) * k / nr) * np.sin(2 * np.pi * i / na),
                   0.0] for k in range(nr + 1) for i in range(na)])
    F = [[k * na + i, k * na + (i + 1) % na, (k + 1) * na + (i + 1) % na, (k + 1) * na + i] for k in range(nr) for i in range(na)]
    return V, F


@pytest.fixture
def meshes():
    return dict(cube=cube, torus=torus, icosahedron=icosahedron, plate_with_hole=plate_with_hole, polar_annulus=polar_annulus)
