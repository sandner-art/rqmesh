# SPDX-License-Identifier: Apache-2.0
import importlib, os, subprocess, sys, collections
import numpy as np
import pytest

from conftest import cube, torus, icosahedron, plate_with_hole, polar_annulus

MODULES = ['analyze', 'compat', 'crease_split', 'defofield', 'features', 'isolation', 'meshio_min', 'partial', 'repair',
           'rosy', 'scalefield', 'skin', 'strip_bounds', 'structsimp', 'subdiv', 'subdiv_tri', 'symmetry', 'topofix',
           'transitions']


def _valence(F):
    nb = collections.defaultdict(set); e = collections.Counter()
    for f in F:
        for k in range(len(f)):
            a, b = f[k], f[(k + 1) % len(f)]; nb[a].add(b); nb[b].add(a); e[min(a, b), max(a, b)] += 1
    return nb, e


def test_version_and_imports():
    import rqmesh
    assert rqmesh.__version__
    for m in MODULES:
        importlib.import_module('rqmesh.' + m)


def test_numpy_only():
    """Blender's Python has NumPy but no SciPy: the core must work without it."""
    code = ("import sys; sys.modules['scipy'] = None\n"
            "import numpy as np\n"
            "from rqmesh import subdiv, structsimp, transitions, isolation, topofix\n"
            "V = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)\n"
            "F = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]\n"
            "Vs, Fs = subdiv.resubdivide(V, F, 'cc', 2)\n"
            "r = subdiv.unsubdivide(Vs, Fs)\n"
            "assert len(r['cage'][1]) == 6, len(r['cage'][1])\n"
            "from rqmesh.skin import SkinField\n"
            "W = np.stack([(V[:, 0] + 1) / 2, (1 - V[:, 0]) / 2], 1)\n"
            "assert np.allclose(SkinField(V, F, W).at(V), W)\n"
            "print('ok')\n")
    env = dict(os.environ, RQM_NO_SCIPY='1')
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env)
    assert out.returncode == 0 and 'ok' in out.stdout, out.stderr[-2000:]


def test_obj_roundtrip(tmp_path):
    from rqmesh.meshio_min import read_obj, write_obj
    V, F = torus(8, 6)
    p = tmp_path / 't.obj'; write_obj(str(p), V, F)
    V2, F2 = read_obj(str(p))
    assert np.allclose(np.asarray(V2), V, atol=1e-6) and [list(f) for f in F2] == F


@pytest.mark.parametrize('scheme', ['cc', 'simple'])
def test_base_cage_recovery_quads(scheme):
    from rqmesh import subdiv
    V, F = cube()
    Vs, Fs = subdiv.resubdivide(V, F, scheme, 2)
    assert len(Fs) == 6 * 16
    r = subdiv.unsubdivide(Vs, Fs)
    Vc, Fc = r['cage']
    assert len(Fc) == 6 and len(r['levels']) == 2
    Vr, Fr = subdiv.resubdivide(Vc, Fc, scheme, 2)
    diag = np.linalg.norm(Vs.max(0) - Vs.min(0))
    # same topology up to vertex order: compare sorted positions
    a = np.sort(np.round(np.asarray(Vr) / diag, 6), axis=0); b = np.sort(np.round(np.asarray(Vs) / diag, 6), axis=0)
    assert np.abs(a - b).max() < 1e-5


@pytest.mark.parametrize('scheme', ['loop', 'midpoint'])
def test_base_cage_recovery_triangles(scheme):
    from rqmesh import subdiv_tri
    V, T = icosahedron()
    Vs, Ts = subdiv_tri.subdivide_tri(V, T, scheme, 2)
    assert len(Ts) == 20 * 16
    r = subdiv_tri.unsubdivide_tri(Vs, Ts)
    assert len(r['cage'][1]) == 20


def test_simplify_torus():
    from rqmesh import transitions
    V, F = torus(48, 24)
    V2, F2, FUV, UV, info = transitions.adaptive_simplify(V, F, len(F) // 4, feature_deg=35, local='adaptive',
                                                          cost_mode='qem')
    assert all(len(f) == 4 for f in F2)
    assert len(F2) <= 1.15 * (len(F) // 4)
    nb, e = _valence(F2)
    assert all(c == 2 for c in e.values())                   # still closed and manifold
    assert all(len(s) == 4 for s in nb.values())             # a torus needs no poles


def test_square_hole_in_plain_grid_has_no_rings():
    """Index theorem: rings around an opening need poles at the box corners; a plain grid has none."""
    from rqmesh import isolation
    V, F = plate_with_hole(16, 4)
    assert sorted(len(o) for o in isolation.openings(F)) == [16, 64]      # hole and outer border
    hole = [r for r in isolation.report(V, F) if r['size'] == 16][0]
    assert hole['rings'] == 0


def test_isolation_rings_around_round_hole():
    from rqmesh import isolation
    V, F = polar_annulus(24, 10)
    rep = isolation.report(V, F)
    # inner and outer border; layers whose small side holds < 45% of the 240 faces: 5 rings each, all pole-free
    assert len(rep) == 2 and all(r['rings'] == 5 and not r['poles_on_inner_rings'] for r in rep)
    edges, info = isolation.isolation_loops(V, F, k=2, target_faces=len(F) // 4)
    assert edges and max(s['loops_kept'] for s in info['seeds']) == 2


def test_fix_poles_keeps_valid_mesh():
    from rqmesh import subdiv, topofix
    V, F = cube()
    Vs, Fs = subdiv.resubdivide(V, F, 'cc', 2)
    pf = topofix.PoleFixer(Vs, Fs, 30.0)
    p0 = pf.poles()
    pf.improve()
    V2, F2 = pf.result()
    nb, e = _valence(F2)
    assert all(len(f) == 4 for f in F2) and all(c == 2 for c in e.values())
    assert sum(1 for s in nb.values() if len(s) != 4) <= 8   # a cube's eight 3-poles are required


def test_symmetry_planes():
    from rqmesh import symmetry
    V, F = torus(32, 16)
    planes = symmetry.detect_planes(V, F)
    assert len(planes) >= 1


def test_scale_matched_field_index_sum():
    pytest.importorskip('scipy')
    from rqmesh import scalefield, subdiv_tri
    from rqmesh.meshio_min import TriMesh
    V, T = icosahedron()
    V, T = subdiv_tri.subdivide_tri(V, T, 'midpoint', 3)
    V = V / np.linalg.norm(V, axis=1)[:, None]
    D, info = scalefield.scale_matched_field(V, np.asarray(T), h=0.2)
    assert info['index_sum'] == 8                             # 4 x Euler characteristic of a sphere


def test_fill_new_holes_closes_remesh_gaps():
    """QuadWild occasionally leaves one patch unfilled; holes the input did not have are closed with quads."""
    from rqmesh import repair
    V, F = torus(40, 20)
    m = 20
    holes = {5, 6, m + 5, m + 6} | {20 * m + 8, 20 * m + 9, 21 * m + 8, 21 * m + 9, 22 * m + 8, 22 * m + 9}
    F_cut = [f for i, f in enumerate(F) if i not in holes]
    F2, info = repair.fill_new_holes(V, F, V, F_cut)
    assert info['holes_new'] == 2 and info['filled_triangles'] == 0
    nb, e = _valence(F2)
    assert all(c == 2 for c in e.values())                   # closed again
    dirs = collections.Counter((f[k], f[(k + 1) % len(f)]) for f in F2 for k in range(len(f)))
    assert max(dirs.values()) == 1                           # consistent orientation


def test_edited_ngon_grid_recovers_its_ngons():
    """Hard-surface round trip: an n-gon cylinder gridded by one simple subdivision, bent, comes back with its
    32-sided caps.  The shape prior prefers the all-quad reading; it must not win when that reading cannot pass as
    a subdivision (fit > 2 LEVEL_TOL) and the true one fits within LEVEL_TOL."""
    from rqmesh import subdiv
    n = 32
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    V = np.array([[np.cos(t), np.sin(t), z] for z in (-1.0, 1.0) for t in a])
    F = [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
    F += [list(range(n - 1, -1, -1)), list(range(n, 2 * n))]
    Vd, Fd = subdiv.resubdivide(V, F, 'simple', 1)
    Vd = np.asarray(Vd, float)
    a45 = np.radians(45.0); R = 2.0 / a45                     # bend 45 degrees across the cylinder
    th = a45 * (Vd[:, 0] + 1) / 2
    Vb = np.c_[(R + Vd[:, 2]) * np.sin(th), Vd[:, 1], (R + Vd[:, 2]) * np.cos(th) - R]
    r = subdiv.unsubdivide(Vb, Fd, tol=None, levels=1)
    cF = r['cage'][1]
    assert sorted(len(f) for f in cF) == [4] * n + [n, n]
    assert {frozenset(r['orig'][v] for v in f) for f in cF} == {frozenset(f) for f in F}


def _inset_quad(V, F, fi, t=0.25):
    """Inset one quad (a topology edit): four inner vertices, the face replaced by five."""
    V = np.asarray(V, float); f = F[fi]; c = V[f].mean(0)
    inner = [len(V) + k for k in range(4)]
    V = np.vstack([V, [V[v] + t * (c - V[v]) for v in f]])
    G = [g for i, g in enumerate(F) if i != fi] + [list(inner)]
    G += [[f[k], f[(k + 1) % 4], inner[(k + 1) % 4], inner[k]] for k in range(4)]
    return V, G


def _valid_mesh(F):
    und, dirs = collections.Counter(), collections.Counter()
    for f in F:
        for k in range(len(f)):
            a, b = f[k], f[(k + 1) % len(f)]; und[(min(a, b), max(a, b))] += 1; dirs[(a, b)] += 1
    return max(und.values()) <= 2 and max(dirs.values()) == 1, sum(c == 1 for c in und.values())


def test_partial_recover_keeps_edited_parts():
    """After a topology edit the global Recover finds nothing; partial Recover peels the rest of the grid and keeps
    the edited part, joined without holes, flips or T-junctions."""
    from rqmesh import subdiv, partial
    n = 32
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    V = np.array([[np.cos(t), np.sin(t), z] for z in (-1.0, 1.0) for t in a])
    F = [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)] + [list(range(n - 1, -1, -1)), list(range(n, 2 * n))]
    Vd, Fd = subdiv.resubdivide(V, F, 'simple', 1)
    r = partial.unsubdivide_partial(Vd, Fd)                     # unedited: the same as the global Recover
    assert sorted(len(f) for f in r['F']) == [4] * n + [n, n]
    side = next(i for i, f in enumerate(Fd) if all(abs(Vd[v][2]) < 1.0 + 1e-9 for v in f) and
                len({round(Vd[v][2], 6) for v in f}) == 2)      # a quad on the side wall
    Ve, Fe = _inset_quad(Vd, [list(f) for f in Fd], side)
    assert not subdiv.unsubdivide(Ve, Fe, tol=None, levels=1)['levels']
    r = partial.unsubdivide_partial(Ve, Fe)
    ok, nb = _valid_mesh(r['F'])
    assert ok and nb == 0
    got = {frozenset(v for v in (r['orig'][u] for u in f) if v < len(V)) for f in r['F']}
    assert sum(frozenset(f) in got for f in F) >= 33               # all cage faces but the edited one
    for f, cs in zip(r['F'], r['corner_src']):                    # every corner points at its own input corner
        assert [r['orig'][u] for u in f] == [Fe[g][k] for g, k in cs]


def test_partial_recover_two_levels_catmull_clark():
    from rqmesh import subdiv, partial
    V, F = torus(12, 8)
    Vd, Fd = subdiv.resubdivide(V, F, 'cc', 2)
    Ve, Fe = _inset_quad(Vd, [list(f) for f in Fd], 0)
    r = partial.unsubdivide_partial(Ve, Fe, levels=2)
    ok, nb = _valid_mesh(r['F'])
    assert ok and nb == 0 and len(r['levels']) == 2
    got = {frozenset(v for v in (r['orig'][u] for u in f) if v < len(V)) for f in r['F']}
    assert sum(frozenset(f) in got for f in F) >= 0.9 * len(F)


def _skinned_tube(n=64, m=16, band=0.1):
    """Open tube along z, two bones blending over |z| < band (an elbow)."""
    z = np.linspace(-1, 1, n); t = np.linspace(0, 2 * np.pi, m, endpoint=False)
    V = np.array([[0.25 * np.cos(a), 0.25 * np.sin(a), zz] for zz in z for a in t])
    F = [[i * m + j, i * m + (j + 1) % m, (i + 1) * m + (j + 1) % m, (i + 1) * m + j]
         for i in range(n - 1) for j in range(m)]
    wb = np.clip((V[:, 2] + band) / (2 * band), 0, 1)
    return V, F, np.stack([1 - wb, wb], 1)


def test_skin_field_interpolates_weights():
    """Weights linear along the tube are carried exactly: barycentric on the closest input triangle."""
    from rqmesh.skin import SkinField
    V, F, _ = _skinned_tube()
    W = np.stack([(V[:, 2] + 1) / 2, (1 - V[:, 2]) / 2], 1)
    sk = SkinField(V, F, W)
    assert np.allclose(sk.at(V), W, atol=1e-12)                       # exact at the vertices
    rng = np.random.default_rng(1); f = rng.integers(len(F), size=200); r = rng.random((200, 1))
    P = V[[F[k][0] for k in f]] * (1 - r) + V[[F[k][2] for k in f]] * r   # points on face diagonals
    assert np.allclose(sk.at(P), np.stack([(P[:, 2] + 1) / 2, (1 - P[:, 2]) / 2], 1), atol=1e-9)
    assert np.allclose(sk.dw(V[:5], V[:5]), 0) and np.isclose(sk.dw(V[:1], V[-1:])[0], 1.0)


def test_simplify_keeps_joint_loops():
    """With skin weights the ring removal keeps the loops where the bones blend (round 17)."""
    from rqmesh import transitions
    from rqmesh.skin import SkinField
    V, F, W = _skinned_tube()
    counts = []
    for sk in (None, SkinField(V, F, W)):
        V2, F2, *_ = transitions.adaptive_simplify(V, F, len(F) // 4, feature_deg=35, skin=sk)
        zs = np.unique(np.round(V2[:, 2], 4))
        counts.append((len(F2), int((np.abs(zs) <= 0.15).sum())))
    (f0, in0), (f1, in1) = counts
    assert f0 == f1                                                   # same budget
    assert in1 > in0          # loops at the elbow: 4 vs 2 with SciPy, 3 vs 2 NumPy-only (ties on the symmetric
                              # tube break differently)
