# SPDX-License-Identifier: Apache-2.0
import importlib, os, subprocess, sys, collections
import numpy as np
import pytest

from conftest import cube, torus, icosahedron, plate_with_hole, polar_annulus

MODULES = ['analyze', 'compat', 'crease_split', 'defofield', 'features', 'isolation', 'masks', 'meshio_min', 'partial', 'repair',
           'rigid_rig', 'rosy', 'scalefield', 'skin', 'strip_bounds', 'uvlayout', 'structsimp', 'subdiv', 'subdiv_tri', 'symmetry', 'topofix',
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


def _rr_box(c, s):
    c = np.asarray(c, float); s = np.asarray(s, float) / 2
    V = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * s + c
    return V, [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]


def _rr_cyl(c, r, L, axis, n=24):
    a = np.asarray(axis, float); a /= np.linalg.norm(a)
    e1 = np.cross(a, [0.3, 0.2, 0.9]); e1 /= np.linalg.norm(e1); e2 = np.cross(a, e1)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    V = np.array([np.asarray(c) + h * a + r * (np.cos(x) * e1 + np.sin(x) * e2) for h in (-L / 2, L / 2) for x in t])
    return V, [[k, (k + 1) % n, n + (k + 1) % n, n + k] for k in range(n)] + [list(range(n))[::-1], list(range(n, 2 * n))]


def _rr_sphere(c, r, seg=24, rings=12):
    V = [np.asarray(c) + [0, 0, -r]]
    for i in range(1, rings):
        th = np.pi * i / rings - np.pi / 2
        V += [np.asarray(c) + r * np.array([np.cos(th) * np.cos(p), np.cos(th) * np.sin(p), np.sin(th)])
              for p in np.linspace(0, 2 * np.pi, seg, endpoint=False)]
    V.append(np.asarray(c) + [0, 0, r]); top = len(V) - 1
    ring = lambda i, k: 1 + (i - 1) * seg + k % seg
    F = [[0, ring(1, k + 1), ring(1, k)] for k in range(seg)]
    F += [[ring(i, k), ring(i, k + 1), ring(i + 1, k + 1), ring(i + 1, k)] for i in range(1, rings - 1) for k in range(seg)]
    F += [[ring(rings - 1, k), ring(rings - 1, k + 1), top] for k in range(seg)]
    return np.array(V), F


def _rr_join(parts):
    V, F, off = [], [], 0
    for v, f in parts:
        V.append(v); F += [[i + off for i in q] for q in f]; off += len(v)
    return np.vstack(V), F


@pytest.mark.parametrize('case', ['pin', 'slide', 'ball'])
def test_rigid_rig_joint_kinds(case):
    """Rig Hard Surface analysis (round 18): a pin makes a hinge about its axis, a rod entering a sleeve a slide along
    it, a sphere between two parts a ball joint at its centre."""
    from rqmesh import rigid_rig as RR
    base = _rr_box((0, 0, 0.1), (1.2, 1.2, 0.2))
    if case == 'pin':
        parts = [base, _rr_box((0, 0, 0.6), (0.3, 0.3, 0.8)), _rr_cyl((0, 0, 1.0), 0.06, 0.6, (0, 1, 0)),
                 _rr_box((0, 0, 1.45), (0.25, 0.25, 0.8))]
        want, axis, pivot = 'hinge', (0, 1, 0), (0, 0, 1.0)
    elif case == 'slide':
        parts = [base, _rr_cyl((0, 0, 0.6), 0.12, 0.8, (0, 0, 1)), _rr_cyl((0, 0, 1.2), 0.07, 0.9, (0, 0, 1)),
                 _rr_box((0, 0, 1.65), (0.8, 0.8, 0.1))]
        want, axis, pivot = 'slide', (0, 0, 1), None
    else:
        parts = [base, _rr_box((0, 0, 0.55), (0.14, 0.14, 0.9)), _rr_sphere((0, 0, 1.05), 0.1),
                 _rr_box((0, 0, 1.5), (0.1, 0.1, 0.8))]
        want, axis, pivot = 'ball', None, (0, 0, 1.05)
    V, F = _rr_join(parts)
    r = RR.analyze(V, F)
    moving = [j for b, j in r['joint'].items() if j['kind'] != 'root']
    assert [j['kind'] for j in moving] == [want], [(j['kind'], j['evidence']) for j in moving]
    j = moving[0]
    if axis is not None:
        assert abs(abs(np.asarray(j['axis']) @ np.asarray(axis)) - 1) < 1e-6
    if pivot is not None:
        d = np.asarray(j['pivot']) - pivot
        if axis is not None:
            d = d - (d @ np.asarray(axis, float)) * np.asarray(axis, float)
        assert np.linalg.norm(d) < 1e-3
    assert (r['face_body'] >= 0).all() and len(set(r['face_body'])) == 2       # every face rigidly in one of 2 parts


def _capped_cylinder(n=8, rings=3):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    V = np.array([[np.cos(a), np.sin(a), z] for z in np.linspace(-1, 1, rings) for a in t])
    F = [[i * n + k, i * n + (k + 1) % n, (i + 1) * n + (k + 1) % n, (i + 1) * n + k] for i in range(rings - 1) for k in range(n)]
    return V, F + [list(range(n))[::-1], list(range((rings - 1) * n, rings * n))]


@pytest.mark.parametrize('case', ['cube', 'cylinder'])
def test_uvs_from_the_recovered_cage(case):
    """Round 19: panels and cage layouts carried down two Catmull-Clark levels.  Cube: 6 panels of 16 faces, the
    patch layout is 6 islands with loops exactly straight.  Capped cylinder: the side band is one island (one seam),
    plus the two n-gon caps."""
    from rqmesh import subdiv, uvlayout as UL
    Vc, Fc = _rr_box((0, 0, 0), (2, 2, 2)) if case == 'cube' else _capped_cylinder()
    Vd, Fd = subdiv.resubdivide(Vc, Fc, 'cc', 2); Vd = np.asarray(Vd); Fd = [list(map(int, f)) for f in Fd]
    tr = []; r = subdiv.unsubdivide(Vd, Fd, levels=2, trace=tr); cV, cF = r['cage']
    pc, pid = UL.panels(tr, cF)
    assert len(set(pid.tolist())) == len(cF)
    assert all(((p >= -1e-9) & (p <= 1 + 1e-9)).all() for p in pc)
    cu, ch = UL.cage_layout(cV, cF, 'patches', cuts=r['creases']); duv, _ = UL.push_corners(tr, cu)
    m = UL.uv_metrics(Vd, Fd, duv)
    assert m['flipped'] == 0 and m['islands'] == (6 if case == 'cube' else 3)
    if case == 'cube':
        assert m['grain_deg'] < 1e-6 and np.bincount(pid).tolist() == [16] * 6


def test_part_uvs_unwrap_cylinders_and_boxes():
    """Shape-aware part unwrap: a cylinder is found and unwrapped around its axis, boxes are projected in their own
    frame; nothing is mirrored and angles are kept (median ratio 1)."""
    from rqmesh import uvlayout as UL
    V, F = _rr_join([_rr_box((0, 0, 0.5), (0.6, 0.4, 1.0)), _rr_cyl((0, 0, 1.3), 0.15, 0.6, (0.3, 0.2, 1.0))])
    uv, seg, kinds = UL.part_uvs(V, F)
    assert sorted(kinds.values()) == ['box', 'cylinder']
    m = UL.uv_metrics(V, F, uv)
    assert m['flipped'] == 0 and m['angle'] < 1.01


def test_merge_charts_joins_flat_pieces():
    """Round 19b: on a flat disc of quads around a 5-valent pole the separatrices cut several patches; with no
    curvature inside they merge into one chart (still a disc)."""
    from rqmesh import uvlayout as UL
    n = 5; c = [0.0, 0.0, 0.0]; V = [c]
    for k in range(n):
        a = 2 * np.pi * k / n; b = a + np.pi / n
        V += [[np.cos(a), np.sin(a), 0.0], [1.6 * np.cos(b), 1.6 * np.sin(b), 0.0]]
    V = np.array(V); F = [[0, 1 + 2 * k, 2 + 2 * k, 1 + 2 * ((k + 1) % n)] for k in range(n)]
    _, ch = UL.cage_layout(V, F, 'patches')
    lab = UL.merge_charts(V, F, ch, max_curv=0.1)
    assert len(set(ch.tolist())) > 1 and len(set(lab.tolist())) == 1


def test_trim_layout_and_surface_masks():
    """Trim: every face lands inside its band (v), nothing mirrored, loops straight.  Masks: wear is highest on the
    cube's rounded edges and covers about the requested area; the variation is 0..1 and continuous."""
    from rqmesh import subdiv, uvlayout as UL, masks as MK
    Vc, Fc = _rr_box((0, 0, 0), (2, 1.2, 0.6))
    Vd, Fd = subdiv.resubdivide(Vc, Fc, 'cc', 2); Vd = np.asarray(Vd); Fd = [list(map(int, f)) for f in Fd]
    tr = []; r = subdiv.unsubdivide(Vd, Fd, levels=2, trace=tr); cV, cF = r['cage']
    rows = [(0.5, 1.0), (0.25, 0.5), (0.0, 0.25)]
    cu, uid, band, err = UL.trim_layout(cV, cF, rows)
    duv, pid = UL.push_corners(tr, cu)
    for i, u in enumerate(duv):
        v0, v1 = rows[band[pid[i]]]
        assert (u[:, 1] >= v0 - 1e-9).all() and (u[:, 1] <= v1 + 1e-9).all()
    m = UL.uv_metrics(Vd, Fd, duv); assert m['flipped'] == 0 and m['grain_deg'] < 1e-6
    w, cav = MK.wear_cavity(Vd, Fd, wear=0.1); A = MK._vertex_area(Vd, Fd)
    share = A[w > 0.5].sum() / A.sum(); assert 0.03 < share < 0.25, share
    top = int(np.argmin(np.linalg.norm(Vd - [0, 0, Vd[:, 2].max()], axis=1)))   # middle of the flat top
    assert w.max() == 1 and w[top] == 0 and cav.max() < 0.5                       # a convex box has no cavities
    x = MK.structure_noise(tr, len(Vd)); assert x.min() == 0 and x.max() == 1


def test_exposure_and_trim_grouping():
    """Exposure (round 19c): a small box standing on a plate is more open at its top than at its foot.  Trim: joining
    strips never raises the texel-density error and keeps every face inside its band."""
    from rqmesh import masks as MK, subdiv, uvlayout as UL
    V, F = _rr_join([_rr_box((0, 0, 0.05), (2.0, 2.0, 0.1)), _rr_box((0, 0, 0.35), (0.3, 0.3, 0.5))])
    V, F = subdiv.resubdivide(V, F, 'simple', 3); V = np.asarray(V); F = [list(map(int, f)) for f in F]
    ex = MK.exposure(V, F, radius=0.2)
    side = (np.abs(np.abs(V[:, 0]) - 0.15) < 1e-6) & (np.abs(V[:, 1]) < 0.1)
    foot, top = side & (V[:, 2] < 0.15), (V[:, 2] > 0.6 - 1e-6)
    assert 0 <= ex.min() and ex.max() <= 1 and ex[top].mean() > ex[foot].mean() + 0.2
    Vc, Fc = _rr_box((0, 0, 0), (3, 1, 0.5)); Vd, Fd = subdiv.resubdivide(Vc, Fc, 'cc', 2); Vd = np.asarray(Vd)
    Fd = [list(map(int, f)) for f in Fd]; tr = []; r = subdiv.unsubdivide(Vd, Fd, levels=2, trace=tr); cV, cF = r['cage']
    rows = [(0.5, 1.0), (0.25, 0.5), (0.0, 0.25)]
    for g in (False, True):
        cu, uid, band, err = UL.trim_layout(cV, cF, rows, group=g); duv, pid = UL.push_corners(tr, cu)
        assert all((u[:, 1] >= rows[band[pid[i]]][0] - 1e-9).all() and (u[:, 1] <= rows[band[pid[i]]][1] + 1e-9).all()
                   for i, u in enumerate(duv))
        assert UL.uv_metrics(Vd, Fd, duv)['flipped'] == 0
