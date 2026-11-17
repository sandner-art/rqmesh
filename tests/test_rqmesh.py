# SPDX-License-Identifier: Apache-2.0
import importlib, os, subprocess, sys, collections
import numpy as np
import pytest

from conftest import cube, torus, icosahedron, plate_with_hole, polar_annulus

MODULES = ['analyze', 'compat', 'crease_split', 'defofield', 'features', 'isolation', 'meshio_min', 'repair', 'rosy',
           'scalefield', 'strip_bounds', 'structsimp', 'subdiv', 'subdiv_tri', 'symmetry', 'topofix', 'transitions']


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
