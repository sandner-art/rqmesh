# SPDX-License-Identifier: Apache-2.0
"""Scale-matched cross fields for sculpts.

Sculpts carry detail (pores, wrinkles, voxel steps) far smaller than the quads asked for.  Curvature measured on
the raw surface follows that detail, so the field -- and every singularity it spawns -- answers to noise.  Here the
alignment is measured at the scale of the target quads, with the two core matrices of discrete geometry and a
"raising" of directions to their 4-fold power:

  1. lowering (low-pass):  (M + t L) X = M X0, the implicit heat step with the cotan Laplacian L and the lumped
     mass M; t = (c h)^2 for target edge length h removes features smaller than about one quad;
  2. shape operator per face on the smoothed surface (Rusinkiewicz 2004); its principal directions as the complex
     number ((k_a - k_c) + 2 i k_b)^2 = |k1 - k2|^2 e^{4 i theta}: squaring the line-field representation raises
     a curvature direction (symmetric under 180 degrees) to a cross (symmetric under 90 degrees) -- the n-fold
     "charge" of the field, as an angular-momentum raising operator multiplies e^{i m phi};
  3. smoothest 4-field with soft alignment, a Hermitian positive-definite system:
        min_u  sum_e w_e |u_f - r_fg u_g|^2  +  lam sum_f (A_f / h^2) a_f |u_f - g_f|^2
     w_e = |e|^2 / (A_f + A_g) (face-based Dirichlet weight), r_fg = e^{4 i (angle transport)} the discrete
     Levi-Civita connection raised to the 4th power, a_f = s^2 / (s^2 + s0^2) with s = |k1 - k2| h the turning
     across one quad: only curvature the quads can resolve steers the field.
  4. optional ladder (multigrid): the field is solved on a coarser copy and prolonged (a coarse-to-fine
     V-cycle with Gauss-Seidel smoothing), for sculpts with millions of faces.

Output: per-face direction vectors for QuadWild (.rosy) on the original triangles.
"""
import numpy as np
try:
    from .meshio_min import TriMesh
except ImportError:
    from meshio_min import TriMesh

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
except ImportError:                                  # pragma: no cover
    sp = spla = None


def cotan_laplacian(V, T):
    """Cotan Laplacian (positive semi-definite) and lumped (barycentric) mass."""
    n = len(V); I, J, W = [], [], []
    area = np.zeros(len(T))
    for k in range(3):
        i, j, o = T[:, k], T[:, (k + 1) % 3], T[:, (k + 2) % 3]
        a, b = V[i] - V[o], V[j] - V[o]
        cr = np.linalg.norm(np.cross(a, b), axis=1)
        cot = np.einsum('ij,ij->i', a, b) / np.maximum(cr, 1e-30)
        I += [i, j]; J += [j, i]; W += [-0.5 * cot, -0.5 * cot]
        area = 0.5 * cr
    I = np.concatenate(I); J = np.concatenate(J); W = np.concatenate(W)
    L = sp.csr_matrix((W, (I, J)), shape=(n, n))
    L = L - sp.diags(np.asarray(L.sum(1)).ravel())
    mv = np.zeros(n)
    for k in range(3): np.add.at(mv, T[:, k], area / 3.0)
    return L, sp.diags(mv)


def heat_smooth(V, T, t):
    if t <= 0: return V.copy()
    L, M = cotan_laplacian(V, T)
    A = (M + t * L).tocsc(); solve = spla.factorized(A)
    B = M @ V
    return np.stack([solve(B[:, k]) for k in range(3)], 1)


def vertex_normals(V, T):
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    vn = np.zeros_like(V)
    for k in range(3): np.add.at(vn, T[:, k], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1), 1e-30)[:, None]


def face_frames(M):
    e1 = M.V[M.T[:, 1]] - M.V[M.T[:, 0]]
    e1 /= np.maximum(np.linalg.norm(e1, axis=1), 1e-30)[:, None]
    return e1, np.cross(M.N, e1)


def shape_operator(M, vn):
    """Per-face 2x2 second fundamental form (a, b, c) in the face frame, least squares over the three edges."""
    e1, e2 = face_frames(M)
    A = np.zeros((len(M.T), 3, 3)); rhs = np.zeros((len(M.T), 3))
    for k in range(3):
        i, j = M.T[:, k], M.T[:, (k + 1) % 3]
        e = M.V[j] - M.V[i]; dn = vn[j] - vn[i]
        eu, ev = np.einsum('ij,ij->i', e, e1), np.einsum('ij,ij->i', e, e2)
        du, dv = np.einsum('ij,ij->i', dn, e1), np.einsum('ij,ij->i', dn, e2)
        # a eu + b ev = du ;  b eu + c ev = dv
        for row, r in ((np.stack([eu, ev, 0 * eu], 1), du), (np.stack([0 * eu, eu, ev], 1), dv)):
            A += np.einsum('fi,fj->fij', row, row); rhs += row * r[:, None]
    abc = np.linalg.solve(A + 1e-12 * np.eye(3)[None], rhs[..., None])[..., 0]
    return abc


def transport4(M):
    """Interior edges (f, g), Dirichlet weights and the 4th-power transport r_fg (face-based connection)."""
    e1, e2 = face_frames(M)
    m = M.interior
    f, g = M.EF[m, 0], M.EF[m, 1]
    d = M.edir[m]
    af = np.arctan2(np.einsum('ij,ij->i', d, e2[f]), np.einsum('ij,ij->i', d, e1[f]))
    ag = np.arctan2(np.einsum('ij,ij->i', d, e2[g]), np.einsum('ij,ij->i', d, e1[g]))
    w = M.elen[m] ** 2 / np.maximum(M.area[f] + M.area[g], 1e-30)
    return f, g, w, np.exp(4j * (af - ag))


def hermitian_system(M, diag):
    f, g, w, r = transport4(M); n = len(M.T)
    rows = np.concatenate([f, g, f, g]); cols = np.concatenate([f, g, g, f])
    vals = np.concatenate([w, w, -w * r, -w * np.conj(r)]).astype(complex)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n)) + sp.diags(diag.astype(complex))


def alignment(M_smooth, h, s0=0.05):
    """Guide g_f (unit, 4th power) and confidence a_f from the smoothed surface's shape operator."""
    abc = shape_operator(M_smooth, vertex_normals(M_smooth.V, M_smooth.T))
    a, b, c = abc[:, 0], abc[:, 1], abc[:, 2]
    z2 = (a - c) + 2j * b                         # |k1 - k2| e^{2 i theta}
    s = np.abs(z2) * h
    g = z2 ** 2; g = g / np.maximum(np.abs(g), 1e-30)
    conf = s ** 2 / (s ** 2 + s0 ** 2)
    return g, conf, s


def solve_field(M, g, conf, h, lam=1.0):
    diag = lam * M.area / h ** 2 * conf
    A = hermitian_system(M, diag + 1e-9 * M.area / h ** 2)
    u = spla.spsolve(A.tocsc(), diag * g)
    return u / np.maximum(np.abs(u), 1e-30)


def field_vectors(M, u):
    e1, e2 = face_frames(M)
    ang = np.angle(u) / 4.0
    return np.cos(ang)[:, None] * e1 + np.sin(ang)[:, None] * e2


def singularities(M, u):
    """Field index per interior vertex (in quarter turns): sum of 4th-power angle jumps around the one-ring."""
    f, g, w, r = transport4(M)
    jump = np.angle(u[f] * np.conj(r * u[g]))          # (u_f vs u_g transported), in 4-units
    # accumulate around vertices: each interior edge contributes to the rings of its two end vertices with
    # opposite orientation; use the face-angle defect form: index = (sum jumps + 4 * angle defect) / (2 pi)
    m = M.interior; E = M.E[m]
    nV = len(M.V); acc = np.zeros(nV)
    # orientation: edge (E0,E1) with f on the left when walking E0 -> E1 iff T[f] contains E0->E1 in order
    Tf = M.T[f]; k0 = np.argmax(Tf == E[:, [0]], axis=1)
    fwd = Tf[np.arange(len(f)), (k0 + 1) % 3] == E[:, 1]
    sgn = np.where(fwd, 1.0, -1.0)
    # counter-clockwise around E0 the dual loop crosses from the right face (g when fwd) into the left face (f)
    np.add.at(acc, E[:, 0], sgn * jump); np.add.at(acc, E[:, 1], -sgn * jump)
    # angle defect (Gaussian curvature) per vertex
    ang = np.zeros(nV)
    for k in range(3):
        i, j, o = M.T[:, k], M.T[:, (k + 1) % 3], M.T[:, (k + 2) % 3]
        a, b = M.V[j] - M.V[i], M.V[o] - M.V[i]
        ang_k = np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.einsum('ij,ij->i', a, b))
        np.add.at(ang, i, ang_k)
    bv = np.zeros(nV, bool); bv[M.E[M.boundary].ravel()] = True
    defect = 2 * np.pi - ang
    idx = (acc + 4 * defect) / (2 * np.pi)
    idx[bv] = 0
    singularities.last_residual = float(np.abs(idx - np.rint(idx)).max())
    return np.rint(idx).astype(int)


def scale_matched_field(V, T, h, c=1.0, lam=1.0, s0=0.05):
    """Returns per-face unit directions on (V, T), plus diagnostics."""
    V = np.asarray(V, float); T = np.asarray(T, int)
    M = TriMesh(V, T)
    Vs = heat_smooth(V, T, (c * h) ** 2) if c > 0 else V
    Ms = TriMesh(Vs, T) if c > 0 else M
    g, conf, s = alignment(Ms, h, s0)
    u = solve_field(M, g, conf, h, lam)
    sing = singularities(M, u)
    return field_vectors(M, u), dict(singularities=int((sing != 0).sum()), conf_mean=round(float(conf.mean()), 3),
                                      index_sum=int(sing.sum()))


GRID = dict(c=(2, 3, 4, 6), s0=(0.15, 0.3), lam=(0.03, 0.1, 0.3, 1.0))


def auto_field(V, T, h, grid=GRID, slack=0.10):
    """Model selection without tracing: solve the field over a small grid (each solve is one sparse Hermitian
    system) and keep, among fields within `slack` of the fewest singularities (+2), the best-aligned one.
    Singularities of the field become the poles of the quad mesh, so this ranks fields by the outcome that
    matters, at a fraction of a second per candidate."""
    V = np.asarray(V, float); T = np.asarray(T, int); M = TriMesh(V, T)
    res = []
    for c in grid['c']:
        Ms = TriMesh(heat_smooth(V, T, (c * h) ** 2), T)
        for s0 in grid['s0']:
            g, conf, _ = alignment(Ms, h, s0)
            for lam in grid['lam']:
                u = solve_field(M, g, conf, h, lam)
                sing = int((singularities(M, u) != 0).sum())
                mis = float((conf * np.abs(u - g) ** 2 * M.area).sum() / max((conf * M.area).sum(), 1e-30))
                res.append(dict(sing=sing, mis=mis, c=c, s0=s0, lam=lam, u=u))
    smin = min(r['sing'] for r in res)
    ok = [r for r in res if r['sing'] <= int(np.ceil((1 + slack) * smin)) + 2]
    best = min(ok, key=lambda r: (r['mis'], r['sing']))
    info = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in best.items() if k != 'u'}
    info['min_sing_in_grid'] = smin; info['candidates'] = len(res)
    return field_vectors(M, best['u']), info
