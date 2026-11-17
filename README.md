# rqmesh

Structure-aware quad remeshing, simplification and retopology, in pure Python + NumPy (SciPy is used when present).

rqmesh works *with* the edge flow a mesh already has rather than starting over:

- **Simplify by edge rings.** It removes whole rings of quads and places transitions (3-1, or gentle 4-2 / 5-3 / 6-4 strips), so the result stays all quads and keeps its loops. It can also keep isolation loops around openings such as eyes and mouths.
- **Fix poles.** It improves irregular vertices with rotations, collapses, loop moves and minimal pattern refills, without breaking guarded isolation rings.
- **Recover base cages.** Given a mesh subdivided with Catmull-Clark, simple, Loop or midpoint subdivision, it recovers the original cage exactly, including creases and UVs.
- **Analyse structure.** It reports what kind of mesh it is (subdivided, structured, sculpt-like) and which workflow fits.
- **Scale-matched cross fields for sculpts.** These are curvature fields measured at the size of the target quads. They can drive the optional QuadWild-BiMDF back-end.

The Blender extension built on it is [rqmesh-blender](https://github.com/sandner-art/rqmesh-blender). The optional
GPL back-end is [rqmesh-backend-quadwild](https://github.com/sandner-art/rqmesh-backend-quadwild).

**Status:** alpha (0.x). The API may change between minor versions.

## Install

```
pip install rqmesh            # once released on PyPI
pip install rqmesh[scipy]     # with SciPy: faster spatial queries and solvers
pip install git+https://github.com/sandner-art/rqmesh   # from source
```

Requires Python 3.11+ and NumPy.

## Quick start

```python
from rqmesh.meshio_min import read_obj_uv, write_obj_uv
from rqmesh import transitions, topofix

V, F, UV, FUV = read_obj_uv("model.obj")
V2, F2, FUV2, UV2, info = transitions.adaptive_simplify(
    V, F, target=len(F) // 3, UV=UV, FUV=FUV,
    feature_deg=35, local="adaptive", cost_mode="qem", transitions="auto")

pf = topofix.PoleFixer(V2, F2, 30.0)
pf.guard_isolation(2)        # keep the innermost 2 rings around every opening
pf.improve(); pf.refill_poles(); pf.improve(); pf.relax(10)
V3, F3 = pf.result()
write_obj_uv("model_simplified.obj", V3, F3, None, None)
```

More in [`examples/`](examples/). Most modules also work as command-line tools, for example
`python -m rqmesh.transitions --help` and `python -m rqmesh.isolation model.obj`.

## Modules

| Module | Purpose |
|---|---|
| `structsimp`, `transitions` | Ring collapse, strip transitions, the error ladder (`adaptive_simplify`) |
| `topofix` | Pole fixing, refills, isolation-ring guard |
| `isolation` | Openings, isolating rings, protected loops, reports |
| `subdiv`, `subdiv_tri` | Base-cage recovery for quad and triangle subdivision schemes |
| `features`, `crease_split`, `symmetry` | Sharp features, crease splitting, mirror planes |
| `analyze` | Structure report and recommendation |
| `scalefield`, `defofield` | Cross fields (scale-matched for sculpts; deformation) |
| `repair` | Welding, small holes, degenerate faces |
| `qw_driver`, `strip_bounds` | Driver for the optional QuadWild-BiMDF back-end package |
| `meshio_min`, `compat`, `rosy` | I/O and NumPy-only stand-ins |

## Development

```
pip install -e .[test,scipy]
pytest
```

The CI runs the tests on Linux, Windows and macOS, on Python 3.11 and 3.13, with and without SciPy.
Blender's bundled Python has NumPy but no SciPy, so the NumPy-only path has to keep working.

## Licence

Apache-2.0 (see [LICENSE](LICENSE)). Apache-2.0 is compatible with the GPL-3.0, so the Blender extension (GPL) can
bundle this library.

## Citing

If you use rqmesh in research, please cite it using [CITATION.cff](CITATION.cff); GitHub shows a "Cite this
repository" button.
