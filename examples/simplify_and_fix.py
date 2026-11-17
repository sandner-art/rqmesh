# SPDX-License-Identifier: Apache-2.0
"""Simplify a quad mesh by edge rings, keep isolation loops around openings, then fix poles.

usage: python simplify_and_fix.py IN.obj OUT.obj [RATIO]      (no input: a generated test torus)"""
import sys, time
import numpy as np
from rqmesh.meshio_min import read_obj_uv, write_obj_uv
from rqmesh import transitions, topofix, isolation


def torus(n=64, m=32, R=1.0, r=0.4):
    V = np.array([[(R + r * np.cos(2 * np.pi * j / m)) * np.cos(2 * np.pi * i / n),
                   (R + r * np.cos(2 * np.pi * j / m)) * np.sin(2 * np.pi * i / n), r * np.sin(2 * np.pi * j / m)]
                  for i in range(n) for j in range(m)])
    F = [[i * m + j, ((i + 1) % n) * m + j, ((i + 1) % n) * m + (j + 1) % m, i * m + (j + 1) % m]
         for i in range(n) for j in range(m)]
    return V, F, None, None


inp = sys.argv[1] if len(sys.argv) > 1 else None
out = sys.argv[2] if len(sys.argv) > 2 else 'simplified.obj'
ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.35
V, F, UV, FUV = read_obj_uv(inp) if inp else torus()
V = np.asarray(V, float); target = max(4, int(len(F) * ratio))

t = time.time()
loops, linfo = isolation.isolation_loops(V, F, k=2, target_faces=target)      # empty if there are no openings
V2, F2, FUV2, UV2, info = transitions.adaptive_simplify(V, F, target, UV=UV, FUV=FUV, feature_deg=35,
                                                        local='adaptive', cost_mode='qem', transitions='auto',
                                                        extra_feat=sorted(loops) or None)
pf = topofix.PoleFixer(np.asarray(V2), F2, 30.0)
pf.guard_isolation(3)
pf.improve(); pf.refill_poles(); pf.improve(); pf.relax(10)
V3, F3 = pf.result()
write_obj_uv(out, V3, F3, None, None)
print(f"{len(F)} -> {len(F3)} faces, {pf.poles()} poles, {time.time() - t:.1f} s -> {out}")
