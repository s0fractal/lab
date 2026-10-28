"""Rotation plane of the memory (post hoc play): stack the class signals c(sigma) (class means of z_T minus their
mean, flattened) over sigma, take the top two right singular vectors, and read each sigma as a point in that
plane. A precessing memory turns around the origin; a dephasing one shrinks toward it without turning.
Angle 0 is the sigma = 0 point; the angle is unwrapped along sigma."""
import json
import math
import sys

import numpy as np

d = json.load(open(sys.argv[1]))
rows = d["rows"]
C = np.array([np.asarray(r["zmeans"]).ravel() for r in rows])          # (n_sigma, 7 * 3Nd)
U, S, Vt = np.linalg.svd(C, full_matrices=False)
xy = C @ Vt[:2].T
ref = xy[0] / np.linalg.norm(xy[0])
perp = np.array([-ref[1], ref[0]])
pts = np.stack([xy @ ref, xy @ perp], 1)
r0 = np.linalg.norm(C[0])
ang = np.unwrap(np.arctan2(pts[:, 1], pts[:, 0]))
if ang[np.argmax(np.abs(ang))] < 0:      # orient so the precession reads clockwise-positive
    ang, pts[:, 1] = -ang, -pts[:, 1]
in_plane = (S[:2] ** 2).sum() / (S ** 2).sum()
out = [{"sigma": r["sigma"], "angle": round(float(a), 4), "radius": round(float(np.linalg.norm(p) / r0), 4),
        "full": round(float(np.linalg.norm(c) / r0), 4), "acc_fixed": r["acc_fixed"], "acc_probe": r["acc_probe"]}
       for r, a, p, c in zip(rows, ang, pts, C)]
for o in out[::2]:
    print(f"sigma {o['sigma']:.2f}  angle {o['angle'] / math.pi:+.2f} pi  radius {o['radius']:.2f}  "
          f"|c| {o['full']:.2f}  acc {o['acc_fixed']:.3f}  probe {o['acc_probe']:.3f}")
print(f"{d['kind']}: top-2 plane holds {in_plane:.2f} of the signal energy over sigma")
json.dump({"kind": d["kind"], "step": d["step"], "in_plane": in_plane, "rows": out}, open(sys.argv[2], "w"))
