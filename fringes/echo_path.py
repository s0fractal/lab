"""The clock hand, token by token (post hoc; a picture of echo.py, not a test). For one frozen model, protocols none48,
echoJ48, echoP48, echoPJ48 of echo.py and sigma 0, 0.3, 0.6, 0.9 (1024 sequences, namespace "echo", the same corpus
and noise as there), the class signal c(t) of z_t (class means minus their mean) after every token. All of them are
projected on the two leading singular axes of the whole set (uncentred: the signal turns about zero), so every
protocol turns in one plane; also |c(t)| and the overlap of c(t) with the sigma = 0 signal of the same protocol at the
same token.
Usage: echo_path.py LAB CKPT OUT [threads]"""
import json
import sys

import numpy as np
import torch

LAB, ckpt, out = sys.argv[1], sys.argv[2], sys.argv[3]
sys.argv = ["echo.py", LAB, ckpt, "/dev/null", sys.argv[4] if len(sys.argv) > 4 else "1"]
ns = {}                                     # echo.py's definitions, up to where its command-line modes begin
src = open(f"{LAB}/fringes/echo.py").read().split('if "--selftest" in sys.argv:')[0]
exec(compile(src, "echo.py", "exec"), ns)
tasks, to_torch, model, K = ns["tasks"], ns["to_torch"], ns["model"], ns["K"]
NAMES, SIGMAS = ("none48", "echoJ48", "echoP48", "echoPJ48"), (0.0, 0.3, 0.6, 0.9)


@torch.no_grad()
def path(name, sigma, n=1024):
    """Class signal after every token: (T, K * 3Nd)."""
    import dataclasses
    from phs.core import Drive, internal_force, wrap
    T, field, flips = ns["PROTOCOLS"][name]
    x, tg, sg = to_torch(tasks.test_set("A", T, sigma, n=n, namespace="echo"))
    y = tg[:, -1].numpy()
    core = model.core
    phys = core.physics()
    G = {"on": core.rotation(phys), "rev": core.rotation(dataclasses.replace(phys, J=-phys.J)),
         "off": torch.eye(model.N, dtype=model.E_u.dtype)}
    c1, c2 = core.ou_coefficients(sg.to(model.E_u.dtype))
    gen = torch.Generator().manual_seed(tasks.derive_seed("echo-noise", T, f"{sigma:.2f}"))
    theta = torch.zeros(n, model.N, model.d, dtype=model.E_u.dtype)
    P = torch.zeros_like(theta)
    f_int = internal_force(theta, phys)
    cs = []
    for t in range(T):
        if t in flips:
            P = -P
        tok = x[:, t]
        drive = Drive(phi=wrap(model.E_phi[tok]), u=model.u_max * torch.sigmoid(model.E_u[tok]))
        theta, P, f_int = core.token(theta, P, f_int, phys, G[field(t)], drive, c1, c2, gen)
        z = torch.cat([torch.cos(theta), torch.sin(theta), P], dim=-1).reshape(n, -1).double().numpy()
        m = np.stack([z[y == k].mean(0) for k in range(K)])
        cs.append((m - m.mean(0, keepdims=True)).ravel())
    return np.stack(cs), "".join({"on": "+", "rev": "-", "off": "0"}[field(t)] for t in range(T)), list(flips)


paths = {(nm, s): path(nm, s) for nm in NAMES for s in SIGMAS}
X = np.concatenate([p[0] for p in paths.values()])
_, sv, Vt = np.linalg.svd(X, full_matrices=False)                  # uncentred: the signal turns about zero
axes = Vt[:2]
res = {"ckpt": ckpt, "kind": ns["c"]["kind"], "seed": ns["c"]["seed"], "n": 1024, "sigmas": list(SIGMAS),
       "explained": [float(v) for v in (sv[:2] ** 2 / (sv ** 2).sum())], "protocols": {}}
for nm in NAMES:
    ref = paths[(nm, 0.0)][0]
    res["protocols"][nm] = {"field": paths[(nm, 0.0)][1], "flips": paths[(nm, 0.0)][2], "by_sigma": {
        f"{s:.1f}": {"xy": (paths[(nm, s)][0] @ axes.T).round(5).tolist(),
                     "norm": np.linalg.norm(paths[(nm, s)][0], axis=1).round(5).tolist(),
                     "r": [round(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)), 4)
                           for a, b in zip(paths[(nm, s)][0], ref)]} for s in SIGMAS}}
    print(nm, "done", flush=True)
json.dump(res, open(out, "w"))
