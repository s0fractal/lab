"""A spin echo for the clock memory (post hoc play; the prediction is in README before the runs).
The class signal of Soliton turns during the hold, by a phase that grows as sigma^2 (T - 3) (the clock law). NMR undoes
a dephasing by a pulse that reverses the sense in which phase accumulates, so that the second half of the wait unwinds
the first: an echo. Which reversal unwinds this clock tells what turns it. Three pulses, each applied between tokens:
  J   the gyroscopic field reversed, J -> -J, so G -> G^-1 (built as the Cayley rotation of -J): unwinds a drift that
      the field makes (a magnetron-like turn)
  P   the momenta flipped, P -> -P, the field kept: unwinds a spring oscillation that does not need the field
  PJ  both: the time reversal of the conservative dynamics (a Loschmidt echo); damping, noise and distractors are not
      reversed by any of them
"off" uses G = I. Nothing else changes: the model's own token step (phs/core.py) runs with the chosen G, the weights
are the frozen ones.

Protocols (the write is tokens 0, 1; tokens 2 .. T - 1 follow it):
  none24, none32, none48   the plain model at T = 24, 32, 48
  off24, off48             the field switched off for tokens 2 .. T - 1
  echoX48                  23 tokens as trained, then pulse X and 23 tokens reversed: zero net turn (X = J, P, PJ)
  rewindX48                8 tokens, pulse, 8 reversed, pulse back, 30 as trained: the net turn of T = 32, at T = 48
For each protocol and sigma 0, 0.05, ..., 1.2 (1024 sequences per cell, namespace "echo", the same corpus and noise for
every protocol of one T): the model's own readout; the class signal c (class means of z_T minus their mean), its
overlap r with the sigma = 0 signal of the same protocol, its norm relative to that, and the first zero of r
(sigma0, linear between grid points); a pooled LDA fit at sigma = 0 only (2048 sequences of namespace "echo-cal", the
same protocol) and scored on every test cell: a decoder that does not know the temperature. Across protocols: the
overlaps of their sigma = 0 signals.
Usage: echo.py LAB CKPT OUT [threads] [--selftest]
--selftest: with no pulse the loop gives the model's own logits bit for bit (sigma 0 and 0.6); G^-1 G = I; and one
reversed token or one momentum flip changes the logits (the bite)."""
import dataclasses
import json
import sys

import numpy as np
import torch

LAB, ckpt, out = sys.argv[1], sys.argv[2], sys.argv[3]
torch.set_num_threads(int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4].isdigit() else 1)
sys.path.insert(0, f"{LAB}/phs-soliton")
from phs import tasks  # noqa: E402
from phs.core import Drive, internal_force, wrap  # noqa: E402
from phs.models import build  # noqa: E402
from phs.train import to_torch  # noqa: E402

c = torch.load(ckpt)
spec = tasks.TASKS["A"]
model = build(c["kind"], spec["vocab"], spec["classes"], c["cfg"], torch.Generator().manual_seed(0))
model.load_state_dict(c["state"])
model.eval()
K = spec["classes"]
GRID = [round(0.05 * i, 2) for i in range(25)]
ON = lambda t: "on"  # noqa: E731


def rev_after(t0, t1=None):
    return lambda t: "rev" if t >= t0 and (t1 is None or t < t1) else "on"


# name: (T, field(t) in {"on", "rev", "off"}, tokens before which P -> -P)
PROTOCOLS = {
    "none24": (24, ON, ()), "none32": (32, ON, ()), "none48": (48, ON, ()),
    "off24": (24, lambda t: "on" if t < 2 else "off", ()), "off48": (48, lambda t: "on" if t < 2 else "off", ()),
    "echoJ48": (48, rev_after(25), ()), "echoP48": (48, ON, (25,)), "echoPJ48": (48, rev_after(25), (25,)),
    "rewindJ48": (48, rev_after(10, 18), ()), "rewindP48": (48, ON, (10, 18)),
    "rewindPJ48": (48, rev_after(10, 18), (10, 18)),
}


@torch.no_grad()
def run(x, sigma, seed, field, flips=()):
    """The model's forward pass with field(t) in {"on", "rev", "off"} choosing G for token t and P -> -P before each
    token in flips; returns z_T and the logits."""
    core = model.core
    phys = core.physics()
    G = {"on": core.rotation(phys), "rev": core.rotation(dataclasses.replace(phys, J=-phys.J)),
         "off": torch.eye(model.N, dtype=model.E_u.dtype)}
    c1, c2 = core.ou_coefficients(sigma.to(model.E_u.dtype))
    gen = torch.Generator().manual_seed(seed)
    theta = torch.zeros(x.shape[0], model.N, model.d, dtype=model.E_u.dtype)
    P = torch.zeros_like(theta)
    f_int = internal_force(theta, phys)
    for t in range(x.shape[1]):
        if t in flips:
            P = -P
        tok = x[:, t]
        drive = Drive(phi=wrap(model.E_phi[tok]), u=model.u_max * torch.sigmoid(model.E_u[tok]))
        theta, P, f_int = core.token(theta, P, f_int, phys, G[field(t)], drive, c1, c2, gen)
    z = torch.cat([torch.cos(theta), torch.sin(theta), P], dim=-1).reshape(x.shape[0], -1)
    return z, model.readout(z)


def cell(name, sigma, namespace="echo", n=1024):
    T, field, flips = PROTOCOLS[name]
    x, tg, sg = to_torch(tasks.test_set("A", T, sigma, n=n, namespace=namespace))
    z, logits = run(x, sg, tasks.derive_seed(f"{namespace}-noise", T, f"{sigma:.2f}"), field, flips)
    return z.double().numpy(), logits.double().numpy().argmax(1), tg[:, -1].numpy()


def class_signal(z, y):
    m = np.stack([z[y == k].mean(0) for k in range(K)])
    return (m - m.mean(0, keepdims=True)).ravel()


def lda_fit(z, y, lam=1e-3):
    mu = np.stack([z[y == k].mean(0) for k in range(K)])
    r = z - mu[y]
    S = r.T @ r / len(z)
    S = S + lam * S.diagonal().mean() * np.eye(S.shape[0])
    W = np.linalg.solve(S, mu.T)
    return W, -0.5 * (mu * W.T).sum(1)


def lda_predict(m, z):
    s = z @ m[0] + m[1]                                                # a class an LDA never saw is never predicted
    return np.where(np.isnan(s), -np.inf, s).argmax(1)


def first_zero(sig, r):
    for i in range(1, len(r)):
        if r[i] < 0 <= r[i - 1]:
            return sig[i - 1] + (sig[i] - sig[i - 1]) * r[i - 1] / (r[i - 1] - r[i])
    return None


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


if "--selftest" in sys.argv:
    ok = True
    x, tg, sg = to_torch(tasks.test_set("A", 48, 0.6, n=64, namespace="echo-selftest"))
    for s in (0.0, 0.6):
        sig = torch.full_like(sg, s)
        with torch.no_grad():
            ref = model(x, sig, generator=torch.Generator().manual_seed(7))[:, -1]
        _, mine = run(x, sig, 7, ON)
        same = torch.equal(ref, mine)
        print(f"{'ok ' if same else 'BAD'} field on everywhere gives the model's logits bit for bit at sigma {s}")
        ok &= same
    with torch.no_grad():
        phys = model.core.physics()
        G, Gr = model.core.rotation(phys), model.core.rotation(dataclasses.replace(phys, J=-phys.J))
        err = float((Gr @ G - torch.eye(model.N)).abs().max())
    print(f"{'ok ' if err < 1e-5 else 'BAD'} the reversed field undoes the field: |G^-1 G - I| = {err:.1e}")
    ok &= err < 1e-5
    for what, field, flips in (("one reversed token", lambda t: "rev" if t == 30 else "on", ()),
                               ("one momentum flip", ON, (30,))):
        _, bitten = run(x, torch.full_like(sg, 0.6), 7, field, flips)
        moved = float((bitten - mine).abs().max())
        print(f"{'ok ' if moved > 1e-4 else 'BAD'} bite: {what} moves the logits")
        ok &= moved > 1e-4
    sys.exit(0 if ok else 1)

res = {"ckpt": ckpt, "kind": c["kind"], "seed": c["seed"], "n": 1024, "n_cal": 2048, "sigma": GRID}
c0 = {}
for name, (T, field, flips) in PROTOCOLS.items():
    zc, _, yc = cell(name, 0.0, namespace="echo-cal", n=2048)
    dec = lda_fit(zc, yc)                                             # fit at sigma = 0 only
    sigs, own, lda0 = [], [], []
    for s in GRID:
        z, pred, y = cell(name, s)
        sigs.append(class_signal(z, y))
        own.append(float((pred == y).mean()))
        lda0.append(float((lda_predict(dec, z) == y).mean()))
    c0[name] = sigs[0]
    r = [cos(v, sigs[0]) for v in sigs]
    norm = [float(np.linalg.norm(v) / np.linalg.norm(sigs[0])) for v in sigs]
    res[name] = {"T": T, "field": "".join({"on": "+", "rev": "-", "off": "0"}[field(t)] for t in range(T)),
                 "flips": list(flips),
                 "own": own, "lda0": lda0, "r": r, "norm": norm, "abs0": float(np.linalg.norm(sigs[0])),
                 "sigma0": first_zero(GRID, r)}
    print(name, "sigma0", res[name]["sigma0"], "own(0)", own[0],
          " ".join(f"{s:.2f}:{a:+.2f}/{b:.2f}" for s, a, b in zip(GRID[::4], r[::4], lda0[::4])), flush=True)
res["overlap0"] = {a: {b: cos(c0[a], c0[b]) for b in PROTOCOLS} for a in PROTOCOLS}
json.dump(res, open(out, "w"), indent=1)
