"""A pause button for the clock memory (post hoc play; the prediction is in README before the runs).
decouple.py showed that a dense train of PJ pulses (P -> -P and J -> -J every two tokens) holds the class signal still
at every temperature. Here the train is a pause: at T = 128 the 96 tokens after the write carry 48 PJ pulses (CPMG,
tau = 1: before tokens 3, 5, ..., 97; an even count, so the field and the momenta are back as trained), and the last
30 tokens run as trained, as many as follow the write at T = 32. If the pause stops the clock, the model's own readout,
trained at T = 32, reads a memory four times older.

Protocols: none32 (T = 32, as trained), none128 (T = 128, as trained), pause128 (the pause, then 30 tokens). For each
and sigma 0, 0.1, ..., 1.2 (1024 sequences per cell, namespace "pause"): the model's own readout; the class signal of
the final state, its overlap r with the sigma = 0 signal of the same protocol, its norm and the first zero of r; and
the overlap of the sigma = 0 signal of pause128 with that of none32 (does the pause hand the readout the pattern it
learned?).
Usage: pause.py LAB CKPT OUT [threads] [--selftest]
--selftest: with no pulse the loop gives the model's own logits bit for bit (sigma 0 and 0.6); the pause has 48
pulses, a zero net turn and ends as trained; one pulse changes the logits (the bite)."""
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
GRID = [round(0.1 * i, 1) for i in range(13)]
PAUSE = tuple(range(3, 98, 2))                                          # 48 PJ pulses over tokens 2 .. 97
PROTOCOLS = {"none32": (32, ()), "none128": (128, ()), "pause128": (128, PAUSE)}


@torch.no_grad()
def run(x, sigma, seed, pulses):
    """The model's forward pass with a PJ pulse before each token in pulses (the field toggles, P -> -P)."""
    core = model.core
    phys = core.physics()
    G = {False: core.rotation(phys), True: core.rotation(dataclasses.replace(phys, J=-phys.J))}
    c1, c2 = core.ou_coefficients(sigma.to(model.E_u.dtype))
    gen = torch.Generator().manual_seed(seed)
    theta = torch.zeros(x.shape[0], model.N, model.d, dtype=model.E_u.dtype)
    P = torch.zeros_like(theta)
    f_int = internal_force(theta, phys)
    rev = False
    for t in range(x.shape[1]):
        if t in pulses:
            P, rev = -P, not rev
        tok = x[:, t]
        drive = Drive(phi=wrap(model.E_phi[tok]), u=model.u_max * torch.sigmoid(model.E_u[tok]))
        theta, P, f_int = core.token(theta, P, f_int, phys, G[rev], drive, c1, c2, gen)
    z = torch.cat([torch.cos(theta), torch.sin(theta), P], dim=-1).reshape(x.shape[0], -1)
    return z, model.readout(z)


def class_signal(z, y):
    m = np.stack([z[y == k].mean(0) for k in range(K)])
    return (m - m.mean(0, keepdims=True)).ravel()


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def first_zero(sig, r):
    for i in range(1, len(r)):
        if r[i] < 0 <= r[i - 1]:
            return sig[i - 1] + (sig[i] - sig[i - 1]) * r[i - 1] / (r[i - 1] - r[i])
    return None


def first_below(sig, acc, thr=0.8):
    """sigma_crit: where the accuracy first falls below thr (linear between grid points); 0 if it starts below."""
    if acc[0] < thr:
        return 0.0
    for i in range(1, len(acc)):
        if acc[i] < thr:
            return sig[i - 1] + (sig[i] - sig[i - 1]) * (acc[i - 1] - thr) / (acc[i - 1] - acc[i])
    return None


if "--selftest" in sys.argv:
    ok = True
    x, tg, sg = to_torch(tasks.test_set("A", 128, 0.6, n=48, namespace="pause-selftest"))
    for s in (0.0, 0.6):
        sig = torch.full_like(sg, s)
        with torch.no_grad():
            ref = model(x, sig, generator=torch.Generator().manual_seed(7))[:, -1]
        _, mine = run(x, sig, 7, ())
        same = torch.equal(ref, mine)
        print(f"{'ok ' if same else 'BAD'} no pulse gives the model's logits bit for bit at sigma {s}")
        ok &= same
    turn, rev = 0, False
    for t in range(2, 98):
        rev = rev ^ (t in PAUSE)
        turn += -1 if rev else 1
    good = len(PAUSE) == 48 and turn == 0 and not rev and all(t in range(3, 98) for t in PAUSE)
    print(f"{'ok ' if good else 'BAD'} the pause: {len(PAUSE)} pulses, net turn {turn}, ends as trained: {not rev}")
    ok &= good
    _, bitten = run(x, torch.full_like(sg, 0.6), 7, (40,))
    moved = float((bitten - mine).abs().max())
    print(f"{'ok ' if moved > 1e-4 else 'BAD'} bite: one pulse moves the logits")
    ok &= moved > 1e-4
    sys.exit(0 if ok else 1)

res = {"ckpt": ckpt, "kind": c["kind"], "seed": c["seed"], "n": 1024, "sigma": GRID, "pause": list(PAUSE)}
c0 = {}
for name, (T, pulses) in PROTOCOLS.items():
    own, sigs = [], []
    for s in GRID:
        x, tg, sg = to_torch(tasks.test_set("A", T, s, n=1024, namespace="pause"))
        z, logits = run(x, sg, tasks.derive_seed("pause-noise", T, f"{s:.2f}"), pulses)
        y = tg[:, -1].numpy()
        own.append(float((logits.double().numpy().argmax(1) == y).mean()))
        sigs.append(class_signal(z.double().numpy(), y))
    c0[name] = sigs[0]
    res[name] = {"T": T, "own": own, "r": [cos(v, sigs[0]) for v in sigs],
                 "norm": [float(np.linalg.norm(v) / np.linalg.norm(sigs[0])) for v in sigs],
                 "abs0": float(np.linalg.norm(sigs[0])), "sigma_crit": first_below(GRID, own)}
    res[name]["sigma0"] = first_zero(GRID, res[name]["r"])
    print(name, "own", " ".join(f"{s:.1f}:{a:.2f}" for s, a in zip(GRID, own)), "sigma_crit",
          res[name]["sigma_crit"], flush=True)
res["overlap0"] = {"pause128_none32": cos(c0["pause128"], c0["none32"]),
                   "none128_none32": cos(c0["none128"], c0["none32"])}
print("overlap with none32 at sigma 0:", res["overlap0"], flush=True)
json.dump(res, open(out, "w"), indent=1)
