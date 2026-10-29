"""Dynamical decoupling of the clock memory (post hoc play; the prediction is in README before the runs).
echo.py showed that one Loschmidt pulse (P -> -P and J -> -J) midway through the hold refocuses most of the thermal
phase of the class signal. Qubits are kept coherent by trains of such pulses (Carr-Purcell-Meiboom-Gill). Here the hold
is T = 128 (126 tokens after the write) and a train of n PJ pulses is laid out as CPMG: the first after tau = 63 / n
tokens, the rest 2 tau apart, so the net turn is zero at the end for every n; each pulse toggles the field and flips the
momenta. n = 0, 1, 3, 7, 21, 63 (tau = -, 63, 21, 9, 3, 1).

For each n and sigma 0, 0.1, ..., 1.2 (1024 test sequences per cell, namespace "decouple", the same corpus and noise for
every n): the class signal c(t) of z_t after every token (class means minus their mean); at the end its overlap r with
the sigma = 0 signal of the same n, its norm relative to that and its absolute norm, and the first zero of r (sigma0);
r(t) for sigma 0.3, 0.6, 0.9; at sigma = 0 the overlap of c(t) with c(3), the signal one token into the hold (does the
clock still turn?) and |c(t)|. Decoders on the final state, fit on a calibration set of the same n (namespace
"decouple-cal", 1024 per sigma): the LDA of the same sigma (knows sigma: what a linear reader can get) and one pooled
LDA over all sigmas (blind to sigma); both scored on the test cells; plus the model's own readout.
Usage: decouple.py LAB CKPT OUT [threads] [--selftest | --only N,... --sigmas S,...]
--only and --sigmas run a subset (sigma 0 first), which is what artifacts/verify.py replays; the pooled LDA, being fit
over the whole sigma grid, is left out there (null), every other number is the same cell as in the full run.
--selftest: with no pulse the loop gives the model's own logits bit for bit (sigma 0 and 0.6); every train has n pulses
and a zero net turn; one pulse changes the logits (the bite)."""
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
K, T = spec["classes"], 128
GRID = [round(0.1 * i, 1) for i in range(13)]
TRACE = (0.3, 0.6, 0.9)
NS = (0, 1, 3, 7, 21, 63)


def train(n):
    """Tokens before which a PJ pulse comes: CPMG over the 126 tokens 2 .. 127 after the write."""
    if n == 0:
        return ()
    tau = (T - 2) // (2 * n)
    return tuple(2 + tau + 2 * tau * i for i in range(n))


@torch.no_grad()
def run(x, sigma, seed, pulses, every=False):
    """The model's forward pass with a PJ pulse before each token in pulses (the field toggles, P -> -P); returns the
    final z and logits, and z after every token if every."""
    core = model.core
    phys = core.physics()
    G = {False: core.rotation(phys), True: core.rotation(dataclasses.replace(phys, J=-phys.J))}
    c1, c2 = core.ou_coefficients(sigma.to(model.E_u.dtype))
    gen = torch.Generator().manual_seed(seed)
    theta = torch.zeros(x.shape[0], model.N, model.d, dtype=model.E_u.dtype)
    P = torch.zeros_like(theta)
    f_int = internal_force(theta, phys)
    rev, zs = False, []
    for t in range(x.shape[1]):
        if t in pulses:
            P, rev = -P, not rev
        tok = x[:, t]
        drive = Drive(phi=wrap(model.E_phi[tok]), u=model.u_max * torch.sigmoid(model.E_u[tok]))
        theta, P, f_int = core.token(theta, P, f_int, phys, G[rev], drive, c1, c2, gen)
        if every or t == x.shape[1] - 1:
            zs.append(torch.cat([torch.cos(theta), torch.sin(theta), P], dim=-1).reshape(x.shape[0], -1))
    return zs, model.readout(zs[-1])


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


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def first_zero(sig, r):
    for i in range(1, len(r)):
        if r[i] < 0 <= r[i - 1]:
            return sig[i - 1] + (sig[i] - sig[i - 1]) * r[i - 1] / (r[i - 1] - r[i])
    return None


def cell(n, sigma, namespace, every=False, size=1024):
    """(final z, class signal after every token if every else None, own prediction, labels)."""
    x, tg, sg = to_torch(tasks.test_set("A", T, sigma, n=size, namespace=namespace))
    zs, logits = run(x, sg, tasks.derive_seed(f"{namespace}-noise", T, f"{sigma:.2f}"), train(n), every)
    y = tg[:, -1].numpy()
    sigs = [class_signal(z.double().numpy(), y) for z in zs] if every else None
    return zs[-1].double().numpy(), sigs, logits.double().numpy().argmax(1), y


if "--selftest" in sys.argv:
    ok = True
    x, tg, sg = to_torch(tasks.test_set("A", T, 0.6, n=48, namespace="decouple-selftest"))
    for s in (0.0, 0.6):
        sig = torch.full_like(sg, s)
        with torch.no_grad():
            ref = model(x, sig, generator=torch.Generator().manual_seed(7))[:, -1]
        _, mine = run(x, sig, 7, ())
        same = torch.equal(ref, mine)
        print(f"{'ok ' if same else 'BAD'} no pulse gives the model's logits bit for bit at sigma {s}")
        ok &= same
    for n in NS[1:]:
        p = train(n)
        turn, rev = 0, False
        for t in range(2, T):
            rev = rev ^ (t in p)
            turn += -1 if rev else 1
        good = len(p) == n and turn == 0 and all(2 < t < T for t in p)
        print(f"{'ok ' if good else 'BAD'} train n = {n}: {len(p)} pulses, net turn {turn}")
        ok &= good
    _, bitten = run(x, torch.full_like(sg, 0.6), 7, (40,))
    moved = float((bitten - mine).abs().max())
    print(f"{'ok ' if moved > 1e-4 else 'BAD'} bite: one pulse moves the logits")
    ok &= moved > 1e-4
    sys.exit(0 if ok else 1)

SUBSET = "--only" in sys.argv
if SUBSET:
    NS = tuple(int(v) for v in sys.argv[sys.argv.index("--only") + 1].split(","))
    GRID = [float(v) for v in sys.argv[sys.argv.index("--sigmas") + 1].split(",")]
    assert GRID[0] == 0.0, "r is measured against sigma = 0, so the sigma list starts there"
res = {"ckpt": ckpt, "kind": c["kind"], "seed": c["seed"], "T": T, "n_test": 1024, "n_cal": 1024, "sigma": GRID,
       "trace_sigma": list(TRACE)}
for n in NS:
    cal = {s: cell(n, s, "decouple-cal") for s in GRID}
    aware = {s: lda_fit(cal[s][0], cal[s][3]) for s in GRID}
    pooled = lda_fit(np.concatenate([cal[s][0] for s in GRID]), np.concatenate([cal[s][3] for s in GRID]))
    del cal
    first = cell(n, 0.0, "decouple", every=True)
    ref = first[1]                                                    # the sigma = 0 signal after every token
    row = {"pulses": list(train(n)), "own": [], "aware": [], "pooled": [], "r": [], "norm": [], "abs": [], "trace": {}}
    for s in GRID:
        z, sigs, pred, y = first if s == 0.0 else cell(n, s, "decouple", every=s in TRACE)
        sig = class_signal(z, y)
        row["own"].append(float((pred == y).mean()))
        row["aware"].append(float((lda_predict(aware[s], z) == y).mean()))
        row["pooled"].append(None if SUBSET else float((lda_predict(pooled, z) == y).mean()))
        row["r"].append(cos(sig, ref[-1]))
        row["norm"].append(float(np.linalg.norm(sig) / np.linalg.norm(ref[-1])))
        row["abs"].append(float(np.linalg.norm(sig)))
        if s in TRACE:
            row["trace"][f"{s:.1f}"] = [round(cos(a, b), 4) for a, b in zip(sigs, ref)]
    row["sigma0"] = first_zero(GRID, row["r"])
    row["anchor0"] = [round(cos(v, ref[3]), 4) for v in ref]           # sigma = 0: c(t) against c(3)
    row["abs0_t"] = [round(float(np.linalg.norm(v)), 4) for v in ref]
    res[f"dd{n}"] = row
    print(f"dd{n} sigma0 {row['sigma0']} anchor(127) {row['anchor0'][-1]:+.2f} " + " ".join(
        f"{s:.1f}:{a:+.2f}/{b:.2f}" for s, a, b in zip(GRID[::3], row["r"][::3], row["aware"][::3])), flush=True)
json.dump(res, open(out, "w"), indent=1)
