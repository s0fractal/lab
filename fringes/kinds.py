"""Which pulse holds the clock: J, P or both? (post hoc play; the prediction is in README before the runs)
decouple.py held the memory with dense trains of PJ pulses (P -> -P and J -> -J). Codex (review of #9) asked for the
same trains with one half of the pulse only. Here the CPMG layout of decouple.py (the first pulse tau = 63 / n tokens
into the hold, the rest 2 tau apart, n = 7, 21, 63) is run with three kinds of pulse:
  PJ   the field toggles and the momenta flip (decouple.py's own train; its numbers must come out the same)
  J    the field toggles (G -> G^-1), the momenta are left alone
  P    the momenta flip, the field is left as trained
and two references without pulses: none (the model as it is) and off (G = I from the first hold token on).
For each protocol and sigma 0, 0.3, 0.6, 0.9 (1024 test sequences per cell; the corpora and noise are decouple.py's,
namespaces "decouple" and "decouple-cal"): the model's own readout; the LDA of the same sigma fit on the calibration
set of the same protocol (knows sigma); the class signal c of the final z, its overlap r with the sigma = 0 signal of the
same protocol and its norm; at sigma = 0, after every token, |c(t)| and the overlap of c(t) with c(3); at sigma = 0.6,
after every token, r(t).
Usage: kinds.py LAB CKPT OUT [threads] [--selftest | --scales | --momenta | --only NAME,... --sigmas S,...]
--selftest: with no pulse the loop gives the model's own logits bit for bit (sigma 0 and 0.6); a PJ train gives
decouple.py's run bit for bit; a J and a P pulse each change the logits; the off field is the identity. --scales prints
the per-token timescales of the checkpoint (no protocol is run). --momenta (post hoc, after the runs) prints the mean
squared momenta over the hold without pulses at sigma 0 and 0.6, against the thermal level 0.6^2 <m> (512 sequences).
--only and --sigmas run a subset (sigma 0 first)."""
import dataclasses
import json
import sys

import numpy as np
import torch

LAB = sys.argv[1]
ns = {}                                     # decouple.py's definitions, up to where its command-line modes begin
src = open(f"{LAB}/fringes/decouple.py").read().split('if "--selftest" in sys.argv:')[0]
exec(compile(src, "decouple.py", "exec"), ns)
model, tasks, to_torch, T, train = ns["model"], ns["tasks"], ns["to_torch"], ns["T"], ns["train"]
Drive, internal_force, wrap = ns["Drive"], ns["internal_force"], ns["wrap"]
class_signal, lda_fit, lda_predict, cos = ns["class_signal"], ns["lda_fit"], ns["lda_predict"], ns["cos"]
GRID = [0.0, 0.3, 0.6, 0.9]
NS = (7, 21, 63)
PROTOCOLS = {"none": ("PJ", 0), "off": ("off", 0), **{f"{k}{n}": (k, n) for k in ("PJ", "J", "P") for n in NS}}


@torch.no_grad()
def run(x, sigma, seed, kind, pulses, every=False):
    """The model's forward pass with a pulse of this kind before each token in pulses (PJ: the field toggles and
    P -> -P; J: the field toggles; P: P -> -P), or with no field from token 2 on (off); returns the final z and
    logits, and z after every token if every."""
    core = model.core
    phys = core.physics()
    G = {False: core.rotation(phys), True: core.rotation(dataclasses.replace(phys, J=-phys.J))}
    eye = torch.eye(model.N, dtype=G[False].dtype)
    c1, c2 = core.ou_coefficients(sigma.to(model.E_u.dtype))
    gen = torch.Generator().manual_seed(seed)
    theta = torch.zeros(x.shape[0], model.N, model.d, dtype=model.E_u.dtype)
    P = torch.zeros_like(theta)
    f_int = internal_force(theta, phys)
    rev, zs = False, []
    for t in range(x.shape[1]):
        if t in pulses:
            if kind in ("PJ", "P"):
                P = -P
            if kind in ("PJ", "J"):
                rev = not rev
        tok = x[:, t]
        drive = Drive(phi=wrap(model.E_phi[tok]), u=model.u_max * torch.sigmoid(model.E_u[tok]))
        g = eye if kind == "off" and t >= 2 else G[rev]
        theta, P, f_int = core.token(theta, P, f_int, phys, g, drive, c1, c2, gen)
        if every or t == x.shape[1] - 1:
            zs.append(torch.cat([torch.cos(theta), torch.sin(theta), P], dim=-1).reshape(x.shape[0], -1))
    return zs, model.readout(zs[-1])


def cell(name, sigma, namespace, every=False, size=1024):
    """(final z, class signal after every token if every else None, own prediction, labels)."""
    kind, n = PROTOCOLS[name]
    x, tg, sg = to_torch(tasks.test_set("A", T, sigma, n=size, namespace=namespace))
    zs, logits = run(x, sg, tasks.derive_seed(f"{namespace}-noise", T, f"{sigma:.2f}"), kind, train(n), every)
    y = tg[:, -1].numpy()
    sigs = [class_signal(z.double().numpy(), y) for z in zs] if every else None
    return zs[-1].double().numpy(), sigs, logits.double().numpy().argmax(1), y


if "--scales" in sys.argv:
    from phs.core import cayley_rotation
    core = model.core
    with torch.no_grad():
        ph = core.physics()
        sq = ph.mass.sqrt()
        w = torch.linalg.eigvals(ph.J / (sq[:, None] * sq[None, :])).imag.abs()
        turn = torch.linalg.eigvals(torch.linalg.matrix_power(cayley_rotation(ph, 0.5 * core.h), 2 * core.S)).angle()
        pin = float(ph.g_pin) * ph.m_pin ** 2 * core.dt ** 2 / float(ph.mass.mean()) if ph.lattice else 0.0
    print(f"{ns['c']['kind']} s{ns['c']['seed']}: token dt {core.dt} ({core.S} sub-steps), damping gamma dt "
          f"{core.gamma * core.dt:.2f}; field |w| dt median {float(w.median()) * core.dt:.2f}, max "
          f"{float(w.max()) * core.dt:.2f}; the token's rotation turns by up to {float(turn.abs().max()):.2f} rad; "
          f"pinning stiffness g_pin m_pin^2 dt^2 / <m> {pin:.2f}")
    sys.exit(0)

if "--momenta" in sys.argv:                 # post hoc: why a J pulse alone is almost a PJ pulse at sigma = 0
    m = float(model.core.physics().mass.mean().detach())
    p2 = {}
    for s in (0.0, 0.6):
        x, tg, sg = to_torch(tasks.test_set("A", T, s, n=512, namespace="decouple"))
        zs, _ = run(x, sg, tasks.derive_seed("decouple-noise", T, f"{s:.2f}"), "PJ", (), every=True)
        p2[s] = float(np.mean([(z.reshape(len(z), model.N, 3 * model.d)[..., 2 * model.d:].double() ** 2).mean()
                               for z in zs[10:]]))
    print(f"{ns['c']['kind']} s{ns['c']['seed']}: <P^2> over tokens 10-127 without pulses, against the thermal level "
          f"0.6^2 <m> = {0.36 * m:.3f}: sigma 0 {p2[0.0] / (0.36 * m):.3f}, sigma 0.6 {p2[0.6] / (0.36 * m):.3f}")
    sys.exit(0)

if "--selftest" in sys.argv:
    ok = True
    x, tg, sg = to_torch(tasks.test_set("A", T, 0.6, n=48, namespace="kinds-selftest"))
    for s in (0.0, 0.6):
        sig = torch.full_like(sg, s)
        with torch.no_grad():
            ref = model(x, sig, generator=torch.Generator().manual_seed(7))[:, -1]
        for kind in ("PJ", "J", "P"):
            _, mine = run(x, sig, 7, kind, ())
            same = torch.equal(ref, mine)
            print(f"{'ok ' if same else 'BAD'} {kind}, no pulse: the model's logits bit for bit at sigma {s}")
            ok &= same
        _, theirs = ns["run"](x, sig, 7, train(7))
        _, mine = run(x, sig, 7, "PJ", train(7))
        same = torch.equal(theirs, mine)
        print(f"{'ok ' if same else 'BAD'} a PJ train (n = 7) is decouple.py's run bit for bit at sigma {s}")
        ok &= same
    sig = torch.full_like(sg, 0.6)
    _, base = run(x, sig, 7, "PJ", ())
    for kind in ("J", "P", "off"):
        _, bitten = run(x, sig, 7, kind, (40,))
        moved = bool((bitten - base).abs().max() > 1e-4)
        print(f"{'ok ' if moved else 'BAD'} bite: {'one ' + kind + ' pulse' if kind != 'off' else 'no field'} "
              f"moves the logits")
        ok &= moved
    sys.exit(0 if ok else 1)

names = list(PROTOCOLS)
if "--only" in sys.argv:
    names = sys.argv[sys.argv.index("--only") + 1].split(",")
    GRID = [float(v) for v in sys.argv[sys.argv.index("--sigmas") + 1].split(",")]
    assert GRID[0] == 0.0, "r is measured against sigma = 0, so the sigma list starts there"
c = ns["c"]
res = {"ckpt": sys.argv[2], "kind": c["kind"], "seed": c["seed"], "T": T, "n_test": 1024, "n_cal": 1024,
       "sigma": GRID, "trace_sigma": 0.6}
for name in names:
    kind, n = PROTOCOLS[name]
    cal = {s: cell(name, s, "decouple-cal") for s in GRID}
    aware = {s: lda_fit(cal[s][0], cal[s][3]) for s in GRID}
    del cal
    first = cell(name, 0.0, "decouple", every=True)
    ref = first[1]                                                    # the sigma = 0 signal after every token
    row = {"kind": kind, "n": n, "pulses": list(train(n)), "own": [], "aware": [], "r": [], "norm": [], "abs": []}
    for s in GRID:
        z, sigs, pred, y = first if s == 0.0 else cell(name, s, "decouple", every=s == 0.6)
        sig = class_signal(z, y)
        row["own"].append(float((pred == y).mean()))
        row["aware"].append(float((lda_predict(aware[s], z) == y).mean()))
        row["r"].append(cos(sig, ref[-1]))
        row["norm"].append(float(np.linalg.norm(sig) / np.linalg.norm(ref[-1])))
        row["abs"].append(float(np.linalg.norm(sig)))
        if s == 0.6:
            row["trace"] = [round(cos(a, b), 4) for a, b in zip(sigs, ref)]
    row["anchor0"] = [round(cos(v, ref[3]), 4) for v in ref]           # sigma = 0: c(t) against c(3)
    row["abs0_t"] = [round(float(np.linalg.norm(v)), 4) for v in ref]
    res[name] = row
    print(f"{name}: aware " + " ".join(f"{s:.1f}:{a:.2f}" for s, a in zip(GRID, row["aware"])) + " r " + " ".join(
        f"{s:.1f}:{v:+.2f}" for s, v in zip(GRID, row["r"])) + f" anchor(127) {row['anchor0'][-1]:+.2f} "
        f"|c(127)| {row['abs0_t'][-1]:.2f}", flush=True)
json.dump(res, open(sys.argv[3], "w"), indent=1)
