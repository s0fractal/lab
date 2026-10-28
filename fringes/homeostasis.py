"""Homeostatic clock (post hoc play): can the Soliton-PHS memory correct its own rate without being told sigma?
The model reads its own temperature from its state: D(t) = mean_i,k cos(m theta_ik), the Debye-Waller factor of the
pinning spring, and at every token rescales g_pin -> g_pin * D0(t) / D(t), with D0(t) the sigma = 0 profile from a
separate calibration set. Nothing in the loop knows sigma.

  none        the trained model
  self        D(t) per sequence, smoothed by an EMA over tokens: one sequence regulates itself
  population  D(t) averaged over the 1024 sequences of the cell: a shared thermometer (still blind to sigma)

homeostasis.py --selftest LAB checks the timing on an untrained model, no checkpoint: a single noiseless sequence
calibrated on itself must be a fixed point (factor exactly 1 at every token, logits bit-identical to the plain run),
and the profile shifted by one token, as before the Codex review (F4), must break it."""
import json
import sys

import torch

SELFTEST = sys.argv[1] == "--selftest"
LAB = sys.argv[2] if SELFTEST else sys.argv[1]
sys.path.insert(0, f"{LAB}/phs-soliton")
from phs import tasks  # noqa: E402
from phs.core import Drive, Physics, internal_force, wrap  # noqa: E402
from phs.models import PHSModel, build  # noqa: E402
from phs.train import to_torch  # noqa: E402

EMA = 0.3


@torch.no_grad()
def run(x, sg, mode, D0=None, D0e=None, groups=None, seed=13):
    """Returns logits and two profiles read before each token t: the raw mean of D and the mean of the per-sequence
    EMA sensor. The controller at token t uses the sensor read before token t, against the profile entry t of the
    same kind (raw for population, EMA for self), so both sides carry the same lag and the same smoothing."""
    B = x.shape[0]
    core, phys = model.core, model.core.physics()
    G = core.rotation(phys)
    c1, c2 = core.ou_coefficients(sg)
    gen = torch.Generator().manual_seed(seed)
    theta = torch.zeros(B, model.N, model.d)
    P = torch.zeros_like(theta)
    Dema = torch.ones(B)
    prof, prof_ema = [], []
    for t in range(T):
        D = torch.cos(mp * theta).mean((1, 2))
        Dema = D if t == 0 else (1 - EMA) * Dema + EMA * D
        prof.append(float(D.mean()))
        prof_ema.append(float(Dema.mean()))
        fac = torch.ones(B)                    # every mode takes the same arithmetic path; only fac differs
        ref = lambda v: torch.tensor(v, dtype=D.dtype)  # noqa: E731  (float / tensor is a reciprocal times float)
        if mode == "self":
            fac = (ref(D0e[t]) / Dema.clamp_min(0.05)).clamp(0.5, 2.0)
        elif mode == "population":
            pop = D.reshape(groups, -1).mean(1).repeat_interleave(B // groups)
            fac = (ref(D0[t]) / pop.clamp_min(0.05)).clamp(0.5, 2.0)
        ph = Physics(kappa=phys.kappa, g_lat=phys.g_lat, g_pin=phys.g_pin * fac[:, None, None], m_pin=phys.m_pin,
                     mass=phys.mass, J=phys.J, lattice=phys.lattice)
        f = internal_force(theta, ph)
        tok = x[:, t]
        drive = Drive(phi=wrap(model.E_phi[tok]), u=model.u_max * torch.sigmoid(model.E_u[tok]))
        theta, P, f = core.token(theta, P, f, ph, G, drive, c1, c2, gen)
    z = torch.cat([torch.cos(theta), torch.sin(theta), P], -1).reshape(B, -1)
    return model.readout(z), prof, prof_ema


def selftest():
    global model, T, mp
    model = PHSModel(9, 7, 6, 2, 0.2, 4, 0.3, True, 3.0, generator=torch.Generator().manual_seed(5)).eval()
    T, mp = 12, model.core.m_pin
    x, s0 = torch.tensor([[3, 3, 0, 5, 0, 0, 1, 0, 0, 0, 0, 8]]), torch.zeros(1)
    ref, D0, D0e = run(x, s0, "none")
    ok = {"theta starts at 0: the first reading is 1": D0[0] == 1.0 and D0e[0] == 1.0,
          "token 0 moves the reading before token 1": D0[1] != 1.0}
    for mode in ("self", "population"):
        lg = run(x, s0, mode, D0=D0, D0e=D0e, groups=1)[0]
        ok[f"{mode}: calibrated on itself it is a fixed point"] = torch.equal(lg, ref)
    lag = run(x, s0, "self", D0e=[1.0] + D0e[:-1])[0]
    gap = float((lag - ref).abs().max())
    ok[f"bite: the one-token lag breaks the fixed point (logits move {gap:.1e})"] = gap > 0
    for k, v in ok.items():
        print(f"{'ok ' if v else 'BAD'} {k}")
    sys.exit(0 if all(ok.values()) else 1)


if SELFTEST:
    selftest()
ckpt, out = sys.argv[2], sys.argv[3]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
EMA = float(sys.argv[5]) if len(sys.argv) > 5 else 0.3
torch.set_num_threads(int(sys.argv[6]) if len(sys.argv) > 6 else 1)
MODES = sys.argv[7].split(",") if len(sys.argv) > 7 else ["none", "self", "population"]
c = torch.load(ckpt)
spec = tasks.TASKS["A"]
model = build(c["kind"], spec["vocab"], spec["classes"], c["cfg"], torch.Generator().manual_seed(0))
model.load_state_dict(c["state"])
model.eval()
T, mp = c["T"], model.core.m_pin
SIGMAS = [round(0.05 * i, 2) for i in range(25)]

# calibration: the sigma = 0 profiles of D(t), read before token t, on their own sequences (no extra shift: an
# earlier version shifted this pre-token profile by one more token, so token t was compared with t - 1)
xc, _, sc = to_torch(tasks.test_set("A", T, 0.0, n=n, namespace="calib-homeo"))
_, D0, D0e = run(xc, sc, "none")
assert D0[0] == 1.0 and D0e[0] == 1.0, "theta starts at 0, so the first reading must be exactly 1"
cells = [to_torch(tasks.test_set("A", T, s, n=n, namespace="comp")) for s in SIGMAS]
X, Y, S = torch.cat([b[0] for b in cells]), torch.cat([b[1][:, -1] for b in cells]), torch.cat([b[2] for b in cells])
res = {}
for mode in MODES:
    logits, _, _ = run(X, S, mode, D0=D0, D0e=D0e, groups=len(SIGMAS))
    res[mode] = [round(float(v), 4) for v in (logits.argmax(1) == Y).double().reshape(len(SIGMAS), n).mean(1)]
    print(mode.ljust(10), " ".join(f"{a:.2f}" for a in res[mode][::2]), flush=True)
print("sigma     ", " ".join(f"{s:.2f}" for s in SIGMAS[::2]))
json.dump({"kind": c["kind"], "ckpt": ckpt, "n": n, "ema": EMA, "sigma": SIGMAS, "D0": D0, "D0_ema": D0e, "acc": res,
           "version": 2}, open(out, "w"))
