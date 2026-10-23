"""Numerical checks of the starter PDF's mathematical claims (sections 1-5 and 8).

Run from the repository root:

    uv run --no-project --with numpy python -I phs-soliton/starter/check_starter.py

Each check prints its predicate, the measured numbers and PASS/FAIL. A check that
expects a defect says so: it passes when the defect is reproduced. The reference
output is check_starter.out next to this file; regenerate it by redirecting the same command.
"""

import numpy as np

TAU = 2 * np.pi
results = []


def report(name, predicate, ok, detail):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}\n       predicate: {predicate}\n       measured:  {detail}")


def wrap(x):
    return (x + np.pi) % TAU - np.pi


def softmax(S):
    E = np.exp(S - S.max(axis=-1, keepdims=True))
    return E / E.sum(axis=-1, keepdims=True)


def pair_diffs(th):
    return th[:, None, :] - th[None, :, :]


def V_att(th, kappa, beta):
    S = (beta * kappa / th.shape[1]) * np.cos(pair_diffs(th)).sum(-1)
    top = S.max(-1)
    return -(top + np.log(np.exp(S - top[:, None]).sum(-1))).sum() / (2 * beta)


def F_att(th, kappa, beta, symmetrize=True):
    D = pair_diffs(th)
    A = softmax((beta * kappa / th.shape[1]) * np.cos(D).sum(-1))
    W = (A + A.T) / 2 if symmetrize else A
    return -(kappa / th.shape[1]) * (W[:, :, None] * np.sin(D)).sum(1)


def V_fk(th, g_lat, g_pin, m):
    bond = np.roll(th, -1, axis=0) - th
    return g_lat * (1 - np.cos(bond)).sum() + g_pin * (1 - np.cos(m * th)).sum()


def F_fk(th, g_lat, g_pin, m, lattice_sign=1.0):
    fwd = np.roll(th, -1, axis=0) - th
    bwd = th - np.roll(th, 1, axis=0)
    return lattice_sign * g_lat * (np.sin(fwd) - np.sin(bwd)) - m * g_pin * np.sin(m * th)


def num_grad(f, x, eps=1e-6):
    g = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        e = np.zeros_like(x)
        e[idx] = eps
        g[idx] = (f(x + e) - f(x - e)) / (2 * eps)
    return g


def cayley(K, t):
    I = np.eye(len(K))
    return np.linalg.solve(I - 0.5 * t * K, I + 0.5 * t * K)


def flow_kinetic(K, t):
    """Exact flow of dTheta = M^-1 P, dP = J M^-1 P in Q = M^-1/2 P coordinates.

    Returns (E, Phi): Q(t) = E Q0 and Theta(t) - Theta0 = M^-1/2 Phi Q0, where
    E = exp(tK) and Phi = int_0^t exp(sK) ds, both from the eigendecomposition of K.
    """
    mu, V = np.linalg.eig(K)  # K is real skew-symmetric, hence normal and diagonalizable
    e = np.exp(t * mu)
    phi = np.where(np.abs(mu) > 1e-12, (e - 1) / np.where(np.abs(mu) > 1e-12, mu, 1), t)
    Vinv = np.linalg.inv(V)
    E = (V @ np.diag(e) @ Vinv).real
    Phi = (V @ np.diag(phi) @ Vinv).real
    return E, Phi


def make_system(rng, N=6, d=2, J_scale=0.5):
    W = rng.standard_normal((N, N)) * J_scale
    return dict(
        m=np.exp(0.3 * rng.standard_normal(N)),
        J=W - W.T,
        kappa=1.5,
        g_lat=0.8,
        g_pin=0.3,
        m_pin=2,
        U=rng.uniform(0, 1, (N, d)),
        Phi=rng.uniform(-np.pi, np.pi, (N, d)),
    )


def forces(th, s):
    return (F_att(th, s["kappa"], 1.0) + F_fk(th, s["g_lat"], s["g_pin"], s["m_pin"])
            - s["U"] * np.sin(th - s["Phi"]))


def energy(th, P, s):
    return (0.5 * (P ** 2 / s["m"][:, None]).sum() + V_att(th, s["kappa"], 1.0)
            + V_fk(th, s["g_lat"], s["g_pin"], s["m_pin"]) + (s["U"] * (1 - np.cos(th - s["Phi"]))).sum())


class Integrator:
    """One token sub-step of BAOAB with the gyroscopic term handled three ways.

    pdf: Algorithm 1 as written, J M^-1 P added explicitly inside both B kicks.
    rot: B(V) / G(h/2) / A / O / A / G(h/2) / B(V), G an exact (Cayley) rotation.
    mag: B(V) / AG(h/2) / O / AG(h/2) / B(V), AG the exact flow of the kinetic part
         under the gyroscopic structure (drift and rotation together).
    """

    def __init__(self, s, h, mode, gamma=0.0, tau0=0.0, force=True):
        self.s, self.h, self.mode, self.force = s, h, mode, force
        sq = np.sqrt(s["m"])
        K = s["J"] / sq[:, None] / sq[None, :]
        self.c1 = np.exp(-gamma * h)
        self.c2 = np.sqrt(tau0 * (1 - self.c1 ** 2))
        self.sq = sq[:, None]
        if mode == "rot":
            self.G = sq[:, None] * cayley(K, h / 2) / sq[None, :]
        if mode == "mag":
            E, Phi = flow_kinetic(K, h / 2)
            self.E, self.Phi = E, Phi

    def F(self, th):
        return forces(th, self.s) if self.force else np.zeros_like(th)

    def gyro(self, P):
        return np.einsum("ij,...jd->...id", self.s["J"], P / self.sq ** 2)

    def kick(self, th, P):
        if self.mode == "pdf":
            return P + self.h / 2 * (self.F(th) + self.gyro(P))
        return P + self.h / 2 * self.F(th)

    def half_kinetic(self, th, P):
        if self.mode == "mag":
            Q = P / self.sq
            th = wrap(th + np.einsum("ij,...jd->...id", self.Phi, Q) / self.sq)
            return th, np.einsum("ij,...jd->...id", self.E, Q) * self.sq
        if self.mode == "rot":
            P = np.einsum("ij,...jd->...id", self.G, P)
        return wrap(th + self.h / 2 * P / self.sq ** 2), P

    def half_kinetic_back(self, th, P):
        if self.mode == "rot":
            th = wrap(th + self.h / 2 * P / self.sq ** 2)
            return th, np.einsum("ij,...jd->...id", self.G, P)
        return self.half_kinetic(th, P)

    def step(self, th, P, rng=None):
        P = self.kick(th, P)
        th, P = self.half_kinetic(th, P)
        if self.c2 > 0 or self.c1 != 1.0:
            P = self.c1 * P + self.c2 * self.sq * rng.standard_normal(P.shape)
        th, P = self.half_kinetic_back(th, P)
        return th, self.kick(th, P)


def max_energy_error(s, th0, P0, h, steps, mode):
    integ = Integrator(s, h, mode)
    th, P, H0, worst = th0.copy(), P0.copy(), energy(th0, P0, s), 0.0
    for _ in range(steps):
        th, P = integ.step(th, P)
        worst = max(worst, abs(energy(th, P, s) - H0))
    return worst


def main():
    rng = np.random.default_rng(20261024)

    # C1: section 1 gradient of the von Mises attention potential
    th = rng.uniform(-np.pi, np.pi, (5, 3))
    g = num_grad(lambda x: V_att(x, 1.7, 0.8), th)
    err_sym = np.abs(F_att(th, 1.7, 0.8) + g).max()
    err_one = np.abs(F_att(th, 1.7, 0.8, symmetrize=False) + g).max()
    report("C1 F_att = -grad V_att needs the symmetrized kernel (A_ij + A_ji)",
           "max|F_att + grad V| < 1e-7 for (A+A^T)/2, and > 1e-3 for A alone",
           err_sym < 1e-7 and err_one > 1e-3, f"symmetrized {err_sym:.2e}, A alone {err_one:.2e}")

    # C2: sections 5 and 8 Frenkel-Kontorova force
    th = rng.uniform(-np.pi, np.pi, (7, 2))
    g = num_grad(lambda x: V_fk(x, 0.9, 0.4, 3), th)
    err = np.abs(F_fk(th, 0.9, 0.4, 3) + g).max()
    err_flip = np.abs(F_fk(th, 0.9, 0.4, 3, lattice_sign=-1.0) + g).max()
    report("C2 Algorithm 1 F_FK = -grad V_FK",
           "max|F_FK + grad V_FK| < 1e-7, and > 1e-3 with the lattice sign flipped",
           err < 1e-7 and err_flip > 1e-3, f"as written {err:.2e}, sign flipped {err_flip:.2e}")

    # C3: section 2 gyroscopic identities
    s = make_system(rng)
    P = rng.standard_normal((6, 2))
    v = P / s["m"][:, None]
    power = np.einsum("ik,ij,jk->", v, s["J"], v)
    trace = np.trace(s["J"] / s["m"][None, :])
    report("C3 gyroscopic force does no work and has zero divergence in P",
           "|sum_k v_k^T J v_k| < 1e-12 and |Tr(J M^-1)| < 1e-12",
           abs(power) < 1e-12 and abs(trace) < 1e-12, f"power {power:.1e}, trace {trace:.1e}")

    # C4: section 8 unit test (1), energy conservation order with J != 0
    th0 = rng.uniform(-np.pi, np.pi, (6, 2))
    P0 = 0.7 * rng.standard_normal((6, 2)) * np.sqrt(s["m"])[:, None]
    hs = [0.04, 0.02, 0.01]
    horizon = 20.0
    fixed_time, fixed_steps = {}, {}
    for mode in ("pdf", "rot", "mag"):
        fixed_time[mode] = [max_energy_error(s, th0, P0, h, int(round(horizon / h)), mode) for h in hs]
        fixed_steps[mode] = [max_energy_error(s, th0, P0, h, 1000, mode) for h in hs]

    def orders(errs):
        return [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]

    o_steps = {k: orders(v) for k, v in fixed_steps.items()}
    o_time = {k: orders(v) for k, v in fixed_time.items()}
    report("C4a the PDF's test (10^3 steps, vary dt) cannot see the explicit gyro kick",
           "observed order >= 1.8 for all three integrators at a fixed step count",
           all(min(o) >= 1.8 for o in o_steps.values()),
           "; ".join(f"{k}: errs {', '.join(f'{e:.2e}' for e in fixed_steps[k])}, orders "
                     f"{', '.join(f'{x:.2f}' for x in o_steps[k])}" for k in fixed_steps))
    report("C4b at a fixed horizon the PDF integrator loses second order (expected defect)",
           f"horizon {horizon}: pdf error >= 100x rot error at every dt, pdf order < 1.5 on the finest pair, "
           "rot and mag orders >= 1.8",
           all(p >= 100 * r for p, r in zip(fixed_time["pdf"], fixed_time["rot"])) and o_time["pdf"][-1] < 1.5
           and min(o_time["rot"]) >= 1.8 and min(o_time["mag"]) >= 1.8,
           "; ".join(f"{k}: errs {', '.join(f'{e:.2e}' for e in fixed_time[k])}, orders "
                     f"{', '.join(f'{x:.2f}' for x in o_time[k])}" for k in fixed_time))

    long_h, long_horizon = 0.04, 2000.0
    sq = np.sqrt(s["m"])
    K = s["J"] / sq[:, None] / sq[None, :]
    lam = np.abs(np.linalg.eigvals(K)).max()
    h = 0.04
    explicit = np.linalg.norm(np.eye(6) + h / 2 * K, 2)
    rotation = np.linalg.norm(cayley(K, h / 2), 2)
    report("C4d the explicit gyro half-kick expands momenta; the Cayley rotation does not (expected defect)",
           "||I + (h/2)K||_2 = sqrt(1 + (h*lambda_max/2)^2) > 1 and ||cayley(K, h/2)||_2 = 1, K = M^-1/2 J M^-1/2",
           abs(explicit - np.sqrt(1 + (h * lam / 2) ** 2)) < 1e-12 and explicit > 1 and abs(rotation - 1) < 1e-12,
           f"lambda_max {lam:.3f}, h {h}: explicit {explicit:.6f}, Cayley {rotation:.15f}; with F = 0 the "
           f"kinetic energy of the top mode grows by explicit^4 per step, x{explicit ** (4 * long_horizon / h):.1e} "
           f"over the C4c horizon")

    drift = {}
    for mode in ("pdf", "rot", "mag"):
        integ = Integrator(s, long_h, mode)
        th, P, H0 = th0.copy(), P0.copy(), energy(th0, P0, s)
        trace_err = []
        for i in range(int(long_horizon / long_h)):
            th, P = integ.step(th, P)
            if i % 500 == 0:
                trace_err.append(energy(th, P, s) - H0)
        drift[mode] = (np.abs(trace_err[: len(trace_err) // 10]).max(), np.abs(trace_err).max())
    report("C4c long run: pdf diverges, rot and mag stay bounded (expected defect)",
           f"dt {long_h}, horizon {long_horizon}: pdf max error grows > 5x from the first tenth to the whole run; "
           "rot and mag grow < 3x",
           drift["pdf"][1] > 5 * drift["pdf"][0] and all(drift[k][1] < 3 * drift[k][0] for k in ("rot", "mag")),
           "; ".join(f"{k}: first tenth {a:.2e}, whole {b:.2e}" for k, (a, b) in drift.items()))

    # C5: section 8 unit test (2), Gibbs momentum variance with J != 0
    N, chains = 4, 200_000
    sg = make_system(rng, N=N, d=1, J_scale=1.0)
    sg["m"] = np.array([0.5, 1.0, 2.0, 4.0])
    tau0, gamma, h = 0.1, 1.0, 0.1
    bias = {}
    for label, mode, J in (("pdf, J=0", "pdf", np.zeros((N, N))), ("pdf, J!=0", "pdf", sg["J"]),
                           ("rot, J!=0", "rot", sg["J"])):
        sys_ = dict(sg, J=J)
        integ = Integrator(sys_, h, mode, gamma=gamma, tau0=tau0, force=False)
        th = np.zeros((chains, N, 1))
        P = np.sqrt(tau0 * sys_["m"])[None, :, None] * rng.standard_normal((chains, N, 1))
        acc = np.zeros(N)
        for i in range(400):
            th, P = integ.step(th, P, rng)
            if i >= 200 and i % 10 == 0:
                acc += (P[:, :, 0] ** 2).mean(0)
        bias[label] = acc / 20 / (tau0 * sys_["m"]) - 1
    worst = {k: np.abs(v).max() for k, v in bias.items()}
    report("C5 the Gibbs test passes only with J = 0 for the PDF integrator (expected defect)",
           "max relative variance bias < 2% for pdf with J=0 and rot with J!=0; > 2% for pdf with J!=0",
           worst["pdf, J=0"] < 0.02 and worst["rot, J!=0"] < 0.02 and worst["pdf, J!=0"] > 0.02,
           "; ".join(f"{k}: {', '.join(f'{x:+.3f}' for x in v)}" for k, v in bias.items()))

    # C6: section 4 and unit test (3), integrality of the winding number
    th32 = rng.uniform(-np.pi, np.pi, (20_000, 16, 4)).astype(np.float32)
    pi32 = np.float32(np.pi)
    w = ((np.roll(th32, -1, axis=1) - th32 + pi32) % np.float32(TAU)) - pi32
    Q = w.sum(1) / np.float32(TAU)
    dev = np.abs(Q - np.round(Q)).max()
    n = np.round((w - (np.roll(th32, -1, axis=1) - th32)) / np.float32(TAU))
    report("C6 Q_top is an integer up to float32 rounding; n_ik in {-1, 0, 1}",
           "max|Q - round Q| < 16 * 2^-23 on 20000 random float32 configurations (N=16, d=4)",
           dev < 16 * 2.0 ** -23 and set(np.unique(n)) <= {-1.0, 0.0, 1.0},
           f"max deviation {dev:.2e}, n values {sorted(set(np.unique(n).tolist()))}")
    anti = np.array([[-np.pi / 2], [np.pi / 2]])
    q_anti = wrap(np.roll(anti, -1, 0) - anti).sum() / TAU
    report("C6b at |bond| = pi the value of Q is fixed by the half-open convention, not by geometry",
           "two antipodal nodes give Q = -1 under wrap into [-pi, pi) (mirror image would give +1)",
           abs(q_anti + 1) < 1e-12, f"Q = {q_anti:+.0f}")

    # C7: figure 2 example sits on the stability boundary of section 5
    fig = np.array([[-3 * np.pi / 4], [-np.pi / 4], [np.pi / 4], [3 * np.pi / 4]])
    bonds = wrap(np.roll(fig, -1, 0) - fig)
    q_fig = bonds.sum() / TAU
    report("C7 figure 2's Q = +1 ring (N = 4) has bond curvature cos(pi/2) = 0",
           "Q = 1 and every bond Hessian g_lat*cos(bond) is 0, i.e. |Q| = N/4, not < N/4",
           abs(q_fig - 1) < 1e-12 and np.abs(np.cos(bonds)).max() < 1e-12,
           f"Q = {q_fig:+.0f}, max |cos(bond)| = {np.abs(np.cos(bonds)).max():.1e}")

    # C8: section 2 Sakaguchi example (N = 2, d = 1, alpha = pi/2, theta = 0)
    alpha, C, kappa = np.pi / 2, 1.0, 1.0

    def F_rot(x):
        return C * np.sin(np.roll(x, -1, 0) - x + alpha)

    def div(f, x):
        return sum((f(x + e)[i, 0] - f(x - e)[i, 0]) / 2e-6
                   for i, e in ((i, np.eye(2)[:, [i]] * 1e-6) for i in range(2)))

    zero = np.zeros((2, 1))
    d_rot, d_att = div(F_rot, zero), div(lambda x: F_att(x, kappa, 1.0), zero)
    x = np.array([[0.0], [0.7]])
    jac = np.array([[(F_rot(x + np.eye(2)[:, [j]] * 1e-6)[i, 0] - F_rot(x - np.eye(2)[:, [j]] * 1e-6)[i, 0]) / 2e-6
                     for j in range(2)] for i in range(2)])
    report("C8 section 2: div F_diff != 0 comes from the conservative F_att; F_rot is non-potential by its curl",
           "div F_rot(0) = 0, div F_att(0) = -kappa, Jacobian of F_rot asymmetric at theta = (0, 0.7)",
           abs(d_rot) < 1e-8 and abs(d_att + kappa) < 1e-6 and abs(jac[0, 1] - jac[1, 0]) > 0.5,
           f"div F_rot {d_rot:+.1e}, div F_att {d_att:+.3f}, dF1/dth2 - dF2/dth1 = {jac[0, 1] - jac[1, 0]:+.3f}")

    print(f"\n{sum(results)}/{len(results)} checks PASS")
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
