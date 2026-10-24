"""The five kernel tests of GOAL decision 3. Each one is burned by the mutants named in mutants.json; a mutant counts
only when it fails a test here on an assertion (tools/run_mutants.py).

Run from phs-soliton/:  .venv/bin/python -m unittest tests.test_core
"""
from __future__ import annotations

import math
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from phs.core import Drive, PortHamiltonianCore, drive_force, energy, internal_force, inv_softplus, potential, q_top
from phs.models import PHSModel

torch.use_deterministic_algorithms(True)


def make_core(N, d, h, gamma=0.0, lattice=True, kappa=1.5, g_lat=0.8, g_pin=0.3, m_pin=2, mass=None, J=None,
              seed=0):
    """A float64 core with one sub-step per token (h = dt) and the given physical values. gyro is left at its
    default: the experiment's integrator is the one under test."""
    gen = torch.Generator().manual_seed(seed)
    core = PortHamiltonianCore(N, d, dt=h, substeps=1, gamma=gamma, lattice=lattice, m_pin=m_pin,
                               generator=gen).double()
    with torch.no_grad():
        core.w_kappa.fill_(inv_softplus(kappa) if kappa > 0 else -1e4)   # softplus(-1e4) == 0 exactly
        if mass is not None:
            core.w_mass.copy_(torch.log(torch.as_tensor(mass, dtype=torch.float64)))
        if J is not None:
            core.w_skew.copy_(torch.as_tensor(J, dtype=torch.float64) / 2)   # J = W - W^T with W = J/2
        if lattice:
            core.w_lat.fill_(inv_softplus(g_lat))
            core.w_pin.fill_(inv_softplus(g_pin))
    return core


def system(seed=20261024, N=6, d=2, j_scale=0.5):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((N, N)) * j_scale
    return dict(mass=np.exp(0.3 * rng.standard_normal(N)), J=W - W.T,
                U=rng.uniform(0, 1, (N, d)), Phi=rng.uniform(-np.pi, np.pi, (N, d)),
                theta0=rng.uniform(-np.pi, np.pi, (N, d)), z=rng.standard_normal((N, d)))


def run_energy(core, s, steps, every=1):
    """Constant drive, gamma = tau0 = 0: |H_t - H_0| sampled every `every` steps."""
    t64 = lambda a: torch.as_tensor(a, dtype=torch.float64)[None]
    drive = Drive(phi=t64(s["Phi"]), u=t64(s["U"]))
    with torch.no_grad():
        phys = core.physics()
        G = core.rotation(phys)
        theta = t64(s["theta0"])
        P = 0.7 * t64(s["z"]) * phys.mass.sqrt()[:, None]
        H0 = float(energy(theta, P, phys, drive)[0])
        f_int = internal_force(theta, phys)
        c1, c2 = core.ou_coefficients(torch.zeros(1, dtype=torch.float64))
        errs = []
        for i in range(steps):
            theta, P, f_int = core.token(theta, P, f_int, phys, G, drive, c1, c2, None)
            if (i + 1) % every == 0:
                errs.append(abs(float(energy(theta, P, phys, drive)[0]) - H0))
    return np.asarray(errs)


class Test1EnergyOrderAtFixedHorizon(unittest.TestCase):
    """gamma = tau0 = 0, J != 0, constant drive: second order at a fixed horizon, bounded on a long run."""

    HS = (0.04, 0.02, 0.01)
    HORIZON = 20.0

    def test_order_at_fixed_horizon(self):
        s = system()
        errs = [run_energy(make_core(6, 2, h, mass=s["mass"], J=s["J"]), s, int(round(self.HORIZON / h))).max()
                for h in self.HS]
        orders = [math.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
        for o in orders:
            self.assertGreaterEqual(o, 1.8, f"max |H_t - H_0| over t <= {self.HORIZON}: {errs}, orders {orders}")

    def test_bounded_long_run(self):
        s = system()
        errs = run_energy(make_core(6, 2, 0.04, mass=s["mass"], J=s["J"]), s, 10_000, every=50)
        first, whole = errs[: len(errs) // 10].max(), errs.max()
        self.assertLess(whole, 3 * first, f"10^4 steps of 0.04: first tenth {first:.3e}, whole {whole:.3e}")
        self.assertLess(whole, 0.05, f"10^4 steps of 0.04: max |H_t - H_0| {whole:.3e}")


class Test2Gibbs(unittest.TestCase):
    """F_cons = 0, gamma > 0, J != 0, unequal masses: Var(P_i) = tau0 m_i within 2% on >= 2e5 samples per node."""

    def test_variance_matches_tau0_mass(self):
        N, d, chains, tau0, gamma, h = 4, 4, 50_000, 0.09, 1.0, 0.1
        mass = np.array([0.5, 1.0, 1.7, 3.0])
        rng = np.random.default_rng(7)
        W = rng.standard_normal((N, N))
        core = make_core(N, d, h, gamma=gamma, lattice=False, kappa=0.0, mass=mass, J=W - W.T)
        zero = Drive(phi=torch.zeros(1, N, d, dtype=torch.float64), u=torch.zeros(1, N, d, dtype=torch.float64))
        gen = torch.Generator().manual_seed(11)
        with torch.no_grad():
            phys = core.physics()
            self.assertEqual(float(phys.kappa), 0.0)
            G = core.rotation(phys)
            theta = torch.zeros(chains, N, d, dtype=torch.float64)
            P = torch.zeros_like(theta)
            f_int = internal_force(theta, phys)
            c1, c2 = core.ou_coefficients(torch.full((chains,), math.sqrt(tau0), dtype=torch.float64))
            for _ in range(int(25 / (gamma * h))):   # gamma t = 25: the start is forgotten (e^-25)
                theta, P, f_int = core.token(theta, P, f_int, phys, G, zero, c1, c2, gen)
        samples = P.permute(1, 0, 2).reshape(N, -1)
        self.assertGreaterEqual(samples.shape[1], 200_000)
        bias = (samples.var(dim=1) / (tau0 * torch.as_tensor(mass)) - 1).numpy()
        for i, b in enumerate(bias):
            self.assertLess(abs(b), 0.02, f"node {i} (m = {mass[i]}): relative variance bias {b:+.4f}; all {bias}")


class Test3Charge(unittest.TestCase):
    """Q_top is an integer up to float32 rounding; figure 2's ring has Q = +1; a bond of exactly pi wraps to -pi."""

    def test_integer_on_random_float32(self):
        gen = torch.Generator().manual_seed(3)
        N, d = 16, 4
        theta = (torch.rand(20_000, N, d, generator=gen) * 2 - 1) * math.pi
        Q = q_top(theta)
        dev = float((Q - Q.round()).abs().max())
        self.assertLessEqual(dev, N * 2 ** -23 * 4, f"max |Q - round Q| = {dev:.3e}")
        self.assertGreater(float(Q.round().abs().max()), 0, "random tensors must include nonzero windings")

    def test_figure2_ring(self):
        for dtype in (torch.float32, torch.float64):
            theta = torch.tensor([-3, -1, 1, 3], dtype=dtype)[None, :, None] * math.pi / 4
            self.assertAlmostEqual(float(q_top(theta)[0, 0]), 1.0, places=5)

    def test_antipodal_half_open(self):
        for dtype in (torch.float32, torch.float64):
            theta = torch.tensor([-0.5, 0.5], dtype=dtype)[None, :, None] * math.pi
            self.assertAlmostEqual(float(q_top(theta)[0, 0]), -1.0, places=5,
                                   msg="bonds of exactly +-pi wrap into [-pi, pi): Q = -1")


class Test4Forces(unittest.TestCase):
    """F_cons = -grad V_total by autograd in float64 to 1e-8, for V_att, V_FK and V_ext separately and together."""

    def check(self, kappa, lattice, with_drive):
        gen = torch.Generator().manual_seed(5)
        B, N, d = 3, 7, 3
        core = make_core(N, d, 0.01, lattice=lattice, kappa=kappa, g_lat=0.9, g_pin=0.4, m_pin=3,
                         mass=np.exp(0.3 * np.arange(N) / N))
        phys = core.physics()
        theta = ((torch.rand(B, N, d, generator=gen, dtype=torch.float64) * 2 - 1) * math.pi).requires_grad_(True)
        drive = Drive(phi=(torch.rand(B, N, d, generator=gen, dtype=torch.float64) * 2 - 1) * math.pi,
                      u=torch.rand(B, N, d, generator=gen, dtype=torch.float64)) if with_drive else None
        (grad,) = torch.autograd.grad(potential(theta, phys, drive).sum(), theta)
        force = internal_force(theta, phys) + (drive_force(theta, drive) if with_drive else 0)
        err = float((force + grad).abs().max())
        self.assertLess(err, 1e-8, f"max |F + grad V| = {err:.3e}")
        self.assertGreater(float(grad.abs().max()), 1e-2, "the potential under test must not be flat")

    def test_attention(self):
        self.check(kappa=1.7, lattice=False, with_drive=False)

    def test_lattice(self):
        self.check(kappa=0.0, lattice=True, with_drive=False)

    def test_drive(self):
        self.check(kappa=0.0, lattice=False, with_drive=True)

    def test_total(self):
        self.check(kappa=1.7, lattice=True, with_drive=True)


class Test5BPTT(unittest.TestCase):
    """The autograd gradient of a tiny Soliton-PHS loss (float64, T = 3, S = 2, no noise) equals central finite
    differences for every parameter entry."""

    def test_gradient_matches_finite_differences(self):
        gen = torch.Generator().manual_seed(9)
        model = PHSModel(vocab=3, classes=2, N=3, d=2, dt=0.5, substeps=2, gamma=0.3, lattice=True, u_max=2.0,
                         m_pin=2, generator=gen).double()
        with torch.no_grad():
            model.E_u.normal_(0, 1, generator=gen)
        x = torch.tensor([[0, 1, 2], [2, 2, 1]])
        targets = torch.tensor([[0, 1, 1], [1, 0, 0]])
        sigma = torch.zeros(2, dtype=torch.float64)

        def loss():
            return F.cross_entropy(model(x, sigma).reshape(-1, 2), targets.reshape(-1))

        grads = torch.autograd.grad(loss(), list(model.parameters()))
        eps, worst = 1e-6, 0.0
        with torch.no_grad():
            for p, g in zip(model.parameters(), grads):
                flat, gflat = p.view(-1), g.reshape(-1)
                for i in range(flat.numel()):
                    keep = float(flat[i])
                    flat[i] = keep + eps
                    up = float(loss())
                    flat[i] = keep - eps
                    down = float(loss())
                    flat[i] = keep
                    fd = (up - down) / (2 * eps)
                    worst = max(worst, abs(fd - float(gflat[i])) / max(1e-3, abs(fd)))
        self.assertLess(worst, 1e-5, f"worst relative gap between autograd and finite differences {worst:.3e}")


if __name__ == "__main__":
    unittest.main()
