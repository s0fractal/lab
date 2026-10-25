"""PortHamiltonianCore: the Pure-PHS / Soliton-PHS state update of the starter's Algorithm 1, with the gyroscopic
term integrated as an exact Cayley rotation (GOAL decision 2) instead of the explicit kick of the PDF.

State: Theta, P of shape (B, N, d). Masses M = exp(w_M) per node, J_PP = W - W^T (N x N), beta = 1.
One sub-step of length h = dt / S:

    B(F_cons, h/2) -> G(h/2) -> A(h/2) -> O(h) -> A(h/2) -> G(h/2) -> B(F_cons, h/2)

G: P <- M^1/2 cay((h/2) K) M^-1/2 P with K = M^-1/2 J M^-1/2 and cay(X) = (I - X/2)^-1 (I + X/2).
gyro="explicit" is the PDF's variant (J M^-1 P added inside both B kicks, no G); it exists only as a negative
control for the tests and is refused by the experiment runners.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

TAU = 2 * math.pi


def wrap(x: torch.Tensor) -> torch.Tensor:
    """Principal value in the half-open interval [-pi, pi)."""
    return torch.remainder(x + math.pi, TAU) - math.pi


def q_top(theta: torch.Tensor) -> torch.Tensor:
    """Winding number per channel on the closed ring, (B, N, d) -> (B, d): (1/2pi) sum_i wrap(theta_{i+1} - theta_i)."""
    bonds = wrap(torch.roll(theta, -1, dims=-2) - theta)
    return bonds.sum(dim=-2) / TAU


def inv_softplus(y: float) -> float:
    return y + math.log(-math.expm1(-y))


@dataclass
class Physics:
    """Constrained physical parameters of one forward pass (shared by every sub-step)."""
    kappa: torch.Tensor          # scalar >= 0
    g_lat: torch.Tensor          # scalar >= 0 (0 for Pure-PHS)
    g_pin: torch.Tensor          # scalar >= 0 (0 for Pure-PHS)
    m_pin: int                   # pinning multiplicity
    mass: torch.Tensor           # (N,) > 0
    J: torch.Tensor              # (N, N) skew-symmetric
    lattice: bool


@dataclass
class Drive:
    """Input port of one token: target phases Phi and pump amplitudes U, both (B, N, d)."""
    phi: torch.Tensor
    u: torch.Tensor


def attention_force(theta: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """F_att = -(kappa/d) sum_j A_sym_ij sin(theta_ik - theta_jk), A = softmax_j(kappa C), C_ij = (1/d) sum_k cos.

    Uses sin(a - b) = sin a cos b - cos a sin b, so no (B, N, N, d) tensor is formed."""
    d = theta.shape[-1]
    c, s = torch.cos(theta), torch.sin(theta)
    C = (c @ c.transpose(-1, -2) + s @ s.transpose(-1, -2)) / d
    A = torch.softmax(kappa * C, dim=-1)
    A_sym = 0.5 * (A + A.transpose(-1, -2))
    return -(kappa / d) * (s * (A_sym @ c) - c * (A_sym @ s))


def lattice_force(theta: torch.Tensor, g_lat: torch.Tensor, g_pin: torch.Tensor, m_pin: int) -> torch.Tensor:
    """F_FK = g_lat [sin(theta_{i+1} - theta_i) - sin(theta_i - theta_{i-1})] - m g_pin sin(m theta_i)."""
    fwd = torch.roll(theta, -1, dims=-2) - theta
    bwd = theta - torch.roll(theta, 1, dims=-2)
    return g_lat * (torch.sin(fwd) - torch.sin(bwd)) - m_pin * g_pin * torch.sin(m_pin * theta)


def drive_force(theta: torch.Tensor, drive: Drive) -> torch.Tensor:
    return -drive.u * torch.sin(theta - drive.phi)


def potential(theta: torch.Tensor, phys: Physics, drive: Drive | None) -> torch.Tensor:
    """V_total per batch element: V_att + V_FK + V_ext, (B,)."""
    d = theta.shape[-1]
    c, s = torch.cos(theta), torch.sin(theta)
    C = (c @ c.transpose(-1, -2) + s @ s.transpose(-1, -2)) / d
    v = -0.5 * torch.logsumexp(phys.kappa * C, dim=-1).sum(-1)
    if phys.lattice:
        bond = torch.roll(theta, -1, dims=-2) - theta
        v = v + phys.g_lat * (1 - torch.cos(bond)).sum((-2, -1))
        v = v + phys.g_pin * (1 - torch.cos(phys.m_pin * theta)).sum((-2, -1))
    if drive is not None:
        v = v + (drive.u * (1 - torch.cos(theta - drive.phi))).sum((-2, -1))
    return v


def energy(theta: torch.Tensor, P: torch.Tensor, phys: Physics, drive: Drive | None) -> torch.Tensor:
    """H = 1/2 Tr(P^T M^-1 P) + V_total per batch element, (B,)."""
    kinetic = 0.5 * (P ** 2 / phys.mass[:, None]).sum((-2, -1))
    return kinetic + potential(theta, phys, drive)


def internal_force(theta: torch.Tensor, phys: Physics) -> torch.Tensor:
    f = attention_force(theta, phys.kappa)
    if phys.lattice:
        f = f + lattice_force(theta, phys.g_lat, phys.g_pin, phys.m_pin)
    return f


def cayley_rotation(phys: Physics, h: float) -> torch.Tensor:
    """G = M^1/2 cay(h K) M^-1/2, K = M^-1/2 J M^-1/2; orthogonal in Q = M^-1/2 P, so |Q| is kept exactly."""
    sq = phys.mass.sqrt()
    K = phys.J / (sq[:, None] * sq[None, :])
    eye = torch.eye(K.shape[0], dtype=K.dtype, device=K.device)
    cay = torch.linalg.solve(eye - 0.5 * h * K, eye + 0.5 * h * K)
    return sq[:, None] * cay / sq[None, :]


class PortHamiltonianCore(nn.Module):
    """Pure-PHS (lattice=False: g_lat = g_pin = 0) or Soliton-PHS (lattice=True) on the cotangent torus."""

    def __init__(self, N: int, d: int, dt: float, substeps: int, gamma: float, lattice: bool,
                 m_pin: int = 2, kappa0: float = 1.0, g_lat0: float = 1.0, g_pin0: float = 0.5,
                 j_scale: float = 0.1, gyro: str = "cayley", generator: torch.Generator | None = None):
        super().__init__()
        if gyro not in ("cayley", "explicit"):
            raise ValueError(f"gyro must be 'cayley' or 'explicit', not {gyro!r}")
        self.N, self.d, self.dt, self.S, self.gamma = N, d, dt, substeps, gamma
        self.lattice, self.m_pin, self.gyro = lattice, m_pin, gyro
        self.h = dt / substeps
        self.w_kappa = nn.Parameter(torch.tensor(inv_softplus(kappa0)))
        self.w_mass = nn.Parameter(torch.zeros(N))
        self.w_skew = nn.Parameter(j_scale * torch.randn(N, N, generator=generator))
        if lattice:
            self.w_lat = nn.Parameter(torch.tensor(inv_softplus(g_lat0)))
            self.w_pin = nn.Parameter(torch.tensor(inv_softplus(g_pin0)))

    def physics(self) -> Physics:
        zero = torch.zeros((), dtype=self.w_kappa.dtype, device=self.w_kappa.device)
        return Physics(kappa=F.softplus(self.w_kappa),
                       g_lat=F.softplus(self.w_lat) if self.lattice else zero,
                       g_pin=F.softplus(self.w_pin) if self.lattice else zero,
                       m_pin=self.m_pin, mass=torch.exp(self.w_mass),
                       J=self.w_skew - self.w_skew.T, lattice=self.lattice)

    def ou_coefficients(self, sigma: torch.Tensor) -> tuple[float, torch.Tensor]:
        """c1 = exp(-gamma h), c2 = sqrt(tau0 (1 - c1^2)) with tau0 = sigma^2; sigma has shape (B,)."""
        c1 = math.exp(-self.gamma * self.h)
        return c1, (sigma ** 2 * (1 - c1 ** 2)).sqrt()[:, None, None]

    def token(self, theta: torch.Tensor, P: torch.Tensor, f_int: torch.Tensor, phys: Physics, G: torch.Tensor | None,
              drive: Drive, c1: float, c2: torch.Tensor, generator: torch.Generator | None):
        """S sub-steps under one token's drive. f_int is the internal force at the incoming theta; returns
        (theta, P, f_int) at the end, so the next token reuses the last force evaluation."""
        h = self.h
        inv_m = (1.0 / phys.mass)[:, None]
        sqrt_m = phys.mass.sqrt()[:, None]
        noisy = bool((c2 > 0).any())
        f = f_int + drive_force(theta, drive)
        for _ in range(self.S):
            if self.gyro == "explicit":
                P = P + 0.5 * h * (f + phys.J @ (inv_m * P))
            else:
                P = P + 0.5 * h * f
                P = G @ P
            theta = wrap(theta + 0.5 * h * inv_m * P)
            if noisy:
                xi = torch.randn(P.shape, generator=generator, dtype=P.dtype, device=P.device)
                P = c1 * P + c2 * sqrt_m * xi
            elif c1 != 1.0:
                P = c1 * P
            theta = wrap(theta + 0.5 * h * inv_m * P)
            f_int = internal_force(theta, phys)
            f = f_int + drive_force(theta, drive)
            if self.gyro == "explicit":
                P = P + 0.5 * h * (f + phys.J @ (inv_m * P))
            else:
                P = G @ P
                P = P + 0.5 * h * f
        return theta, P, f_int

    def rotation(self, phys: Physics) -> torch.Tensor | None:
        return None if self.gyro == "explicit" else cayley_rotation(phys, 0.5 * self.h)
