"""The four models of the experiment: Pure-PHS, Soliton-PHS, Control-GRU (D_state = 2Nd) and Control-GRU-pm
(hidden size whose parameter count is closest to Soliton-PHS). Every model maps tokens (B, T) and a noise level
sigma (B,) to logits (B, T, C) and diagnostics."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .core import Drive, PortHamiltonianCore, internal_force, q_top, wrap


class PHSModel(nn.Module):
    """Algorithm 1 with the Cayley gyroscopic step: input port E_phi, E_u; Theta_0 = P_0 = 0; readout on
    z_t = [cos Theta_t, sin Theta_t, P_t]."""

    family = "phs"

    def __init__(self, vocab: int, classes: int, N: int, d: int, dt: float, substeps: int, gamma: float,
                 lattice: bool, u_max: float, m_pin: int = 2, kappa0: float = 1.0, g_lat0: float = 1.0,
                 g_pin0: float = 0.5, j_scale: float = 0.1, generator: torch.Generator | None = None):
        super().__init__()
        self.N, self.d, self.u_max = N, d, u_max
        self.core = PortHamiltonianCore(N, d, dt, substeps, gamma, lattice, m_pin=m_pin, kappa0=kappa0,
                                        g_lat0=g_lat0, g_pin0=g_pin0, j_scale=j_scale, generator=generator)
        self.E_phi = nn.Parameter(torch.empty(vocab, N, d).uniform_(-math.pi, math.pi, generator=generator))
        self.E_u = nn.Parameter(torch.zeros(vocab, N, d))
        self.readout = nn.Linear(3 * N * d, classes)
        with torch.no_grad():
            bound = 1 / math.sqrt(3 * N * d)
            self.readout.weight.uniform_(-bound, bound, generator=generator)
            self.readout.bias.uniform_(-bound, bound, generator=generator)

    @property
    def state_dim(self) -> int:
        return 2 * self.N * self.d

    def forward(self, x: torch.Tensor, sigma: torch.Tensor, generator: torch.Generator | None = None,
                diagnostics: bool = False):
        B, T = x.shape
        core = self.core
        phys = core.physics()
        G = core.rotation(phys)
        c1, c2 = core.ou_coefficients(sigma.to(self.E_u.dtype))
        theta = torch.zeros(B, self.N, self.d, dtype=self.E_u.dtype)
        P = torch.zeros_like(theta)
        f_int = internal_force(theta, phys)
        logits, charges = [], []
        for t in range(T):
            tok = x[:, t]
            drive = Drive(phi=wrap(self.E_phi[tok]), u=self.u_max * torch.sigmoid(self.E_u[tok]))
            theta, P, f_int = core.token(theta, P, f_int, phys, G, drive, c1, c2, generator)
            z = torch.cat([torch.cos(theta), torch.sin(theta), P], dim=-1).reshape(B, -1)
            logits.append(self.readout(z))
            if diagnostics:
                charges.append(q_top(theta.detach()))
        out = torch.stack(logits, dim=1)
        if not diagnostics:
            return out
        return out, {"q_top": torch.stack(charges, dim=1), "theta_T": theta.detach()}

    def learned_physics(self) -> dict:
        phys = self.core.physics()
        return {"kappa": float(phys.kappa), "g_lat": float(phys.g_lat), "g_pin": float(phys.g_pin),
                "gamma": self.core.gamma, "mass_min": float(phys.mass.min()), "mass_max": float(phys.mass.max()),
                "J_norm": float(torch.linalg.matrix_norm(phys.J, ord=2))}


class ControlGRU(nn.Module):
    """h_t = GRUCell(onehot(x_t), h_{t-1}) + sigma xi_t, xi_t ~ N(0, I) once per token; logits = h_t W_out + b."""

    family = "gru"

    def __init__(self, vocab: int, classes: int, hidden: int, generator: torch.Generator | None = None):
        super().__init__()
        self.vocab, self.hidden = vocab, hidden
        self.cell = nn.GRUCell(vocab, hidden)
        self.readout = nn.Linear(hidden, classes)
        with torch.no_grad():
            for p in self.cell.parameters():
                bound = 1 / math.sqrt(hidden)
                p.uniform_(-bound, bound, generator=generator)
            self.readout.weight.uniform_(-1 / math.sqrt(hidden), 1 / math.sqrt(hidden), generator=generator)
            self.readout.bias.uniform_(-1 / math.sqrt(hidden), 1 / math.sqrt(hidden), generator=generator)

    @property
    def state_dim(self) -> int:
        return self.hidden

    def forward(self, x: torch.Tensor, sigma: torch.Tensor, generator: torch.Generator | None = None,
                diagnostics: bool = False):
        B, T = x.shape
        dtype = self.readout.weight.dtype
        h = torch.zeros(B, self.hidden, dtype=dtype)
        onehot = torch.nn.functional.one_hot(x, self.vocab).to(dtype)
        s = sigma.to(dtype)[:, None]
        noisy = bool((s > 0).any())
        logits = []
        for t in range(T):
            h = self.cell(onehot[:, t], h)
            if noisy:
                h = h + s * torch.randn(h.shape, generator=generator, dtype=dtype)
            logits.append(self.readout(h))
        out = torch.stack(logits, dim=1)
        return (out, {}) if diagnostics else out

    def learned_physics(self) -> dict:
        return {}


def gru_params(vocab: int, classes: int, hidden: int) -> int:
    return 3 * hidden * (vocab + hidden) + 6 * hidden + hidden * classes + classes


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def matched_hidden(vocab: int, classes: int, target: int) -> int:
    """GRU hidden size whose parameter count is closest to target (ties: the smaller)."""
    best = min(range(1, 1024), key=lambda h: (abs(gru_params(vocab, classes, h) - target), h))
    return best


def build(kind: str, vocab: int, classes: int, cfg: dict, generator: torch.Generator) -> nn.Module:
    """kind in {pure, soliton, gru, gru_pm}; cfg holds N, d and the physical settings of PHS models.
    gru_pm needs cfg["pm_target"]: the Soliton-PHS parameter count it is matched to."""
    if kind in ("pure", "soliton"):
        return PHSModel(vocab, classes, cfg["N"], cfg["d"], cfg["dt"], cfg["S"], cfg["gamma"],
                        lattice=(kind == "soliton"), u_max=cfg["u_max"], m_pin=cfg.get("m_pin", 2),
                        kappa0=cfg.get("kappa0", 1.0), g_lat0=cfg.get("g_lat0", 1.0),
                        g_pin0=cfg.get("g_pin0", 0.5), j_scale=cfg.get("j_scale", 0.1), generator=generator)
    if kind == "gru":
        return ControlGRU(vocab, classes, 2 * cfg["N"] * cfg["d"], generator=generator)
    if kind == "gru_pm":
        return ControlGRU(vocab, classes, matched_hidden(vocab, classes, cfg["pm_target"]), generator=generator)
    raise ValueError(f"unknown model kind {kind!r}")
