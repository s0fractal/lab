"""The homeostatic forward of homeo_arch.py as an importable function: the pinning force normalised by an EMA of
<cos m theta> of each sequence's own state (detached, clamped >= 0.3). bind(model, ema) makes a PHSModel use it."""
import torch

from phs.core import Drive, Physics, internal_force, q_top, wrap

EMA = 0.3


def homeostatic_forward(self, x, sigma, generator=None, diagnostics=False):
    B, T = x.shape
    core = self.core
    phys = core.physics()
    G = core.rotation(phys)
    c1, c2 = core.ou_coefficients(sigma.to(self.E_u.dtype))
    theta = torch.zeros(B, self.N, self.d, dtype=self.E_u.dtype)
    P = torch.zeros_like(theta)
    Dhat = torch.ones(B, dtype=theta.dtype)
    logits, charges, sensor = [], [], []
    for t in range(T):
        D = torch.cos(core.m_pin * theta.detach()).mean((1, 2))
        Dhat = D if t == 0 else (1 - EMA) * Dhat + EMA * D
        g_eff = phys.g_pin / Dhat.clamp_min(0.3)[:, None, None]
        ph = Physics(kappa=phys.kappa, g_lat=phys.g_lat, g_pin=g_eff, m_pin=phys.m_pin, mass=phys.mass, J=phys.J,
                     lattice=True)
        f = internal_force(theta, ph)
        tok = x[:, t]
        drive = Drive(phi=wrap(self.E_phi[tok]), u=self.u_max * torch.sigmoid(self.E_u[tok]))
        theta, P, f = core.token(theta, P, f, ph, G, drive, c1, c2, generator)
        z = torch.cat([torch.cos(theta), torch.sin(theta), P], dim=-1).reshape(B, -1)
        logits.append(self.readout(z))
        if diagnostics:
            charges.append(q_top(theta.detach()))
            sensor.append(Dhat)
    out_ = torch.stack(logits, dim=1)
    if not diagnostics:
        return out_
    return out_, {"q_top": torch.stack(charges, dim=1), "theta_T": theta.detach(), "D": torch.stack(sensor, 1)}


def bind(model, ema=0.3):
    global EMA
    EMA = ema
    model.forward = homeostatic_forward.__get__(model)
    return model
