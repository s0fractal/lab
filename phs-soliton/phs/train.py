"""One training run and its evaluation (GOAL decisions 4-7): AdamW with one clip, batch and step count for every
model, the same data stream per (task, T, seed), checkpoint chosen on a validation set with its own seed, then
every test cell (T, sigma) of the run's T. CPU, float32, deterministic.
"""
from __future__ import annotations

import copy
import math
import resource
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import tasks
from .models import build, count_params

SHARED_KEYS = ("steps", "batch", "lr", "weight_decay", "clip", "eval_every", "n_val")


def setup_determinism(threads: int = 1) -> None:
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(threads)


def peak_rss_bytes() -> int:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r if sys.platform == "darwin" else r * 1024)


def to_torch(b: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.from_numpy(b["x"]), torch.from_numpy(b["targets"]), torch.from_numpy(b["sigma"]).float()


def loss_and_hits(task: str, logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Training loss (task A: last step only, via ignore_index; task B: every step) and per-sequence hits of the
    final prediction (Acc(T) reads the last step in both tasks)."""
    C = logits.shape[-1]
    loss = F.cross_entropy(logits.reshape(-1, C), targets.reshape(-1), ignore_index=-100)
    hits = logits[:, -1].argmax(-1) == targets[:, -1]
    return loss, hits


def erank(theta: torch.Tensor) -> torch.Tensor:
    """Effective rank exp(-sum p ln p), p = s / sum s, of Z = [cos Theta, sin Theta] in R^{N x 2d}, per sequence."""
    Z = torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1).double()
    s = torch.linalg.svdvals(Z)
    p = s / s.sum(-1, keepdim=True)
    return torch.exp(-(p * torch.log(p.clamp_min(1e-300))).sum(-1))


def slips(task: str, q: torch.Tensor) -> torch.Tensor:
    """Phase-slip events |Q_t - Q_{t-1}| >= 1 per sequence, summed over channels, on the holding segment:
    t = 2..T-2 for task A (after the marker, before the query), every t for task B. Q_{-1} = 0 (Theta_0 = 0)."""
    q = torch.cat([torch.zeros_like(q[:, :1]), q.round()], dim=1)   # (B, T + 1, d), index t + 1 is token t
    jumps = (q[:, 1:] - q[:, :-1]).abs() >= 1                       # jumps[:, t]: token t moved the charge
    T = jumps.shape[1]
    seg = slice(2, T - 1) if task == "A" else slice(0, T)
    return jumps[:, seg].sum((1, 2)).double()


@torch.no_grad()
def evaluate(model, task: str, b: dict, seed: int, chunk: int = 1024) -> dict:
    model.eval()
    x, targets, sigma = to_torch(b)
    gen = torch.Generator().manual_seed(seed)
    hits, losses, slip, ranks = [], [], [], []
    for i in range(0, x.shape[0], chunk):
        sl = slice(i, i + chunk)
        logits, diag = model(x[sl], sigma[sl], generator=gen, diagnostics=True)
        loss, hit = loss_and_hits(task, logits, targets[sl])
        hits.append(hit)
        losses.append(float(loss) * hit.shape[0])
        if model.family == "phs":
            slip.append(slips(task, diag["q_top"]))
            ranks.append(erank(diag["theta_T"]))
    n = x.shape[0]
    out = {"acc": float(torch.cat(hits).double().mean()), "loss": sum(losses) / n, "n": n}
    if slip:
        out["r_slip"] = float(torch.cat(slip).mean())
        out["erank"] = float(torch.cat(ranks).mean())
    model.train()
    return out


def flops_per_token(kind: str, cfg: dict, vocab: int, classes: int, hidden: int | None = None) -> int:
    """Analytic multiply-adds x 2 per token per sequence (leading terms; elementwise maps counted once each)."""
    if kind in ("pure", "soliton"):
        N, d, S = cfg["N"], cfg["d"], cfg["S"]
        force = 2 * (2 * N * N * d) + 6 * N * N + 2 * (2 * N * N * d) + 12 * N * d   # C, softmax, A_sym@[c,s]
        if kind == "soliton":
            force += 12 * N * d
        sub = force + 2 * (2 * N * N * d) + 20 * N * d                              # two G rotations, kicks, drifts, O
        return S * sub + 2 * 3 * N * d * classes + 10 * N * d                       # + port and readout
    H = hidden
    return 2 * 3 * H * (vocab + H) + 12 * H + 2 * H * classes


def step_cost_ms(model, task: str, T: int, batch: int = 64, reps: int = 3) -> float:
    """C_step: forward wall time per token for a batch of `batch` sequences, after one warm-up pass, median."""
    b = tasks.batch(task, np.random.default_rng(0), batch, T, sigma=0.0)
    x, _, sigma = to_torch(b)
    times = []
    with torch.no_grad():
        model(x, sigma)
        for _ in range(reps):
            t0 = time.perf_counter()
            model(x, sigma)
            times.append(time.perf_counter() - t0)
    return 1000 * sorted(times)[len(times) // 2] / T


def run(kind: str, task: str, T: int, seed: int, cfg: dict, sigmas=tasks.SIGMA_GRID, n_test: int = 4096,
        test_cells: bool = True, namespace: str = "test", log_every: int = 50, progress=None) -> dict:
    """Train one model and evaluate it. cfg holds the model settings (N, d, dt, S, gamma, u_max, m_pin, ...) and
    the shared optimisation settings SHARED_KEYS. Returns a JSON-able record; a non-finite loss stops the run with
    status FAILED and its reason."""
    t_start = time.perf_counter()
    spec = tasks.TASKS[task]
    init_gen = torch.Generator().manual_seed(tasks.derive_seed("init", kind, task, T, seed))
    noise_gen = torch.Generator().manual_seed(tasks.derive_seed("noise", kind, task, T, seed))
    if kind == "gru_pm" and "pm_target" not in cfg:
        raise ValueError("gru_pm needs cfg['pm_target'], the Soliton-PHS parameter count")
    model = build(kind, spec["vocab"], spec["classes"], cfg, init_gen)
    init_phys = model.learned_physics()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    stream = tasks.train_stream(task, T, seed)
    val = tasks.val_set(task, T, seed, cfg["n_val"])
    record = {"kind": kind, "task": task, "T": T, "seed": seed, "cfg": cfg, "params": count_params(model),
              "state_dim": model.state_dim, "init_physics": init_phys, "curve": [], "val": [], "status": "OK"}
    best = (-1.0, math.inf, -1, None)
    for step in range(1, cfg["steps"] + 1):
        x, targets, sigma = to_torch(tasks.batch(task, stream, cfg["batch"], T))
        logits = model(x, sigma, generator=noise_gen)
        loss, hits = loss_and_hits(task, logits, targets)
        if not torch.isfinite(loss):
            record.update(status="FAILED", reason=f"non-finite training loss at step {step}")
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
        if not torch.isfinite(gnorm):
            record.update(status="FAILED", reason=f"non-finite gradient norm at step {step}")
            break
        opt.step()
        if step % log_every == 0:
            record["curve"].append([step, float(loss), float(hits.double().mean())])
        if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
            v = evaluate(model, task, val, tasks.derive_seed("valnoise", task, T, seed))
            record["val"].append([step, v["acc"], v["loss"]])
            if (v["acc"], -v["loss"]) > (best[0], -best[1]):
                best = (v["acc"], v["loss"], step, copy.deepcopy(model.state_dict()))
            if progress:
                progress(f"{kind} {task} T={T} seed={seed} step {step}: loss {float(loss):.4f} "
                         f"val acc {v['acc']:.4f} ({time.perf_counter() - t_start:.0f}s)")
    record["train_seconds"] = time.perf_counter() - t_start
    if best[3] is None:
        record.setdefault("reason", "no validation checkpoint")
        record["status"] = "FAILED"
        record["peak_rss_bytes"] = peak_rss_bytes()
        return record
    model.load_state_dict(best[3])
    record.update(best_step=best[2], best_val_acc=best[0], best_val_loss=best[1], physics=model.learned_physics())
    if kind == "soliton":
        record["lattice_unused"] = (record["physics"]["g_lat"] < 0.01 * init_phys["g_lat"]
                                    or record["physics"]["g_pin"] < 0.01 * init_phys["g_pin"])
    if test_cells:
        cells = {}
        for s in sigmas:
            b = tasks.test_set(task, T, s, n=n_test, namespace=namespace)
            ev = evaluate(model, task, b, tasks.derive_seed(namespace + "noise", task, T, f"{s:.2f}", seed))
            ev["digest"] = tasks.digest(b)
            cells[f"{s:.2f}"] = ev
        record["cells"] = cells
        record["cell_namespace"] = namespace
        record["c_step_ms"] = step_cost_ms(model, task, T)
        record["flops_per_token"] = flops_per_token(kind, cfg, spec["vocab"], spec["classes"],
                                                    getattr(model, "hidden", None))
    record["wall_seconds"] = time.perf_counter() - t_start
    record["peak_rss_bytes"] = peak_rss_bytes()
    return record
