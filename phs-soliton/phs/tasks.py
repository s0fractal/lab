"""Task generators (GOAL decision 5). Every stream is a numpy PCG64 generator seeded from a named path, so the
same (task, split, T, seed) gives the same sequences to every model, and a test set is pinned by its SHA-256.

Task A, selective copy-recall: input tokens 0 (blank), 1..7 (symbols), 8 (query). The marker y is shown at t = 0, 1;
t = 2..T-2 carry blanks with symbol distractors at p_dist = 0.25 (same alphabet); t = T-1 is the query. Label y - 1
in 0..6, loss on the last step only, chance 1/7.

Task B, modular automaton (q = 5): commands 0..3 uniform, s_{-1} = 0, s_t = (s_{t-1} + x_t) mod q for x_t < 3 and
s_t = (q - s_{t-1}) mod q for x_t = 3. Targets s_t at every step; Acc(T) reads the final state.
"""
from __future__ import annotations

import hashlib

import numpy as np

SIGMA_GRID = tuple(round(0.05 * i, 2) for i in range(8))   # 0.00 .. 0.35
T_GRID = (16, 32, 64, 128)
P_DIST = 0.25
SYMBOLS = 7
Q_MOD = 5

TASKS = {
    "A": {"vocab": SYMBOLS + 2, "classes": SYMBOLS, "dense": False},
    "B": {"vocab": 4, "classes": Q_MOD, "dense": True},
}


def derive_seed(*parts) -> int:
    """A 64-bit seed from a named path, e.g. derive_seed("test", "A", 16, "0.10")."""
    return int.from_bytes(hashlib.sha256("/".join(str(p) for p in parts).encode()).digest()[:8], "little")


def task_a(rng: np.random.Generator, n: int, T: int) -> tuple[np.ndarray, np.ndarray]:
    y = rng.integers(1, SYMBOLS + 1, size=n)
    x = np.zeros((n, T), dtype=np.int64)
    x[:, 0] = y
    x[:, 1] = y
    hold = T - 3
    dist = rng.random((n, hold)) < P_DIST
    sym = rng.integers(1, SYMBOLS + 1, size=(n, hold))
    x[:, 2:T - 1] = np.where(dist, sym, 0)
    x[:, T - 1] = SYMBOLS + 1
    targets = np.full((n, T), -100, dtype=np.int64)   # ignored by cross-entropy except at the last step
    targets[:, T - 1] = y - 1
    return x, targets


def task_b(rng: np.random.Generator, n: int, T: int) -> tuple[np.ndarray, np.ndarray]:
    x = rng.integers(0, 4, size=(n, T))
    targets = np.empty((n, T), dtype=np.int64)
    s = np.zeros(n, dtype=np.int64)
    for t in range(T):
        c = x[:, t]
        s = np.where(c == 3, (Q_MOD - s) % Q_MOD, (s + c) % Q_MOD)
        targets[:, t] = s
    return x.astype(np.int64), targets


GENERATORS = {"A": task_a, "B": task_b}


def batch(task: str, rng: np.random.Generator, n: int, T: int, sigma: float | None = None) -> dict:
    """n sequences; sigma=None draws one sigma per sequence uniformly from SIGMA_GRID (training and validation)."""
    x, targets = GENERATORS[task](rng, n, T)
    if sigma is None:
        sig = np.asarray(SIGMA_GRID)[rng.integers(0, len(SIGMA_GRID), size=n)]
    else:
        sig = np.full(n, float(sigma))
    return {"x": x, "targets": targets, "sigma": sig.astype(np.float64)}


def digest(b: dict) -> str:
    h = hashlib.sha256()
    for key in ("x", "targets", "sigma"):
        a = np.ascontiguousarray(b[key])
        h.update(key.encode())
        h.update(str(a.shape).encode())
        h.update(a.astype("<i8" if a.dtype.kind == "i" else "<f8").tobytes())
    return h.hexdigest()


def test_set(task: str, T: int, sigma: float, n: int = 4096, namespace: str = "test") -> dict:
    """The fixed test cell (task, T, sigma): its own seed, shared by every model and seed. Calibration reads
    cells of the same shape from the namespace "calib", never the test cells."""
    rng = np.random.default_rng(derive_seed(namespace, task, T, f"{sigma:.2f}"))
    return batch(task, rng, n, T, sigma=sigma)


def train_stream(task: str, T: int, seed: int) -> np.random.Generator:
    return np.random.default_rng(derive_seed("train", task, T, seed))


def val_set(task: str, T: int, seed: int, n: int) -> dict:
    return batch(task, np.random.default_rng(derive_seed("val", task, T, seed)), n, T)
