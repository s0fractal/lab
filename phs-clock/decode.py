"""Honest decoding of one trained checkpoint (PREREG decisions D3 and D4): the frozen fringes decoders, unchanged.

Pure-PHS and Soliton-PHS: fringes/readout.py runs as it is on the checkpoint. It draws its calibration cells (namespace
"calib-readout", sigma 0, 0.05, ..., 1.2, 2048 sequences each), fits the pooled LDA and the sensor decoder (16
quantile bins of D = <cos m theta>, one LDA per bin) and writes them (--decoder) with its own scores on its held-out
cells ("play", the same sigmas, scored on one half). transfer.py's own states() and score() then score those
decoders. First on readout.py's play cells, where they must give readout.py's numbers exactly (the decoders scored
are the ones readout.py fit), then on the PREREG test cells (namespace "test", every sigma of the grid, scored on the
second half).

Control-GRU has no phases, so no sensor. Its pooled LDA is readout.py's lda_fit on the same calibration cells, drawn
and seeded as readout.py draws them; lda_predict scores it on test cells drawn and seeded as transfer.py's states().

The fringes files run from bytes checked against their SHA-256 in prereg.json; phs/ must already be imported from
verified bytes (run.py)."""
from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

CAL_SIGMAS = [round(0.05 * i, 2) for i in range(25)]   # readout.py's SIG
CAL_N = 2048                                          # readout.py's default n
SCORED = ("fixed", "pooled", "sensor", "sigma_lda", "oracle", "D")


def frozen_bytes(lab: Path, rel: str, frozen: dict) -> bytes:
    data = (lab / rel).read_bytes()
    if hashlib.sha256(data).hexdigest() != frozen[rel]:
        raise RuntimeError(f"{rel} differs from the SHA-256 pinned in prereg.json")
    return data


def functions(data: bytes, filename: str, names: tuple, env: dict) -> dict:
    """Execute only the named top-level functions of a frozen file, with env as their globals."""
    tree = ast.parse(data, filename)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if sorted(n.name for n in nodes) != sorted(names):
        raise RuntimeError(f"{filename} does not define each of {names} once")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec"), env)
    return env


def load_model(ckpt: Path):
    from phs import tasks
    from phs.models import build
    c = torch.load(ckpt)
    spec = tasks.TASKS["A"]
    model = build(c["kind"], spec["vocab"], spec["classes"], c["cfg"], torch.Generator().manual_seed(0))
    model.load_state_dict(c["state"])
    model.eval()
    grab = {}
    model.readout.register_forward_hook(lambda mod, inp, o: grab.__setitem__("z", inp[0].detach()))
    return c, model, grab, spec["classes"]


def phs(lab: Path, frozen: dict, ckpt: Path, work: Path, grid: list, n_test: int) -> dict:
    from phs import tasks
    from phs.train import to_torch
    readout_src = frozen_bytes(lab, "fringes/readout.py", frozen)
    transfer_src = frozen_bytes(lab, "fringes/transfer.py", frozen)
    work.mkdir(parents=True, exist_ok=True)
    rjson, npz = work / f"{ckpt.stem}.readout.json", work / f"{ckpt.stem}.decoder.npz"
    argv = sys.argv
    sys.argv = ["readout.py", str(lab), str(ckpt), str(rjson), str(CAL_N), "1", "--decoder", str(npz)]
    try:
        exec(compile(readout_src, "fringes/readout.py", "exec"), {"__name__": "__main__"})
    finally:
        sys.argv = argv
    ref = json.loads(rjson.read_text())
    c, model, grab, K = load_model(ckpt)
    env = functions(transfer_src, "fringes/transfer.py", ("states", "lda_fit", "lda_predict", "score"),
                    {"np": np, "torch": torch, "tasks": tasks, "to_torch": to_torch, "model": model, "grab": grab,
                     "N": model.N, "d": model.d, "mp": model.core.m_pin, "K": K, "SPREAD": False})
    a = np.load(npz, allow_pickle=False)

    def m(name):
        return a[f"{name}_W"], a[f"{name}_b"]
    # as transfer.py's frozen_decoders(): the decoder file and readout.py's JSON come from one run, so exactly equal
    if not (a["sigmas"].tolist() == CAL_SIGMAS == [r["sigma"] for r in ref["rows"]] and a["edges"].shape == (15,)
            and a["edges"].tolist() == ref["edges"] and a["D_cal"].tolist() == [r["D_cal"] for r in ref["rows"]]):
        raise RuntimeError("the decoder file is not the one readout.py fit with its JSON")
    dec = {"pooled": m("pooled"), "edges": a["edges"], "bins": [m(f"bin{k:02d}") for k in range(16)],
           "cells": {s: m(f"cell{i:02d}") for i, s in enumerate(CAL_SIGMAS)}}
    T = c["T"]
    play_diff = 0.0
    for r in ref["rows"]:
        got = env["score"](dec, env["states"](r["sigma"], "play", T, CAL_N), r["sigma"])
        play_diff = max(play_diff, max(abs(got[k] - r[k]) for k in SCORED))
    if play_diff != 0:
        raise RuntimeError(f"transfer.py's score() on readout.py's play cells is off by {play_diff}")
    rows = []
    for s in grid:
        cell = env["states"](s, "test", T, n_test)
        rows.append({"sigma": s, **env["score"](dec, cell, s), "nonfinite_z": int((~np.isfinite(cell[0])).any(1).sum())})
    return {"decoders": ["fixed", "pooled", "sensor", "oracle"], "rows": rows, "play": ref["rows"],
            "edges": ref["edges"], "play_check": "exact", "decoder_sha256": hashlib.sha256(npz.read_bytes()).hexdigest()}


def gru(lab: Path, frozen: dict, ckpt: Path, grid: list, n_test: int) -> dict:
    from phs import tasks
    from phs.train import to_torch
    readout_src = frozen_bytes(lab, "fringes/readout.py", frozen)
    c, model, grab, K = load_model(ckpt)
    env = functions(readout_src, "fringes/readout.py", ("lda_fit", "lda_predict"), {"np": np, "K": K})
    T = c["T"]

    @torch.no_grad()
    def states(sigma, namespace, n, fmt):
        b = tasks.test_set("A", T, sigma, n=n, namespace=namespace)
        x, tg, sg = to_torch(b)
        seed = tasks.derive_seed(f"{namespace}-noise", T, format(sigma, fmt))
        logits = model(x, sg, generator=torch.Generator().manual_seed(seed))
        return grab["z"].double().numpy(), logits[:, -1].double().numpy().argmax(1), tg[:, -1].numpy()

    cal = [states(s, "calib-readout", CAL_N, ".2f") for s in CAL_SIGMAS]   # readout.py seeds with two decimals
    pooled = env["lda_fit"](np.concatenate([z for z, _, _ in cal]), np.concatenate([y for _, _, y in cal]))
    rows = []
    for s in grid:
        z, fixed, y = states(s, "test", n_test, ".3f")                     # transfer.py seeds "test" with three
        half = len(y) // 2
        oracle = env["lda_fit"](z[:half], y[:half])
        rows.append({"sigma": s, "fixed": float((fixed[half:] == y[half:]).mean()),
                     "pooled": float((env["lda_predict"](pooled, z[half:]) == y[half:]).mean()),
                     "oracle": float((env["lda_predict"](oracle, z[half:]) == y[half:]).mean()),
                     "nonfinite_z": int((~np.isfinite(z)).any(1).sum())})
    return {"decoders": ["fixed", "pooled", "oracle"], "rows": rows}
