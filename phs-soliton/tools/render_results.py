#!/usr/bin/env python3
"""S6 views: RESULTS.md and figures/*.png generated from results/results.json (never from manifests directly).

    .venv/bin/python tools/render_results.py            # writes RESULTS.md and figures/
    .venv/bin/python tools/render_results.py --check    # RESULTS.md is the rendering of results.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
NAMES = {"pure": "Pure-PHS", "soliton": "Soliton-PHS", "gru": "Control-GRU", "gru_pm": "Control-GRU-pm"}
# categorical slots 1-4 of the reference palette (fixed order, never cycled), with a marker as secondary encoding
COLOR = {"pure": "#2a78d6", "soliton": "#eb6834", "gru": "#1baf7a", "gru_pm": "#eda100"}
MARKER = {"pure": "o", "soliton": "s", "gru": "^", "gru_pm": "v"}
# ordinal blue ramp (T in the R_slip scatter), light -> dark
SIGMA_RAMP = ["#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#0d366b"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:+.{nd}f}" if isinstance(x, float) and x < 0 else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def render(r: dict, prereg: dict) -> str:
    g = prereg["grid"]
    o = ["# Results", "",
         f"> Generated from `results/results.json` by `tools/render_results.py`. PREREG `{prereg['tag']}` "
         f"(prereg.json SHA-256 `{r['prereg_sha256'][:16]}…`), `analyze.py` SHA-256 `{r['analyze_sha256'][:16]}…`.", "",
         f"## Verdict: `{r['verdict']}`", "", r["verdict_reason"] + ".", "",
         f"Runs: {r['runs_present']} of {r['runs_expected']} with a manifest; FAILED: {len(r['failed'])}; "
         f"missing: {len(r['missing'])}.", ""]
    for kind, rows in (("FAILED", r["failed"]), ("missing", r["missing"])):
        if rows:
            o += [f"{kind}: " + ", ".join("/".join(map(str, x)) for x in rows), ""]
    o += ["## Hypotheses", "", "| flag | metric | Soliton vs | delta (mean) | min effect | p (exact, one-sided) | "
          "Holm | holds |", "|---|---|---|---|---|---|---|---|"]
    for name, h in r["hypotheses"].items():
        if h.get("available"):
            o.append(f"| {name} | {h['metric']} | {NAMES[h['versus']]} | {fmt(h['delta'], 4)} | {h['min_effect']} | "
                     f"{h['p']:.4f} | {h['holm_reject']} | **{h['holds']}** |")
        else:
            o.append(f"| {name} | {h['metric']} | {NAMES[h['versus']]} | - | - | - | - | **False** (no data) |")
    B, V, room = r["B"], r["V"], r["room"]
    o += ["", f"- **B** (Soliton not worse than Pure on task B, mean Acc over T at sigma 0, margin {B['margin']}): "
          + (f"Soliton {B['soliton']:.4f}, Pure {B['pure']:.4f}, delta {B['delta']:+.4f} → **{B['holds']}**"
             if B["available"] else "no data → **False**"),
          f"- **V** (gate and Acc ≥ 0.95 on A at T = 16, sigma 0 in ≥ 4 of 5 seeds): gate {V['gate_ok']}; "
          + "; ".join(f"{NAMES[k]} {v['seeds_ok']}/5" for k, v in V["models"].items()) + f" → **{V['holds']}**",
          f"- **room** (REFUTED needs Pure-PHS mean sigma_crit ≤ {room.get('sigma_max')} and mean lambda ≤ "
          f"{room.get('lambda_max')}): "
          + (f"Pure sigma_crit {room['pure_sigma_crit_mean']:.4f}, lambda {room['pure_lambda']:+.4f} → "
             f"**{room['holds']}**" if room["available"] else "no data"), ""]
    o += ["## Per seed", "", "| model | seed | mean sigma_crit (A) | sigma_crit at T = " + ", ".join(map(str, g["T"]))
          + " | lambda (A, sigma 0.10) | Acc A T=16 sigma 0 | mean Acc B sigma 0 |", "|---|---|---|---|---|---|---|"]
    for k, seeds in r["per_seed"].items():
        for s, row in seeds.items():
            sc = row.get("sigma_crit_T")
            o.append(f"| {NAMES[k]} | {s} | {fmt(row.get('sigma_crit_mean'), 4)} | "
                     f"{', '.join(fmt(sc[str(T)], 2) for T in g['T']) if sc else '-'} | {fmt(row.get('lambda'), 4)} | "
                     f"{fmt(row.get('acc_A_T16_sigma0'), 4)} | {fmt(row.get('B_mean_acc_sigma0'), 4)} |")
    o += [""]
    for task in ("A", "B"):
        o += [f"## Accuracy by cell, task {task} (mean over seeds)", "",
              "| model | T | " + " | ".join(f"{s:.2f}" for s in g["sigma"]) + " |", "|---|---|" + "---|" * len(g["sigma"])]
        for k in g["models"]:
            for T in g["T"]:
                c = r["cells"].get(k, {}).get(task, {}).get(str(T), {})
                o.append(f"| {NAMES[k]} | {T} | " + " | ".join(fmt(c.get(f'{s:.2f}', {}).get('acc_mean'), 3)
                                                            for s in g["sigma"]) + " |")
        o += [""]
    o += ["## Phase slips and effective rank, task A (PHS models, mean over seeds)", "",
          "| model | T | R_slip by sigma | erank(Z_T) by sigma |", "|---|---|---|---|"]
    for k in ("pure", "soliton"):
        for T in g["T"]:
            c = r["cells"].get(k, {}).get("A", {}).get(str(T), {})
            o.append(f"| {NAMES[k]} | {T} | " + ", ".join(fmt(c.get(f'{s:.2f}', {}).get('r_slip_mean'), 2)
                                                        for s in g["sigma"]) + " | "
                     + ", ".join(fmt(c.get(f'{s:.2f}', {}).get('erank_mean'), 2) for s in g["sigma"]) + " |")
    sec = r["secondary"]
    o += ["", "## Secondary measures (not in the verdict)", "",
          f"- Spearman rho(R_slip, Acc), Soliton-PHS, task A cells (mean over seeds): "
          f"{fmt(sec.get('spearman_rslip_acc_soliton_A'), 3)} (prediction of the starter: < 0)."]
    for key in ("sigma_crit_mean", "lambda"):
        v = sec.get(f"soliton_vs_gru_pm_{key}")
        if v:
            o.append(f"- Soliton-PHS vs Control-GRU-pm, {key}: delta {v['delta']:+.4f}, exact one-sided p {v['p']:.4f}.")
    o += ["", "| model | params (task, n) | C_step ms/token (median) | FLOPs/token (task, n) | learned physics (medians) |",
          "|---|---|---|---|---|"]
    for k, info in sec["models"].items():
        phys = ", ".join(f"{key[:-7]} {info[key]:.3g}" for key in info if key.endswith("_median") and key != "c_step_ms_median")
        if "gamma" in info:
            phys += f", gamma {info['gamma']}"
        if "lattice_unused_rate" in info:
            phys += f", LATTICE_UNUSED rate {info['lattice_unused_rate']:.2f}"
        o.append(f"| {NAMES[k]} | {info['params']} | {fmt(info['c_step_ms_median'], 3)} | {info['flops_per_token']} | "
                 f"{phys or '-'} |")
    o += ["", "## Figures", "", "![Acc(T) by sigma, task A](figures/acc_T_sigma_A.png)", "",
          "![Acc(T) by sigma, task B](figures/acc_T_sigma_B.png)", "", "![sigma_crit by T](figures/sigma_crit_T.png)", "",
          "![R_slip vs Acc, Soliton-PHS, task A](figures/rslip_vs_acc.png)", ""]
    return "\n".join(o)


def style(ax, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)


def figures(r: dict, prereg: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = prereg["grid"]
    FIG.mkdir(exist_ok=True)
    Ts = g["T"]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("acc", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
    for task in ("A", "B"):
        fig, axes = plt.subplots(len(g["models"]), 1, figsize=(0.42 * len(g["sigma"]) + 2.2, 1.25 * len(g["models"]) + 1.2),
                                 facecolor=SURFACE, sharex=True)
        for ax, k in zip(axes, g["models"]):
            grid_acc = [[r["cells"].get(k, {}).get(task, {}).get(str(T), {}).get(f"{s:.2f}", {}).get("acc_mean",
                                                                                                float("nan"))
                         for s in g["sigma"]] for T in Ts]
            im = ax.imshow(grid_acc, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
            for yi, row in enumerate(grid_acc):
                for xi, v in enumerate(row):
                    if v == v:
                        ax.text(xi, yi, f"{v:.2f}"[1:] if v < 1 else "1", ha="center", va="center", fontsize=6,
                                color=INK if v < 0.55 else SURFACE)
            ax.set_yticks(range(len(Ts)), [str(T) for T in Ts])
            ax.set_ylabel(f"{NAMES[k]}\nT", color=MUTED, fontsize=8)
            ax.tick_params(colors=MUTED, labelsize=7)
            for side in ax.spines.values():
                side.set_visible(False)
            if "train_sigma" in g:
                edge = len([s for s in g["sigma"] if s <= max(g["train_sigma"]) + 1e-9]) - 0.5
                ax.axvline(edge, color=MUTED, linewidth=1, linestyle=(0, (3, 2)))
        axes[-1].set_xticks(range(len(g["sigma"])), [f"{s:.2f}" for s in g["sigma"]], rotation=90)
        axes[-1].set_xlabel("σ (dashed line: top of the training grid)", color=MUTED, fontsize=8)
        fig.colorbar(im, ax=axes, shrink=0.6, label="Acc, mean over seeds")
        fig.suptitle(f"Task {task}: accuracy at the last step by (T, σ)", color=INK, fontsize=11)
        fig.savefig(FIG / f"acc_T_sigma_{task}.png", dpi=130, facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.2, 3.4), facecolor=SURFACE)
    for k in g["models"]:
        rows = r["per_seed"].get(k, {})
        means = []
        for T in Ts:
            vals = [row["sigma_crit_T"][str(T)] for row in rows.values() if row.get("sigma_crit_T")]
            means.append(sum(vals) / len(vals) if vals else None)
            ax.scatter([T] * len(vals), vals, color=COLOR[k], marker=MARKER[k], s=10, alpha=0.35, linewidths=0)
        pts = [(T, m) for T, m in zip(Ts, means) if m is not None]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=COLOR[k], marker=MARKER[k], markersize=6,
                    linewidth=2, label=NAMES[k])
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ts, [str(T) for T in Ts])
    style(ax, "T", "σ_crit (task A)")
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=8, frameon=False, labelcolor=MUTED)
    ax.set_title("σ_crit by T: mean (line) and seeds (points)", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "sigma_crit_T.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.2, 3.4), facecolor=SURFACE)
    for j, T in enumerate(Ts):
        cells = r["cells"].get("soliton", {}).get("A", {}).get(str(T), {})
        xs = [c["r_slip_mean"] for c in cells.values() if "r_slip_mean" in c]
        ys = [c["acc_mean"] for c in cells.values() if "r_slip_mean" in c]
        ax.scatter(xs, ys, color=SIGMA_RAMP[2 * j + 1], s=28, marker="o", edgecolors=SURFACE, linewidths=1,
                   label=f"T {T}")
    style(ax, "R_slip (slips per sequence, holding segment)", "Acc")
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=8, frameon=False, labelcolor=MUTED)
    rho = r["secondary"].get("spearman_rslip_acc_soliton_A")
    ax.set_title(f"Soliton-PHS, task A cells: Spearman ρ = {fmt(rho, 3)}", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "rslip_vs_acc.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)


def main(argv: list[str]) -> int:
    r = json.loads((ROOT / "results" / "results.json").read_text())
    prereg = json.loads((ROOT / "prereg.json").read_text())
    text = render(r, prereg)
    if "--check" in argv:
        ok = (ROOT / "RESULTS.md").exists() and (ROOT / "RESULTS.md").read_text() == text
        print("RESULTS.md: ok" if ok else "FAIL RESULTS.md is not the rendering of results/results.json")
        return 0 if ok else 1
    (ROOT / "RESULTS.md").write_text(text)
    figures(r, prereg)
    print("RESULTS.md and figures/ written")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
