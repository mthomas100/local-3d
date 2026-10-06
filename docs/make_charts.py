#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.8"]
# ///
"""Draw the README's stage charts from bench/runs.tsv (every real m3d stage: wall time, peak memory).

    docs/make_charts.py            # writes docs/media/stage-seconds.png and docs/media/stage-memory.png

Every row is plotted, failures included (hollow orange rings), so nothing is cherry-picked.
run_stage polls every 5 s, so stages shorter than that read as 5 s.
"""
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "bench" / "runs.tsv"
OUT = ROOT / "docs" / "media"

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
OK, FAIL = "#2a78d6", "#eb6834"
SERVER_ENGINES = ("pixal3d", "trellis2", "hunyuan21")


def load():
    rows = list(csv.DictReader(RUNS.open(), delimiter="\t"))
    groups = defaultdict(list)
    for r in rows:
        try:
            groups[f"{r['stage']} · {r['model']}"].append((float(r["seconds"]), float(r["peak_gb"]), r["ok"] == "1"))
        except ValueError:
            continue
    dates = sorted(r["when"][:10] for r in rows)
    return groups, dates[0], dates[-1], len(rows)


def chart(groups, idx, xlabel, title, path, log, span, note=""):
    order = sorted(groups, key=lambda g: statistics.median(v[idx] for v in groups[g]))
    fig, ax = plt.subplots(figsize=(9, 0.28 * len(order) + 1.4), dpi=150)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    for y, g in enumerate(order):
        ok = [v[idx] for v in groups[g] if v[2]]
        bad = [v[idx] for v in groups[g] if not v[2]]
        ax.scatter(ok, [y] * len(ok), s=22, color=OK, alpha=0.55, linewidths=0, zorder=3)
        ax.scatter(bad, [y] * len(bad), s=40, facecolors="none", edgecolors=FAIL, linewidths=1.6, zorder=4)
        med = statistics.median(v[idx] for v in groups[g])
        ax.plot([med, med], [y - 0.32, y + 0.32], color=INK, linewidth=2, zorder=5)
        ax.text(1.005, y, f"n={len(groups[g])}", transform=ax.get_yaxis_transform(), va="center",
                fontsize=7, color=MUTED)
    ax.set_yticks(range(len(order)), order, fontsize=8, color=INK)
    if log:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8); ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=40 if note else 30)
    ax.text(0, 1.012, f"Each dot is one stage run; black tick = median; hollow orange = failed run.{note}\n"
            f"Source: bench/runs.tsv, {span}.", transform=ax.transAxes, fontsize=7.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print(path)


def main():
    groups, first, last, n = load()
    span = f"{n} rows, {first} to {last}, M5 Max 128 GB"
    OUT.mkdir(parents=True, exist_ok=True)
    chart(groups, 0, "wall time per stage (seconds, log scale)", "m3d stage wall time",
          OUT / "stage-seconds.png", True, span)
    # Server-backed engines (ComfyUI, mlx-serve) run in another process, so the client's footprint says
    # nothing about them; a run killed before it reported has peak 0. Both are left out of the memory chart.
    mem = {g: [v for v in vs if v[1] > 0] for g, vs in groups.items()
           if not any(m in g for m in SERVER_ENGINES)}
    mem = {g: vs for g, vs in mem.items() if vs}
    chart(mem, 1, "peak memory footprint (GB, /usr/bin/time -l)", "m3d stage peak memory",
          OUT / "stage-memory.png", False, span,
          note="\nNot shown: Pixal3D, TRELLIS.2 and mlx-serve (another process holds their memory), and runs killed before reporting.")


if __name__ == "__main__":
    main()
