"""Record a short gif of the sand env for the README banner.

Run with `.venv/bin/python scripts/make_demo_gif.py`.
Outputs `docs/sand_demo.gif`.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from autumn_py import Runtime  # noqa: E402
from examples.sand import SandGame  # noqa: E402

GRID = 10
OUT = ROOT / "docs" / "sand_demo.gif"
OUT.parent.mkdir(exist_ok=True)

r = Runtime(SandGame, seed=42)
frames: list[list[dict]] = [r.render_all()]

# Script: switch to water mode, drop water in two spots, let physics play out.
r.click(7, 0)
r.step(); frames.append(r.render_all())
r.click(3, 0)
r.step(); frames.append(r.render_all())
r.click(6, 0)
r.step(); frames.append(r.render_all())
for _ in range(40):
    r.step()
    frames.append(r.render_all())

fig, ax = plt.subplots(figsize=(4.5, 4.5), facecolor="black")
ax.set_aspect("equal")
ax.set_axis_off()
ax.set_facecolor("black")
ax.set_xlim(0, GRID)
ax.set_ylim(0, GRID)


def _draw(idx: int):
    for p in list(ax.patches):
        p.remove()
    for c in frames[idx]:
        x = c["x"]
        y = GRID - 1 - c["y"]  # flip so 0 is at the top
        ax.add_patch(
            Rectangle(
                (x, y), 1, 1,
                facecolor=c["color"],
                edgecolor="white",
                linewidth=0.3,
            )
        )
    return ax.patches


ani = FuncAnimation(fig, _draw, frames=len(frames), interval=120)
ani.save(str(OUT), writer=PillowWriter(fps=8))
print(f"wrote {OUT} ({len(frames)} frames)")
