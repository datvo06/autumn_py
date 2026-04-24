"""Matplotlib-based live player for an autumn_py Runtime.

Usage::

    python examples/player.py                  # runs particles
    python examples/player.py particles        # same

Click a grid cell to send a click event to the program. Arrow keys send
left/right/up/down events. `q` quits.

This is deliberately tiny (~80 lines) — a slimmer sibling of Autumn.cpp's
python_test_mpl.py adapted to our flat `render_all()` JSON shape.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

# When invoked as `python examples/player.py`, Python only puts examples/ on
# sys.path — not the project root. Prepend the project root so `examples.*`
# and `autumn_py.*` both resolve.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from autumn_py import Runtime

STEP_INTERVAL_MS = 200  # matplotlib timer interval


def load_program(name: str):
    """Import examples/<name>.py and return its @program-decorated class."""
    mod = importlib.import_module(f"examples.{name}")
    for attr in vars(mod).values():
        if hasattr(attr, "_autumn_spec"):
            return attr
    raise RuntimeError(f"no @program class found in examples.{name}")


class Player:
    def __init__(self, program_cls, grid_size: int = 16, seed: int = 42) -> None:
        self.runtime = Runtime(program_cls, seed=seed)
        self.grid_size = program_cls._autumn_spec.config.get("grid_size", grid_size)

        self.fig, self.ax = plt.subplots(figsize=(8, 8), facecolor="black")
        self.ax.set_aspect("equal")
        self.ax.set_axis_off()
        self.ax.set_facecolor("black")
        self.ax.set_xlim(0, self.grid_size)
        self.ax.set_ylim(0, self.grid_size)

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.timer = self.fig.canvas.new_timer(interval=STEP_INTERVAL_MS)
        self.timer.add_callback(self._tick)
        self.timer.start()

        self._draw()

    def _on_click(self, event) -> None:
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        col = int(event.xdata)
        # flip y so row 0 is at the top, matching Autumn.cpp's screen coords
        row = int(self.grid_size - event.ydata)
        if 0 <= col < self.grid_size and 0 <= row < self.grid_size:
            self.runtime.click(col, row)

    def _on_key(self, event) -> None:
        k = event.key
        if k == "q":
            plt.close(self.fig)
            return
        if k in ("left", "right", "up", "down"):
            getattr(self.runtime, k)()

    def _tick(self) -> None:
        self.runtime.step()
        self._draw()

    def _draw(self) -> None:
        # Remove any existing rectangles without clearing axis limits.
        for p in list(self.ax.patches):
            p.remove()

        cells = self.runtime.render_all()
        for c in cells:
            # Flip y so (0, 0) is at the top-left, matching Autumn's convention.
            x = c["x"]
            y = self.grid_size - 1 - c["y"]
            rect = Rectangle(
                (x, y), 1, 1, facecolor=c["color"], edgecolor="white", linewidth=0.5
            )
            self.ax.add_patch(rect)

        self.fig.canvas.draw_idle()

    def run(self) -> None:
        plt.show()


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "particles"
    Player(load_program(name)).run()


if __name__ == "__main__":
    main()
