from __future__ import annotations

from autumn_py import Runtime
from examples.particles import Particles


def _cardinal_step(a: dict, b: dict) -> bool:
    """Return True iff two cells differ by exactly one cardinal grid step."""
    return abs(a["x"] - b["x"]) + abs(a["y"] - b["y"]) == 1


def test_particles_runs_without_events():
    with Runtime(Particles, seed=42) as r:
        r.step()
        out = r.render_all()
    assert out == []


def test_particles_spawns_on_click():
    with Runtime(Particles, seed=42) as r:
        r.click(3, 4)
        r.step()
        cells = r.render_all()
    assert cells == [{"x": 3, "y": 4, "color": "blue"}]


def test_particle_trajectory_is_a_walk_of_unit_cardinal_steps():
    """After a single click spawns one particle, every subsequent step moves
    it by exactly one grid unit in a cardinal direction (adjPositions neighbours
    are the only candidates the next-expression can produce)."""
    with Runtime(Particles, seed=42) as r:
        r.click(3, 4)
        r.step()
        trajectory = [r.render_all()]
        for _ in range(20):
            r.step()
            trajectory.append(r.render_all())

    for frame in trajectory:
        assert len(frame) == 1, frame
        assert frame[0]["color"] == "blue"

    for prev_frame, next_frame in zip(trajectory, trajectory[1:]):
        assert _cardinal_step(prev_frame[0], next_frame[0]), (
            f"non-cardinal transition {prev_frame} -> {next_frame}"
        )


def test_particles_are_deterministic_under_same_seed():
    """Two Runtimes with the same seed, driven by the same event sequence,
    produce identical render outputs at every step."""

    def run(seed: int) -> list[list[dict]]:
        with Runtime(Particles, seed=seed) as r:
            r.click(3, 4)
            r.step()
            out = [r.render_all()]
            for _ in range(10):
                r.step()
                out.append(r.render_all())
        return out

    assert run(42) == run(42)
    assert run(0) != run(42)  # different seed → different trajectory


def test_particles_multiple_clicks_accumulate():
    with Runtime(Particles, seed=42) as r:
        r.click(3, 4)
        r.step()
        r.click(7, 8)
        r.step()
        cells = r.render_all()
    # Second click rebuilds particles = addObj(prev, new), where prev is the
    # list after step 1 (one wandered particle). Two particles total.
    assert len(cells) == 2
    positions = sorted((c["x"], c["y"]) for c in cells)
    assert (7, 8) in positions
