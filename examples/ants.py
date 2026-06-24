"""Port of Autumn.cpp/tests/ants.sexp.

Ants walk toward the closest food. Clicking spawns a pair of food pellets
at random grid positions. Food that an ant touches is consumed."""
from __future__ import annotations

from autumn_py import (
    Cell,
    Position,
    StateVar,
    clicked,
    obj,
    on,
    prev,
    program,
    spec,
)
from autumn_py.stdlib import (
    closest,
    concat,
    intersects,
    move,
    randomPositions,
    sqdist,
    unitVector,
    updateObj,
)


@obj
class Ant:
    cell = Cell(0, 0, "gray")


@obj
class Food:
    cell = Cell(0, 0, "red")


@program(grid_size=16)
class AntsGame:
    ants = StateVar(list)
    foods = StateVar(list, init=[])

    @ants.initializer
    def _():
        return [Ant(Position(5, 5)), Ant(Position(1, 14))]

    @ants.next
    @spec(
        # After each tick, every ant's nearest-food distance — measured
        # against the union of pre-tick and post-tick food (so newly
        # spawned pellets count as valid targets, and just-eaten pellets
        # still count as distance 0) — must not exceed its pre-tick
        # nearest-food distance. Per-ant ``min``-of-``min`` rather than a
        # summed total: catches an ant that ignores a freshly spawned
        # adjacent pellet, while staying sound under food removal
        # (eating a close pellet doesn't falsely fail the spec because
        # the eaten pellet is still in ``foods(t)``).
        #
        # Vacuous when no food exists at the start of the tick.
        # ``ants(t)`` / ``foods(t)`` are per-tick lookup functions —
        # the gate's TrajectoryInvariantGoal walks t over recorded
        # snapshots and binds these by parameter name.
        trajectory_invariant=lambda ants, foods, t:
            (len(foods(t)) == 0) or all(
                min(sqdist(a_next.origin, f.origin)
                    for f in (foods(t) + foods(t + 1)))
                <= min(sqdist(a.origin, f.origin) for f in foods(t))
                for a, a_next in zip(ants(t), ants(t + 1))
            ),
        trajectory_steps=8,
    )
    def _():
        return prev(AntsGame.ants)

    @foods.next
    def _():
        return prev(AntsGame.foods)

    # Every tick: remove any food an ant is sitting on.
    @on(True)
    def _():
        ants = prev(AntsGame.ants)
        AntsGame.foods.set(
            [f for f in prev(AntsGame.foods) if not intersects(f, ants)]
        )

    # Every tick: each ant steps one grid unit toward the closest food.
    # ``foods`` here reads prev: the filter clause above buffers its write
    # until after the on-phase, so the ant targets prev-food (matching the
    # C++ semantics where all clauses fire against the pre-tick snapshot).
    @on(True)
    def _():
        foods = prev(AntsGame.foods)

        def step(a):
            if not foods:
                return a
            target = closest(a, foods)
            return move(a, unitVector(a, target))

        AntsGame.ants.set(updateObj(prev(AntsGame.ants), step))

    # Click: spawn two food pellets at random grid positions.
    @on(clicked)
    def _():
        new_foods = [Food(p) for p in randomPositions(16, 2)]
        AntsGame.foods.set(concat([prev(AntsGame.foods), new_foods]))
