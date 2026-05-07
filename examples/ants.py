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


def _total_dist_to_food(ant_list, food_list) -> int:
    """Sum, over every ant, of the squared distance from that ant to its
    nearest food. Used by the trajectory_invariant on AntsGame.ants — the
    property ``ants close in on food`` is exactly: this total does not
    increase between ticks (when food exists at the start of the tick)."""
    if not food_list:
        return 0
    return sum(
        min(sqdist(a.origin, f.origin) for f in food_list) for a in ant_list
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
        # After each tick, every ant's distance to its nearest food
        # (measured against the *pre-tick* food set, so a just-eaten
        # food still counts as distance 0) must not increase. Vacuous
        # when there's no food.
        #
        # ``ants(t)`` / ``foods(t)`` are per-tick lookup functions: the
        # gate's TrajectoryInvariantGoal walks t over the recorded
        # snapshots and binds these by parameter name.
        trajectory_invariant=lambda ants, foods, t:
            (len(foods(t)) == 0) or (
                _total_dist_to_food(ants(t + 1), foods(t))
                <= _total_dist_to_food(ants(t), foods(t))
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
