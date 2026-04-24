"""Port of Autumn.cpp/tests/grow.sexp.

A sun patrols the top-left; a cloud sits on the top-right. Arrow keys
left/right move the cloud. Down-arrow drops water below the cloud; water
falls each tick via gravity. When water would hit a green leaf, it stops;
when water hits a leaf AND the sun is not occluded by the cloud, the leaf
grows upward (new leaves spawn one cell up from the watered leaf, turning
mediumpurple at y=12). Clicking the sun steps it one cell left or right
along its patrol direction.
"""
from __future__ import annotations

from autumn_py import (
    Cell,
    Position,
    StateVar,
    clicked,
    down,
    left,
    obj,
    on,
    prev,
    program,
    right,
)
from autumn_py.stdlib import (
    addObj,
    intersects,
    isWithinBounds,
    moveDown,
    moveLeft,
    moveRight,
    moveUp,
    updateObj,
)


@obj
class Water:
    cell = Cell(0, 0, "blue")


@obj
class Leaf:
    color: str
    cell = Cell(0, 0, lambda inst: inst.color)


_CLOUD_CELLS = [
    Cell(dx, dy, "gray")
    for dy in range(3)
    for dx in (-1, 0, 1, 2)
]


@obj
class Cloud:
    cells = _CLOUD_CELLS


_SUN_CELLS = [Cell(dx, dy, "gold") for dy in range(3) for dx in range(3)]


@obj
class Sun:
    movingLeft: bool
    cells = _SUN_CELLS


@program(grid_size=16)
class GrowGame:
    sun = StateVar(object)
    water = StateVar(list, init=[])
    cloud = StateVar(object)
    leaves = StateVar(list)

    # ---- initializers ----------------------------------------------------

    @sun.initializer
    def _():
        return Sun(False, Position(0, 0))

    @cloud.initializer
    def _():
        return Cloud(Position(13, 0))

    @leaves.initializer
    def _():
        return [Leaf("green", Position(x, 15)) for x in (1, 3, 5, 7, 9, 11, 13, 15)]

    # ---- default next-expressions ----------------------------------------

    @sun.next
    def _():
        return prev(GrowGame.sun)

    @water.next
    def _():
        # Gravity: each drop falls one cell; drops that leave the grid vanish.
        return [w for w in (moveDown(d) for d in prev(GrowGame.water)) if isWithinBounds(w)]

    @cloud.next
    def _():
        return prev(GrowGame.cloud)

    @leaves.next
    def _():
        return prev(GrowGame.leaves)

    # ---- input: down-arrow drops a water pellet below the cloud ---------

    @on(down)
    def _():
        drop_origin = moveDown(prev(GrowGame.cloud)).origin
        GrowGame.water.set(addObj(prev(GrowGame.water), Water(drop_origin)))

    # ---- water-on-leaf interactions -------------------------------------

    @on(lambda: intersects(
        [moveDown(w) for w in prev(GrowGame.water)],
        prev(GrowGame.leaves),
    ))
    def _():
        # Water about to land on a leaf is consumed rather than continuing
        # to fall. Note this reads from the *live* water list, which is the
        # post-filter result intended to survive into the next tick.
        leaves_prev = prev(GrowGame.leaves)
        GrowGame.water.set([
            w for w in GrowGame.water.get()
            if not intersects(moveDown(w), leaves_prev)
        ])

    # ---- growth: new leaves spawn one cell above watered green leaves ---

    @on(lambda: (
        intersects(
            [moveDown(w) for w in prev(GrowGame.water)],
            [l for l in prev(GrowGame.leaves) if l.color == "green"],
        )
        and not intersects(prev(GrowGame.sun), prev(GrowGame.cloud))
    ))
    def _():
        watered = [
            lf for lf in prev(GrowGame.leaves)
            if intersects(moveUp(lf), prev(GrowGame.water))
        ]
        new_leaves = [
            Leaf(
                "mediumpurple" if moveUp(lf).origin.y == 12 else "green",
                moveUp(lf).origin,
            )
            for lf in watered
        ]
        GrowGame.leaves.set(addObj(prev(GrowGame.leaves), new_leaves))

    # ---- cloud controls --------------------------------------------------

    @on(left)
    def _():
        GrowGame.cloud.set(moveLeft(prev(GrowGame.cloud)))

    @on(right)
    def _():
        GrowGame.cloud.set(moveRight(prev(GrowGame.cloud)))

    # ---- sun patrol -----------------------------------------------------

    @on(lambda: prev(GrowGame.sun).origin.x == 0)
    def _():
        GrowGame.sun.set(updateObj(prev(GrowGame.sun), "movingLeft", False))

    @on(lambda: prev(GrowGame.sun).origin.x == 5)
    def _():
        GrowGame.sun.set(updateObj(prev(GrowGame.sun), "movingLeft", True))

    @on(lambda: clicked(prev(GrowGame.sun)))
    def _():
        s = prev(GrowGame.sun)
        GrowGame.sun.set(moveLeft(s) if s.movingLeft else moveRight(s))
