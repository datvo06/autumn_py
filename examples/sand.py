"""Port of Autumn.cpp/tests/sand.sexp.

A 10x10 grid with two buttons at the top (red=sand, green=water). A pile
of sand sits at the bottom. Clicking the grid adds sand or water (based on
the currently-selected button). Sand falls under gravity (``nextSolid``);
water falls and flows sideways looking for a hole (``nextLiquid``).
Solid sand adjacent to water liquefies: sandybrown drops become skyblue-
friendly 'liquid' sand.
"""
from __future__ import annotations

from autumn_py import (
    Cell,
    Position,
    StateVar,
    click,
    clicked,
    obj,
    on,
    prev,
    program,
)
from autumn_py.stdlib import (
    addObj,
    adjacentObjs,
    intersects,
    isFreePos,
    nextLiquid,
    nextSolid,
    updateObj,
)

GRID_SIZE = 10


@obj
class Button:
    color: str
    cell = Cell(0, 0, lambda inst: inst.color)


def _sand_color(inst) -> str:
    return "sandybrown" if inst.liquid else "tan"


@obj
class Sand:
    liquid: bool
    cell = Cell(0, 0, _sand_color)


@obj
class Water:
    cell = Cell(0, 0, "skyblue")


_INITIAL_SAND_POSITIONS = [
    *[(x, 9) for x in range(2, 8)],
    *[(x, 8) for x in range(2, 8)],
    *[(x, 7) for x in range(2, 8)],
    *[(x, 6) for x in (2, 4, 5, 7)],
    *[(x, 5) for x in (2, 4, 5, 7)],
]


@program(grid_size=GRID_SIZE)
class SandGame:
    sandButton = StateVar(object)
    waterButton = StateVar(object)
    sand = StateVar(list)
    water = StateVar(list, init=[])
    clickType = StateVar(str, init="sand")

    @sandButton.initializer
    def _():
        return Button("red", Position(2, 0))

    @waterButton.initializer
    def _():
        return Button("green", Position(7, 0))

    @sand.initializer
    def _():
        return [Sand(False, Position(x, y)) for (x, y) in _INITIAL_SAND_POSITIONS]

    # ---- default next-expressions ---------------------------------------

    @sandButton.next
    def _():
        return prev(SandGame.sandButton)

    @waterButton.next
    def _():
        return prev(SandGame.waterButton)

    @clickType.next
    def _():
        return prev(SandGame.clickType)

    @sand.next
    def _():
        return prev(SandGame.sand)

    @water.next
    def _():
        return updateObj(prev(SandGame.water), lambda o: nextLiquid(o))

    # ---- every tick: sand physics ---------------------------------------
    #
    # Upstream sequences two writes to ``sand`` in a single let-body:
    #   1. Apply nextSolid/nextLiquid per grain.
    #   2. Promote non-liquid grains adjacent to prev-water to liquid.
    #
    # In our DSL we stage the intermediate value in a local so the second
    # update reads the first's result (the write-buffer means that
    # ``SandGame.sand.get()`` inside the on-clause still sees the pre-tick
    # value until the buffer flushes at end of on-phase).

    @on(True)
    def _():
        phase1 = updateObj(
            prev(SandGame.sand),
            lambda o: nextLiquid(o) if o.liquid else nextSolid(o),
        )
        water_prev = prev(SandGame.water)
        phase2 = updateObj(
            phase1,
            lambda o: updateObj(o, "liquid", True),
            lambda o: (not o.liquid) and intersects(adjacentObjs(o, 1), water_prev),
        )
        SandGame.sand.set(phase2)

    # ---- click mode switching -------------------------------------------

    @on(lambda: clicked(prev(SandGame.sandButton)))
    def _():
        SandGame.clickType.set("sand")

    @on(lambda: clicked(prev(SandGame.waterButton)))
    def _():
        SandGame.clickType.set("water")

    # ---- grid click: spawn a grain at the click -------------------------

    @on(lambda: (
        clicked()
        and isFreePos(Position(click.x, click.y))
        and prev(SandGame.clickType) == "sand"
    ))
    def _():
        SandGame.sand.set(
            addObj(prev(SandGame.sand), Sand(False, Position(click.x, click.y)))
        )

    @on(lambda: (
        clicked()
        and isFreePos(Position(click.x, click.y))
        and prev(SandGame.clickType) == "water"
    ))
    def _():
        SandGame.water.set(
            addObj(prev(SandGame.water), Water(Position(click.x, click.y)))
        )
