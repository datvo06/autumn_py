"""Port of Autumn.cpp/tests/gameOfLife.sexp.

A 16x16 Conway's Game of Life. Each Particle cell tracks a ``living`` field
(rendered pink if alive, black if dead). Click a particle to toggle it;
click the green button (top-left) to advance one Life generation; click
the silver button (top-right) to reset (kill all)."""
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
from autumn_py.stdlib import allPositions, updateObj

GRID_SIZE = 16


def _color_of_living(inst) -> str:
    return "lightpink" if inst.living else "black"


@obj
class Particle:
    living: bool
    cell = Cell(0, 0, _color_of_living)


@obj
class Button:
    color: str

    @staticmethod
    def _button_color(inst) -> str:
        return inst.color

    cell = Cell(0, 0, _button_color)


def _initial_particles() -> list:
    seeds = {
        (2, 3), (3, 3), (3, 1), (4, 3), (4, 2),
    }
    out = []
    for pos in allPositions(GRID_SIZE):
        out.append(Particle((pos.x, pos.y) in seeds, pos))
    return out


def _neighbours(p: Position) -> list[Position]:
    return [
        Position(p.x + 1, p.y + 1), Position(p.x + 1, p.y - 1),
        Position(p.x - 1, p.y + 1), Position(p.x - 1, p.y - 1),
        Position(p.x,     p.y + 1), Position(p.x,     p.y - 1),
        Position(p.x + 1, p.y),     Position(p.x - 1, p.y),
    ]


def _life_step(particles: list) -> list:
    living_origins = {(o.origin.x, o.origin.y) for o in particles if o.living}

    def update(pobj):
        nbr_positions = _neighbours(pobj.origin)
        live_nbrs = sum(1 for p in nbr_positions if (p.x, p.y) in living_origins)
        if pobj.living:
            if live_nbrs <= 1 or live_nbrs >= 4:
                return updateObj(pobj, "living", False)
            return pobj
        if live_nbrs == 3:
            return updateObj(pobj, "living", True)
        return pobj

    return updateObj(particles, update)


@program(grid_size=GRID_SIZE)
class GameOfLife:
    particles = StateVar(list)
    buttonNext = StateVar(object)
    buttonReset = StateVar(object)

    @particles.initializer
    def _():
        return _initial_particles()

    @buttonNext.initializer
    def _():
        return Button("green", Position(0, GRID_SIZE - 1))

    @buttonReset.initializer
    def _():
        return Button("silver", Position(GRID_SIZE - 1, GRID_SIZE - 1))

    @particles.next
    def _():
        return prev(GameOfLife.particles)

    @buttonNext.next
    def _():
        return prev(GameOfLife.buttonNext)

    @buttonReset.next
    def _():
        return prev(GameOfLife.buttonReset)

    # Click a grid cell → toggle that particle's living bit.
    @on(lambda: clicked() and not clicked(prev(GameOfLife.buttonNext)) and not clicked(prev(GameOfLife.buttonReset)))
    def _():
        ps = prev(GameOfLife.particles)
        target_xy = (click.x, click.y)
        def _toggle(p):
            if (p.origin.x, p.origin.y) == target_xy:
                return updateObj(p, "living", not p.living)
            return p
        GameOfLife.particles.set(updateObj(ps, _toggle))

    # Click the green button → advance one Life generation.
    @on(lambda: clicked(prev(GameOfLife.buttonNext)))
    def _():
        GameOfLife.particles.set(_life_step(prev(GameOfLife.particles)))

    # Click the silver button → reset (kill all).
    @on(lambda: clicked(prev(GameOfLife.buttonReset)))
    def _():
        ps = prev(GameOfLife.particles)
        GameOfLife.particles.set([updateObj(p, "living", False) for p in ps])
