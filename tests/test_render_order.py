"""Pin down Autumn's z-order rendering semantics.

Two laws:

  1. Cross-var: cells belonging to a StateVar declared later in the @program
     class body appear later in render_all()'s output list → drawn on top.
  2. Within-var: cells belonging to instances earlier in an object list appear
     earlier in the output → later-inserted instances draw on top.
"""
from __future__ import annotations

from autumn_py import Cell, Position, Runtime, StateVar, clicked, obj, on, program
from autumn_py.stdlib import addObj


@obj
class _Floor:
    cell = Cell(0, 0, "brown")


@obj
class _Token:
    cell = Cell(0, 0, "gold")


def test_render_order_follows_state_var_declaration_order():
    """Later-declared StateVar's cells must appear AFTER earlier-declared ones.
    Two objects at the SAME grid position — the one declared later wins the
    z-order."""

    @program(grid_size=4)
    class P:
        floors = StateVar(list, init=[])
        tokens = StateVar(list, init=[])

        @on(clicked)
        def _():
            P.floors.set([_Floor(Position(2, 2))])
            P.tokens.set([_Token(Position(2, 2))])

    r = Runtime(P, seed=0)
    r.click(0, 0)
    r.step()
    cells = r.render_all()

    # Both at (2, 2). Floor declared first → floor cell first; token last.
    assert cells == [
        {"x": 2, "y": 2, "color": "brown"},
        {"x": 2, "y": 2, "color": "gold"},
    ]


def test_render_order_within_list_preserves_insertion_order():
    """Within a single StateVar holding a list of instances, cells come out in
    list-insertion order. Objects inserted later render on top."""

    @program(grid_size=4)
    class P:
        particles = StateVar(list, init=[])

        @on(clicked)
        def _():
            # Build the list via successive addObj calls to exercise the
            # allocation-order path (addObj = [*xs, new]).
            xs: list = []
            xs = addObj(xs, _Floor(Position(0, 0)))
            xs = addObj(xs, _Token(Position(1, 1)))
            xs = addObj(xs, _Floor(Position(2, 2)))
            P.particles.set(xs)

    r = Runtime(P, seed=0)
    r.click(0, 0)
    r.step()
    cells = r.render_all()

    assert cells == [
        {"x": 0, "y": 0, "color": "brown"},
        {"x": 1, "y": 1, "color": "gold"},
        {"x": 2, "y": 2, "color": "brown"},
    ]


def test_render_order_cross_var_and_within_var_compose():
    """Combined law: outer loop is state-var declaration order; inner loop is
    within-list insertion order. An earlier-var's last cell still precedes a
    later-var's first cell."""

    @program(grid_size=4)
    class P:
        layer_a = StateVar(list, init=[])
        layer_b = StateVar(list, init=[])

        @on(clicked)
        def _():
            P.layer_a.set([_Floor(Position(0, 0)), _Floor(Position(1, 0))])
            P.layer_b.set([_Token(Position(2, 0)), _Token(Position(3, 0))])

    r = Runtime(P, seed=0)
    r.click(0, 0)
    r.step()
    cells = r.render_all()

    colors_in_order = [c["color"] for c in cells]
    # All layer_a cells (brown) must precede all layer_b cells (gold).
    assert colors_in_order == ["brown", "brown", "gold", "gold"]
