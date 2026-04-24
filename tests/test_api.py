from __future__ import annotations

import pytest

from autumn_py import Cell, StateVar, clicked, obj, on, program


def test_state_var_captures_its_attribute_name():
    class C:
        foo = StateVar(list, init=[])

    assert C.foo.name == "foo"
    assert C.foo.init == []


def test_state_var_next_registers_callable():
    class C:
        foo = StateVar(list, init=[])

        @foo.next
        def _():
            return "bar"

    assert C.foo._next_fn is not None
    assert C.foo._next_fn() == "bar"


def test_obj_decorator_builds_spec_and_factory():
    @obj
    class Widget:
        cell = Cell(0, 0, "green")

    spec = Widget.__autumn_obj_spec__
    assert spec.name == "Widget"
    assert spec.cells == (Cell(0, 0, "green"),)


def test_obj_decorator_requires_cell_or_cells():
    with pytest.raises(ValueError):
        @obj
        class Bad:
            pass


def test_on_decorator_tags_function_with_predicate():
    @on(clicked)
    def body():
        pass

    assert body.__autumn_on_pred__ is clicked


def test_program_decorator_collects_state_and_on_clauses():
    @obj
    class W:
        cell = Cell(0, 0, "red")

    @program(grid_size=8)
    class P:
        thing = StateVar(list, init=[])

        @on(clicked)
        def _click():
            pass

    spec = P._autumn_spec
    assert spec.config == {"grid_size": 8}
    assert [sv.name for sv in spec.state_vars] == ["thing"]
    assert len(spec.on_clauses) == 1
    assert spec.on_clauses[0].predicate is clicked
