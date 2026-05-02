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


# -------------------------------------------------------------------------
# StateVar direct-use delegates to .get() — arithmetic / boolean /
# coercion on a StateVar object goes through the get_var op, which
# flows through the installed handler stack (Runtime → concrete value;
# SmtCollectHandler → Z3 expression; read_set → records the atom).
#
# `MyClass.step_count + 1` reads as natural Python AND preserves the
# term/reducibility property — the underlying op is invoked the same
# as if the user had written `MyClass.step_count.get() + 1`.
# -------------------------------------------------------------------------

def test_statevar_arithmetic_delegates_to_get_var_op():
    """`SV + n` calls .get() (→ get_var op), then arithmetic on the
    returned value. Under read_set, the atom is recorded."""
    from autumn_py import StateVar
    from autumn_py.smt import read_set

    sv = StateVar(int, init=0, name="step_count")

    def uses_arithmetic():
        return sv + 1

    atoms = read_set(uses_arithmetic)
    assert ("get_var", "step_count", 0) in atoms


def test_statevar_boolean_coercion_delegates_to_get_var_op():
    """`if SV: ...` calls .get(), then bools the result. Under read_set,
    the atom is still recorded."""
    from autumn_py import StateVar
    from autumn_py.smt import read_set

    sv = StateVar(bool, init=False, name="spawn_event")

    def uses_boolean():
        if sv:
            return "fired"
        return "not fired"

    atoms = read_set(uses_boolean)
    assert ("get_var", "spawn_event", 0) in atoms


def test_statevar_comparison_delegates_to_get_var_op():
    from autumn_py import StateVar
    from autumn_py.smt import read_set

    sv = StateVar(int, init=0, name="step_count")

    def uses_comparison():
        return sv < 5

    atoms = read_set(uses_comparison)
    assert ("get_var", "step_count", 0) in atoms


def test_statevar_eq_and_hash_remain_identity_based():
    """__eq__ and __hash__ are NOT overridden — StateVar must remain
    hashable and dict/set-keyable. Identity-based comparison preserved."""
    from autumn_py.api import StateVar

    sv1 = StateVar(int, init=0, name="step_count")
    sv2 = StateVar(int, init=0, name="step_count")

    assert sv1 == sv1               # same object
    assert sv1 != sv2               # distinct objects (identity)

    d = {sv1: "first", sv2: "second"}
    assert d[sv1] == "first"
    assert d[sv2] == "second"
    assert len({sv1, sv2}) == 2


def test_statevar_explicit_get_set_still_work():
    """The auto-delegation overrides don't break explicit .get()/.set()."""
    from autumn_py import StateVar
    from autumn_py.smt import read_set

    sv = StateVar(int, init=0, name="step_count")

    def reads_via_get():
        return sv.get()

    def writes_via_set():
        sv.set(5)

    assert ("get_var", "step_count", 0) in read_set(reads_via_get)
    assert ("set_var", "step_count") in read_set(writes_via_set)


def test_statevar_arithmetic_works_under_runtime_with_concrete_values():
    """End-to-end: under Runtime, `MyClass.step_count + 1` returns an int.
    Auto-delegation: StateHandler returns concrete value → arithmetic."""
    from autumn_py import Runtime
    from examples.space_invaders import SpaceInvadersR2Fixed

    with Runtime(SpaceInvadersR2Fixed, seed=42) as r:
        r.step()
        # `+ 1` auto-calls .get() (→ get_var → StateHandler returns 1
        # after one step) and does the arithmetic on the int.
        result = SpaceInvadersR2Fixed.step_count + 1
        assert result == 2
