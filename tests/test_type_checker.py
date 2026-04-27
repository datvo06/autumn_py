"""Pin down the lightweight type checker's behaviour.

The checker validates state-var init values, @obj field values, initializer
return types, and next-expression return types against their declared
annotations. It is intentionally permissive (object/None pass; parameterised
generics pass through) and strict on ground types and @obj-class instances.
"""
from __future__ import annotations

import pytest

from autumn_py import Cell, Position, Runtime, StateVar, obj, on, prev, program
from autumn_py.api import TypeMismatch, _check_type
from autumn_py.events import clicked


# -------------------------------------------------------------------------
# _check_type unit tests
# -------------------------------------------------------------------------

def test_check_type_passes_matching_ground_types():
    _check_type(0, int, "test")
    _check_type([], list, "test")
    _check_type("sand", str, "test")
    _check_type(True, bool, "test")
    _check_type(Position(1, 2), Position, "test")


def test_check_type_rejects_mismatched_ground_types():
    with pytest.raises(TypeMismatch):
        _check_type(0, list, "test")
    with pytest.raises(TypeMismatch):
        _check_type("hello", int, "test")
    with pytest.raises(TypeMismatch):
        _check_type([1, 2, 3], str, "test")


def test_check_type_object_is_universal():
    # `object` annotation accepts anything (Mario uses StateVar(object) for
    # the Mario instance whose initializer hasn't run yet).
    _check_type(None, object, "test")
    _check_type(0, object, "test")
    _check_type([1, 2, 3], object, "test")


def test_check_type_none_means_unannotated():
    # `type_=None` means "no annotation given" — accept anything.
    _check_type(0, None, "test")
    _check_type("anything", None, "test")


def test_check_type_obj_factory_check():
    @obj
    class Widget:
        cell = Cell(0, 0, "green")

    w = Widget(Position(0, 0))
    _check_type(w, Widget, "test")  # passes — Widget factory has __autumn_obj_spec__

    # Wrong-class object fails
    @obj
    class Other:
        cell = Cell(0, 0, "red")

    o = Other(Position(0, 0))
    with pytest.raises(TypeMismatch):
        _check_type(o, Widget, "test")


# -------------------------------------------------------------------------
# Integration tests with @obj field annotations
# -------------------------------------------------------------------------

def test_obj_decorator_rejects_wrong_field_type():
    @obj
    class Mario:
        bullets: int
        cell = Cell(0, 0, "red")

    # Correct: bullets is int
    Mario(0, Position(0, 0))

    # Wrong: bullets is a string
    with pytest.raises(TypeMismatch):
        Mario("zero", Position(0, 0))


def test_obj_decorator_accepts_bool_field():
    @obj
    class Sand:
        liquid: bool
        cell = Cell(0, 0, "tan")

    Sand(True, Position(0, 0))
    Sand(False, Position(0, 0))


# -------------------------------------------------------------------------
# Integration tests with StateVar init values and next-expressions
# -------------------------------------------------------------------------

def test_state_var_init_type_is_checked_at_runtime_init():
    @program()
    class P:
        counter = StateVar(int, init="not an int")

    with pytest.raises(TypeMismatch):
        Runtime(P)


def test_state_var_init_type_passes_when_correct():
    @program()
    class P:
        counter = StateVar(int, init=0)
        items = StateVar(list, init=[])

    r = Runtime(P)
    assert r.state.get("counter") == 0
    assert r.state.get("items") == []


def test_state_var_initializer_return_type_is_checked():
    @program()
    class P:
        x = StateVar(int)

        @x.initializer
        def _():
            return "not an int"

    with pytest.raises(TypeMismatch):
        Runtime(P)


def test_state_var_next_expression_return_type_is_checked():
    @program()
    class P:
        x = StateVar(int, init=0)

        @x.next
        def _():
            return [1, 2, 3]  # wrong type

    r = Runtime(P)
    with pytest.raises(TypeMismatch):
        r.step()


def test_state_var_with_object_type_accepts_anything():
    # Mario's pattern: object-valued state var, initializer returns
    # an ObjectInstance via alloc_obj_id.
    @obj
    class Foo:
        cell = Cell(0, 0, "blue")

    @program()
    class P:
        thing = StateVar(object)

        @thing.initializer
        def _():
            return Foo(Position(3, 3))

    r = Runtime(P)
    assert r.state.get("thing").cls.name == "Foo"


# -------------------------------------------------------------------------
# Gap 1: on-clause predicate must return bool (§2.3).
# -------------------------------------------------------------------------

def test_on_clause_predicate_must_return_bool():
    @program()
    class P:
        x = StateVar(int, init=0)

        @on(lambda: 42)  # truthy int, not a bool
        def _():
            P.x.set(P.x.get() + 1)

    r = Runtime(P)
    with pytest.raises(TypeMismatch):
        r.step()


def test_on_clause_predicate_accepts_bool_lambdas():
    @program()
    class P:
        x = StateVar(int, init=0)

        @on(lambda: True)
        def _():
            P.x.set(P.x.get() + 1)

    r = Runtime(P)
    r.step()
    assert r.state.get("x") == 1


def test_on_clause_predicate_accepts_event_sentinels():
    # Sentinel returns bool via __bool__; should pass.
    @program()
    class P:
        x = StateVar(int, init=0)

        @on(clicked)
        def _():
            P.x.set(P.x.get() + 1)

    r = Runtime(P)
    r.step()  # no click; pred returns False (still bool)
    assert r.state.get("x") == 0
    r.click(0, 0)
    r.step()
    assert r.state.get("x") == 1


# -------------------------------------------------------------------------
# Gap 4: list[T] parameterization is checked.
# -------------------------------------------------------------------------

def test_parameterized_list_type_checks_elements():
    @program()
    class P:
        nums = StateVar(list[int], init=[1, 2, 3])

    Runtime(P)  # should pass

    @program()
    class Q:
        nums = StateVar(list[int], init=[1, "two", 3])

    with pytest.raises(TypeMismatch):
        Runtime(Q)


def test_parameterized_list_type_with_object_class():
    @obj
    class Particle:
        cell = Cell(0, 0, "blue")

    @obj
    class Coin:
        cell = Cell(0, 0, "gold")

    @program()
    class P:
        items = StateVar(list[Particle])

        @items.initializer
        def _():
            return [Particle(Position(0, 0)), Particle(Position(1, 1))]

    Runtime(P)  # should pass

    @program()
    class Q:
        items = StateVar(list[Particle])

        @items.initializer
        def _():
            return [Particle(Position(0, 0)), Coin(Position(1, 1))]

    with pytest.raises(TypeMismatch):
        Runtime(Q)


# -------------------------------------------------------------------------
# Gap 5: cell color closure must return str (§2.3).
# -------------------------------------------------------------------------

def test_cell_color_closure_must_return_str():
    @obj
    class Bad:
        flag: bool
        cell = Cell(0, 0, lambda inst: 42 if inst.flag else 0)  # int, not str

    @program()
    class P:
        x = StateVar(object)

        @x.initializer
        def _():
            return Bad(True, Position(0, 0))

    r = Runtime(P)
    with pytest.raises(TypeMismatch):
        r.render_all()


def test_cell_color_closure_accepts_str_returns():
    @obj
    class Good:
        flag: bool
        cell = Cell(0, 0, lambda inst: "red" if inst.flag else "blue")

    @program()
    class P:
        x = StateVar(object)

        @x.initializer
        def _():
            return Good(True, Position(0, 0))

    r = Runtime(P)
    out = r.render_all()
    assert out == [{"x": 0, "y": 0, "color": "red"}]


# -------------------------------------------------------------------------
# Gap 7: function return-type annotations on @.next / @.initializer are
# checked at decoration time against the StateVar's declared type.
# -------------------------------------------------------------------------

def test_next_function_return_annotation_must_match_state_var_type():
    with pytest.raises(TypeMismatch):
        @program()
        class P:
            counter = StateVar(int, init=0)

            @counter.next
            def _() -> str:
                return "wrong"


def test_initializer_function_return_annotation_must_match_state_var_type():
    with pytest.raises(TypeMismatch):
        @program()
        class P:
            counter = StateVar(int)

            @counter.initializer
            def _() -> list:
                return [0]


def test_next_function_return_annotation_passes_when_matching():
    @program()
    class P:
        counter = StateVar(int, init=0)

        @counter.next
        def _() -> int:
            return P.counter.get() + 1

    r = Runtime(P)
    r.step()
    assert r.state.get("counter") == 1


def test_next_function_unannotated_passes():
    # Backwards compatibility: function with no return annotation works.
    @program()
    class P:
        counter = StateVar(int, init=0)

        @counter.next
        def _():
            return P.counter.get() + 1

    r = Runtime(P)
    r.step()
    assert r.state.get("counter") == 1


def test_state_var_with_object_type_skips_return_annotation_check():
    # If StateVar's type_ is `object` (universal), function annotations
    # are not enforced — gives users an escape hatch.
    @program()
    class P:
        thing = StateVar(object)

        @thing.initializer
        def _() -> int:  # would conflict with anything else, but type=object
            return 42

    r = Runtime(P)
    assert r.state.get("thing") == 42


# -------------------------------------------------------------------------
# Gap 3 (partial): prev() on an unbound StateVar raises early.
# -------------------------------------------------------------------------

def test_prev_on_unbound_state_var_raises():
    sv = StateVar(int, init=0)  # not in a class body — __set_name__ never ran
    assert sv.name is None
    with pytest.raises(TypeMismatch):
        prev(sv)
