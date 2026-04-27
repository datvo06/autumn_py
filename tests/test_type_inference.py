"""Pin the static type-inference handler.

Each test runs an expression under TypeOfHandler and verifies the result
type, including the §2.2 uniformChoice : List T → T rule that runtime
checking can't enforce statically.
"""
from __future__ import annotations

import pytest

from autumn_py import Cell, Position, StateVar, obj, prev, program
from autumn_py.inference import Type, env_from_program, infer_type
from autumn_py.stdlib import (
    addObj,
    adjPositions,
    removeObj,
    uniformChoice,
    updateObj,
)
from autumn_py.values import ObjectInstance


# -------------------------------------------------------------------------
# Type token construction
# -------------------------------------------------------------------------

def test_type_of_lifts_python_annotations():
    assert Type.of(int) == Type(int)
    assert Type.of(bool) == Type(bool)
    assert Type.of(str) == Type(str)
    assert Type.of(Position) == Type(Position)


def test_type_of_handles_parameterised_list():
    t = Type.of(list[int])
    assert t.is_list
    assert t.elem == Type(int)


def test_type_of_handles_obj_factory():
    @obj
    class Particle:
        cell = Cell(0, 0, "blue")

    t = Type.of(Particle)
    assert t.is_obj
    assert t.obj_spec.name == "Particle"


# -------------------------------------------------------------------------
# Direct handler tests: each @defop returns the right Type
# -------------------------------------------------------------------------

def test_prev_returns_state_var_type():
    env = {"counter": Type(int)}
    t = infer_type(lambda: prev("counter"), env)
    assert t == Type(int)


def test_prev_on_unbound_var_raises():
    env = {"counter": Type(int)}
    with pytest.raises(NameError):
        infer_type(lambda: prev("missing"), env)


def test_uniform_choice_on_typed_list_returns_element_type():
    """The §2.2 rule that runtime _check_type cannot enforce."""
    env = {"particles": Type.list_of(Type(Position))}
    t = infer_type(lambda: uniformChoice(prev("particles")), env)
    assert t == Type(Position)


def test_uniform_choice_on_concrete_list_uses_first_element():
    env = {}
    t = infer_type(lambda: uniformChoice([1, 2, 3]), {})
    assert t == Type(int)


# -------------------------------------------------------------------------
# Pipeline tests: prev → adjPositions → uniformChoice
# -------------------------------------------------------------------------

def test_adj_positions_returns_list_of_position():
    """Type-domain dispatch: adjPositions(Type[Position]) = Type[list[Position]].

    Routed through `adjPositions_op` so TypeOfHandler can intercept.
    """
    result = infer_type(lambda: adjPositions(Type(Position)), {})
    assert result.is_list
    assert result.elem == Type(Position)


def test_random_walk_pattern_pins_uniform_choice_to_position():
    """The canonical particles next-expression pattern, type-checked.

    `uniformChoice(adjPositions(o.origin))` should return `Type(Position)`
    when `o` is a `Type(Particle)`.
    """
    @obj
    class Particle:
        cell = Cell(0, 0, "blue")

    # Skip the `o.origin` step (would require Type field-projection); jump
    # straight to a Position type as input.
    env = {}
    t = infer_type(lambda: uniformChoice(adjPositions(Type(Position))), env)
    assert t == Type(Position)


# -------------------------------------------------------------------------
# Object-list operations preserve list type
# -------------------------------------------------------------------------

def test_add_obj_preserves_list_type_in_inference():
    @obj
    class Particle:
        cell = Cell(0, 0, "blue")

    elem_t = Type(ObjectInstance, obj_spec=Particle.__autumn_obj_spec__)
    list_t = Type.list_of(elem_t)
    env = {"particles": list_t}
    t = infer_type(lambda: addObj(prev("particles"), elem_t), env)
    assert t == list_t


def test_remove_obj_preserves_list_type():
    elem_t = Type(int)
    list_t = Type.list_of(elem_t)
    env = {"items": list_t}
    t = infer_type(lambda: removeObj(prev("items"), lambda x: True), env)
    assert t == list_t


def test_update_obj_on_list_preserves_list_type():
    elem_t = Type(int)
    list_t = Type.list_of(elem_t)
    env = {"items": list_t}
    t = infer_type(lambda: updateObj(prev("items"), lambda x: x + 1), env)
    assert t == list_t


# -------------------------------------------------------------------------
# Integration with @program: env_from_program builds the type env
# -------------------------------------------------------------------------

def test_env_from_program_extracts_state_var_types():
    @program()
    class P:
        counter = StateVar(int, init=0)
        name = StateVar(str, init="hello")

    env = env_from_program(P)
    assert env["counter"] == Type(int)
    assert env["name"] == Type(str)


def test_env_from_program_extracts_parameterised_list_types():
    @program()
    class P:
        items = StateVar(list[int], init=[])

    env = env_from_program(P)
    assert env["items"].is_list
    assert env["items"].elem == Type(int)


def test_full_particles_next_expression_types_through():
    """End-to-end: a particles-style next-expression types under the
    program's env."""
    @obj
    class Particle:
        cell = Cell(0, 0, "blue")

    @program()
    class P:
        particles = StateVar(list[Particle], init=[])

    env = env_from_program(P)
    # The element type recovered from the typed prev list:
    t = infer_type(lambda: uniformChoice(prev("particles")), env)
    assert t.is_obj
    assert t.obj_spec.name == "Particle"
