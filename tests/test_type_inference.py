"""Pin the static type-inference handler (spike: unify-backed, raw types).

Each test runs an expression under TypeOfHandler and verifies the result
*Python type*, including the uniformChoice : list[T] -> T rule that runtime
checking can't enforce statically. The representation is plain types now
(no bespoke Type token), so assertions compare against int / Position /
list[...] / the @obj factory directly.
"""
from __future__ import annotations

import typing

import pytest

from autumn_py import Cell, Position, StateVar, obj, prev, program
from autumn_py.inference import env_from_program, infer_type
from autumn_py.stdlib import (
    addObj,
    adjPositions,
    removeObj,
    uniformChoice,
    updateObj,
)


def _elem(t):
    """Element type of a list/sequence type token."""
    return typing.get_args(t)[0]


# -------------------------------------------------------------------------
# Direct handler tests: each @defop returns the right type
# -------------------------------------------------------------------------

def test_prev_returns_state_var_type():
    assert infer_type(lambda: prev("counter"), {"counter": int}) is int


def test_prev_on_unbound_var_raises():
    with pytest.raises(NameError):
        infer_type(lambda: prev("missing"), {"counter": int})


def test_uniform_choice_on_typed_list_returns_element_type():
    """The rule that runtime _check_type cannot enforce."""
    env = {"particles": list[Position]}
    assert infer_type(lambda: uniformChoice(prev("particles")), env) is Position


def test_uniform_choice_on_concrete_list_uses_element_type():
    assert infer_type(lambda: uniformChoice([1, 2, 3]), {}) is int


# -------------------------------------------------------------------------
# Pipeline: prev → adjPositions → uniformChoice
# -------------------------------------------------------------------------

def test_adj_positions_returns_list_of_position():
    result = infer_type(lambda: adjPositions(Position), {})
    assert typing.get_origin(result) is list
    assert _elem(result) is Position


def test_random_walk_pattern_pins_uniform_choice_to_position():
    t = infer_type(lambda: uniformChoice(adjPositions(Position)), {})
    assert t is Position


# -------------------------------------------------------------------------
# Object-list operations preserve list type
# -------------------------------------------------------------------------

def test_add_obj_preserves_list_type_in_inference():
    @obj
    class Particle:
        cell = Cell(0, 0, "blue")

    env = {"particles": list[Particle]}
    t = infer_type(lambda: addObj(prev("particles"), Particle), env)
    assert t == list[Particle]


def test_remove_obj_preserves_list_type():
    env = {"items": list[int]}
    t = infer_type(lambda: removeObj(prev("items"), lambda x: True), env)
    assert t == list[int]


def test_update_obj_on_list_preserves_list_type():
    env = {"items": list[int]}
    t = infer_type(lambda: updateObj(prev("items"), lambda x: x + 1), env)
    assert t == list[int]


# -------------------------------------------------------------------------
# Integration with @program: env_from_program builds the type env
# -------------------------------------------------------------------------

def test_env_from_program_extracts_state_var_types():
    @program()
    class P:
        counter = StateVar(int, init=0)
        name = StateVar(str, init="hello")

    env = env_from_program(P)
    assert env["counter"] is int
    assert env["name"] is str


def test_env_from_program_extracts_parameterised_list_types():
    @program()
    class P:
        items = StateVar(list[int], init=[])

    env = env_from_program(P)
    assert typing.get_origin(env["items"]) is list
    assert _elem(env["items"]) is int


def test_full_particles_next_expression_types_through():
    """End-to-end: a particles-style next-expression types under the
    program's env — the @obj nominal element type is recovered."""
    @obj
    class Particle:
        cell = Cell(0, 0, "blue")

    @program()
    class P:
        particles = StateVar(list[Particle], init=[])

    env = env_from_program(P)
    t = infer_type(lambda: uniformChoice(prev("particles")), env)
    assert hasattr(t, "__autumn_obj_spec__")
    assert t.__autumn_obj_spec__.name == "Particle"
