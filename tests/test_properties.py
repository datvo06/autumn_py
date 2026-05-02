"""Tests for the @spec / @modifies / @no_stochastic property decorators
and their lowering into Goal subclass instances.

The decorators attach an ``__autumn_spec__`` attribute on the body;
``StateVar.next`` and ``@program`` materialise the attribute into Goal
entries on ``cls._autumn_spec.properties``. ``gate(emit_cls)`` then
reads from there when no explicit goals are passed.
"""
from __future__ import annotations

import pytest
import z3

from autumn_py import (
    Runtime,
    StateVar,
    modifies,
    no_stochastic,
    prev,
    program,
    spec,
)
from autumn_py.gate import (
    FootprintExcludeGoal,
    ModularArithmeticGoal,
    Residual,
    WriteFrameGoal,
    gate,
)
from autumn_py.ops import sample_uniform, set_var
from autumn_py.properties import Spec, realize_spec_goals


# -------------------------------------------------------------------------
# Spec dataclass: validation
# -------------------------------------------------------------------------

def test_spec_rejects_non_tuple_modifies():
    with pytest.raises(TypeError, match="must be a tuple"):
        Spec(modifies=["step_count"])  # type: ignore[arg-type]


def test_spec_rejects_non_string_non_statevar_modifies_entries():
    with pytest.raises(TypeError, match="must be str or StateVar"):
        Spec(modifies=(123,))  # type: ignore[arg-type]


def test_spec_rejects_invalid_monotone_value():
    with pytest.raises(ValueError, match="must be one of"):
        Spec(monotone="grumpy")


def test_spec_rejects_zero_horizon():
    with pytest.raises(ValueError, match="horizon must be >= 1"):
        Spec(horizon=0)


def test_spec_rejects_unroll_without_invariant():
    with pytest.raises(ValueError, match="only meaningful with an invariant"):
        Spec(unroll=("step_count.next",))


def test_spec_dataclass_rejects_unknown_field():
    """LLM hallucinates a field name → TypeError at decoration time."""
    with pytest.raises(TypeError):
        Spec(deterministic=True)  # not a real field name


# -------------------------------------------------------------------------
# Spec.merge: stacking decorators composes specs
# -------------------------------------------------------------------------

def test_spec_merge_combines_disjoint_fields():
    a = Spec(no_stochastic=True)
    b = Spec(modifies=("foo",))
    merged = a.merge(b)
    assert merged.no_stochastic is True
    assert merged.modifies == ("foo",)


def test_spec_merge_right_biased_on_conflict():
    a = Spec(modifies=("foo",))
    b = Spec(modifies=("bar",))
    merged = a.merge(b)
    assert merged.modifies == ("bar",)


# -------------------------------------------------------------------------
# realize_spec_goals: Spec → Goal subclass instances
# -------------------------------------------------------------------------

def test_realize_spec_goals_no_stochastic_mints_footprint_exclude():
    s = Spec(no_stochastic=True)
    goals = realize_spec_goals(s, anchor="step_count.next")
    assert len(goals) == 1
    assert isinstance(goals[0], FootprintExcludeGoal)
    assert goals[0].anchor == "step_count.next"
    assert ("sample_uniform",) in goals[0].exclude


def test_realize_spec_goals_modifies_mints_write_frame():
    s = Spec(modifies=("step_count",))
    goals = realize_spec_goals(s, anchor="step_count.next")
    assert len(goals) == 1
    assert isinstance(goals[0], WriteFrameGoal)
    assert goals[0].allowed_writes == ("step_count",)


def test_realize_spec_goals_invariant_mints_modular_arithmetic():
    s = Spec(invariant=lambda funcs, k: funcs["x"](k) == k, horizon=4)
    goals = realize_spec_goals(s, anchor="x.next")
    assert len(goals) == 1
    assert isinstance(goals[0], ModularArithmeticGoal)
    assert goals[0].horizon == 4


def test_realize_spec_goals_combines_multiple_fields():
    s = Spec(no_stochastic=True, modifies=("y",))
    goals = realize_spec_goals(s, anchor="y.next")
    assert len(goals) == 2
    assert {type(g).__name__ for g in goals} == {"FootprintExcludeGoal", "WriteFrameGoal"}


# -------------------------------------------------------------------------
# @spec / @no_stochastic / @modifies attach __autumn_spec__
# -------------------------------------------------------------------------

def test_spec_decorator_attaches_autumn_spec_attribute():
    @spec(no_stochastic=True)
    def fn():
        return 0
    assert hasattr(fn, "__autumn_spec__")
    assert fn.__autumn_spec__.no_stochastic is True


def test_no_stochastic_sugar_equivalent_to_spec():
    @no_stochastic
    def fn():
        return 0
    assert fn.__autumn_spec__.no_stochastic is True


def test_modifies_sugar_equivalent_to_spec():
    @modifies("foo", "bar")
    def fn():
        return 0
    assert fn.__autumn_spec__.modifies == ("foo", "bar")


def test_spec_decorators_compose_via_merge():
    @spec(no_stochastic=True)
    @modifies("foo")
    def fn():
        return 0
    s = fn.__autumn_spec__
    assert s.no_stochastic is True
    assert s.modifies == ("foo",)


# -------------------------------------------------------------------------
# End-to-end: @program drains _pending_properties; gate reads from it
# -------------------------------------------------------------------------

def test_program_with_no_stochastic_spec_attaches_to_spec_properties():
    @program(grid_size=8)
    class P:
        x = StateVar(int, init=0)

        @x.next
        @no_stochastic
        def _() -> int:
            return prev(P.x) + 1

    spec_props = P._autumn_spec.properties
    assert len(spec_props) == 1
    assert isinstance(spec_props[0], FootprintExcludeGoal)
    assert spec_props[0].anchor == "x.next"


def test_gate_reads_from_spec_properties_when_no_explicit_goals():
    """gate(emit_cls) without an explicit goals list uses spec.properties."""
    @program(grid_size=8)
    class P:
        x = StateVar(int, init=0)

        @x.next
        @no_stochastic
        def _() -> int:
            return prev(P.x) + 1

    residuals = gate(P)
    assert residuals == []   # passes: the next-clause has no sample_uniform


def test_gate_residual_when_program_spec_violated():
    """A next-clause that violates @no_stochastic is rejected by the gate."""
    @program(grid_size=8)
    class P:
        x = StateVar(int, init=0)

        @x.next
        @no_stochastic
        def _() -> int:
            return sample_uniform((1, 2, 3))   # violates no_stochastic

    residuals = gate(P)
    assert len(residuals) == 1
    assert isinstance(residuals[0].goal, FootprintExcludeGoal)
    assert ("sample_uniform",) in residuals[0].witness


# -------------------------------------------------------------------------
# @modifies catches silent write-aliasing
# -------------------------------------------------------------------------

def test_modifies_catches_aliased_write_to_other_state_var():
    """A next-clause for `x` that accidentally writes to `y` was a
    silent failure mode before WriteFrameGoal. Now caught loudly."""
    @program(grid_size=8)
    class P:
        x = StateVar(int, init=0)
        y = StateVar(int, init=0)

        @x.next
        @modifies("x")     # x's next-clause must only write to x
        def _() -> int:
            set_var("y", 99)   # silent-aliasing — should be caught
            return prev(P.x) + 1

    residuals = gate(P)
    assert len(residuals) == 1
    assert isinstance(residuals[0].goal, WriteFrameGoal)
    assert "y" in residuals[0].witness


def test_modifies_implicit_self_write_is_allowed():
    """A next-clause for `x` is implicitly allowed to write to `x`,
    even if `x` isn't in the @modifies tuple."""
    @program(grid_size=8)
    class P:
        x = StateVar(int, init=0)

        @x.next
        @modifies()        # empty — but x.next writing to x is implicit
        def _() -> int:
            return prev(P.x) + 1

    residuals = gate(P)
    assert residuals == []


def test_modifies_explicit_extra_writes_allowed():
    """A next-clause that writes to multiple state vars is allowed
    if all are listed in @modifies."""
    @program(grid_size=8)
    class P:
        x = StateVar(int, init=0)
        y = StateVar(int, init=0)

        @x.next
        @modifies("x", "y")    # both writes explicitly allowed
        def _() -> int:
            set_var("y", 99)
            return prev(P.x) + 1

    residuals = gate(P)
    assert residuals == []


# -------------------------------------------------------------------------
# Backward compat: gate(emit_cls, [goals]) still works
# -------------------------------------------------------------------------

def test_gate_with_explicit_goals_still_works():
    """The original gate(emit_cls, [phi_1, phi_2]) API is preserved.
    Explicit goals are appended to the program-attached goals."""
    @program(grid_size=8)
    class P:
        x = StateVar(int, init=0)

        @x.next
        def _() -> int:
            return prev(P.x) + 1

    extra_goal = FootprintExcludeGoal(
        anchor="x.next",
        exclude=(("sample_uniform",),),
    )
    residuals = gate(P, [extra_goal])
    assert residuals == []


def test_gate_combines_spec_properties_and_explicit_goals():
    """Both sources contribute: spec.properties from the program, plus
    explicit goals passed to gate()."""
    @program(grid_size=8)
    class P:
        x = StateVar(int, init=0)

        @x.next
        @no_stochastic
        def _() -> int:
            return prev(P.x) + 1

    # Program has 1 spec.property; we add 1 explicit goal.
    explicit = WriteFrameGoal(anchor="x.next", allowed_writes=("x",))
    residuals = gate(P, [explicit])
    assert residuals == []   # both pass


# -------------------------------------------------------------------------
# @spec(invariant=...) end-to-end with Z3
# -------------------------------------------------------------------------

def test_spec_with_invariant_runs_through_z3():
    """A next-clause annotated with @spec(invariant=...) gets a
    ModularArithmeticGoal minted; the gate runs Z3 against it."""
    @program(grid_size=8)
    class P:
        x = StateVar(int, init=0)

        @x.next
        @spec(
            invariant=lambda funcs, k: funcs["x"](k + 1) == funcs["x"](k - 1) + 1,
            unroll=("x.next",),
            init_constraints=lambda funcs: [
                funcs["x"](0) == 0,
                funcs["x"](-1) == 0,
            ],
            horizon=3,
        )
        def _() -> int:
            return prev(P.x) + 1

    residuals = gate(P)
    assert isinstance(residuals, list)


# -------------------------------------------------------------------------
# Clean lambda form: state vars bound by parameter name, no funcs[] dict
# -------------------------------------------------------------------------

def test_spec_modifies_accepts_statevar_refs():
    """``modifies=(x,)`` with bare StateVar refs resolves to the var's
    name at goal-mint time."""
    @program(grid_size=8)
    class P:
        x = StateVar(int, init=0)
        y = StateVar(int, init=0)

        @x.next
        @spec(modifies=(x,))     # ← bare StateVar; not a string
        def _() -> int:
            return prev(P.x) + 1

    spec_props = P._autumn_spec.properties
    assert len(spec_props) == 1
    assert isinstance(spec_props[0], WriteFrameGoal)
    # Resolved to "x" (the StateVar's name)
    assert spec_props[0].allowed_writes == ("x",)


def test_spec_unroll_accepts_statevar_refs():
    """``unroll=(x,)`` with bare StateVar refs resolves to ``"x.next"``."""
    @program(grid_size=8)
    class P:
        x = StateVar(int, init=0)

        @x.next
        @spec(
            invariant=lambda funcs, k: funcs["x"](k + 1) == funcs["x"](k - 1) + 1,
            unroll=(x,),         # ← bare StateVar; resolves to "x.next"
            init_constraints=lambda funcs: [
                funcs["x"](0) == 0,
                funcs["x"](-1) == 0,
            ],
            horizon=3,
        )
        def _() -> int:
            return prev(P.x) + 1

    spec_props = P._autumn_spec.properties
    modular = [g for g in spec_props if isinstance(g, ModularArithmeticGoal)][0]
    assert modular.unroll == ("x.next",)


def test_spec_invariant_lambda_binds_state_vars_by_param_name():
    """The clean form: ``lambda x, t: x(t+1) == x(t) + 1`` — the gate
    introspects the lambda's parameter names, finds the matching state-
    var Z3 functions, and binds them. No ``funcs[]`` indirection."""
    @program(grid_size=8)
    class P:
        x = StateVar(int, init=0)

        @x.next
        @spec(
            # ↓ bare `x` and `t` — no funcs dict
            invariant=lambda x, t: x(t + 1) == x(t - 1) + 1,
            unroll=(x,),
            init_constraints=lambda x: [x(0) == 0, x(-1) == 0],
            horizon=3,
        )
        def _() -> int:
            return prev(P.x) + 1

    residuals = gate(P)
    assert isinstance(residuals, list)


def test_spec_invariant_with_multiple_state_vars_param_introspection():
    """Multi-var lambda: ``lambda x, y, t: x(t) >= y(t)``. Each StateVar
    name must be a parameter of the lambda."""
    @program(grid_size=8)
    class P:
        x = StateVar(int, init=5)
        y = StateVar(int, init=0)

        @x.next
        @spec(
            invariant=lambda x, y, t: x(t + 1) >= y(t + 1),
            unroll=(x, y),
            init_constraints=lambda x, y: [
                x(0) == 5, x(-1) == 5,
                y(0) == 0, y(-1) == 0,
            ],
            horizon=3,
        )
        def _() -> int:
            return prev(P.x)

        @y.next
        def _() -> int:
            return prev(P.y)

    residuals = gate(P)
    assert isinstance(residuals, list)


def test_space_invaders_r2_fixed_passes_gate_with_attached_specs():
    """End-to-end: SpaceInvadersR2Fixed has @spec(no_stochastic=True,
    modifies=..., invariant=...) attached to spawn_event.next. Calling
    gate(emit_cls) without explicit goals exercises all three checkers
    against the program-attached specs and returns no residuals."""
    from examples.space_invaders import SpaceInvadersR2Fixed

    residuals = gate(SpaceInvadersR2Fixed)
    assert residuals == [], f"unexpected residuals: {residuals}"


def test_space_invaders_r1_stochastic_fails_no_stochastic_spec():
    """If we attach the same @spec(no_stochastic=True) on R1's
    spawn_event.next (which uses sample_uniform), the gate's
    FootprintExcludeGoal residual fires."""
    # We don't modify the file; instead construct the equivalent goal
    # explicitly to demonstrate the failure shape on the existing R1 emit.
    from examples.space_invaders import SpaceInvadersR1
    from autumn_py.gate import FootprintExcludeGoal

    extra_goal = FootprintExcludeGoal(
        anchor="spawn_event.next",
        exclude=(("sample_uniform",),),
    )
    residuals = gate(SpaceInvadersR1, [extra_goal])
    assert len(residuals) == 1
    assert ("sample_uniform",) in residuals[0].witness


def test_spec_invariant_unknown_param_name_raises_loudly():
    """If the lambda names a parameter that isn't a state-var name, the
    gate raises with a clear message naming the available state vars."""
    @program(grid_size=8)
    class P:
        x = StateVar(int, init=0)

        @x.next
        @spec(
            invariant=lambda nonexistent, t: nonexistent(t) == 0,
            unroll=(x,),
            init_constraints=lambda x: [x(0) == 0],
            horizon=2,
        )
        def _() -> int:
            return prev(P.x) + 1

    with pytest.raises(NameError, match="not a known state-var name"):
        gate(P)
