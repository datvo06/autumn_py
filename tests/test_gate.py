"""End-to-end gate runs against the `space_invaders` running example.

These tests instantiate the gate machinery from `autumn_py/gate.py`
against three concrete `@program` emits in `examples/space_invaders.py`,
mirroring the rounds 1 → 2 → 2-fixed walkthrough in
`drafts/autumn-pl-handlers-and-properties.md` Part 1 D4.
"""
from __future__ import annotations

import z3

from autumn_py.gate import (
    FootprintExcludeGoal,
    ModularArithmeticGoal,
    Residual,
    gate,
    select_ast,
)
from autumn_py.smt import read_set
from examples.space_invaders import (
    SpaceInvadersR1,
    SpaceInvadersR2Fixed,
    SpaceInvadersR2OffByOne,
)


# -------------------------------------------------------------------------
# Concrete goal specifications mirroring P_1 and P_3
# -------------------------------------------------------------------------

# P_1: spawn-event guard has no sample_uniform atom in its read-set.
phi_1_no_stochastic = FootprintExcludeGoal(
    anchor="spawn_event.next",
    exclude=(("sample_uniform",),),
)


def _phi_2_goal(emit_cls):
    """Build the modular-arithmetic goal for the spawn-timing predicate:
    `forall t in [0, horizon). spawn_event(t+1) iff (step_count(t) mod 15 == 3)`.

    init_constraints binds:
    * step_count(t) = t for all unrolled ticks
    * next_spawn_step at tick 0 and tick -1 to the StateVar's init value
      (the latter anchors `prev(next_spawn_step)` at tick 0).
    """
    init_value = select_ast(emit_cls, "next_spawn_step.init")

    def init_constraints(funcs):
        step_count = funcs["step_count"]
        next_spawn_step = funcs["next_spawn_step"]
        return [
            *(step_count(k) == k for k in range(8)),
            next_spawn_step(0) == init_value,
            next_spawn_step(-1) == init_value,
        ]

    def goal_factory(funcs, k):
        spawn_event = funcs["spawn_event"]
        return spawn_event(k + 1) == ((k % 15) == 3)

    return ModularArithmeticGoal(
        anchor="spawn_event.next",
        unroll=("next_spawn_step.next", "spawn_event.next"),
        init_constraints=init_constraints,
        goal_factory=goal_factory,
        horizon=6,
    )


# -------------------------------------------------------------------------
# Sanity checks — select_ast resolves anchors
# -------------------------------------------------------------------------

def test_select_ast_resolves_next_clause_callable():
    target = select_ast(SpaceInvadersR1, "spawn_event.next")
    assert callable(target)


def test_select_ast_resolves_init_value():
    target = select_ast(SpaceInvadersR2OffByOne, "next_spawn_step.init")
    assert target == 4
    target_fixed = select_ast(SpaceInvadersR2Fixed, "next_spawn_step.init")
    assert target_fixed == 3


def test_select_ast_raises_on_unknown_anchor():
    import pytest
    with pytest.raises(KeyError):
        select_ast(SpaceInvadersR1, "no_such_var.next")


# -------------------------------------------------------------------------
# read_set on the round-1 spawn-event next clause
# -------------------------------------------------------------------------

def test_read_set_round_1_includes_sample_uniform():
    target = select_ast(SpaceInvadersR1, "spawn_event.next")
    atoms = read_set(target)
    assert ("sample_uniform",) in atoms
    # Also reads step_count
    assert ("get_var", "step_count", 0) in atoms


def test_read_set_round_2_excludes_sample_uniform():
    target = select_ast(SpaceInvadersR2OffByOne, "spawn_event.next")
    atoms = read_set(target)
    assert ("sample_uniform",) not in atoms
    # Reads step_count and prev(next_spawn_step)
    assert ("get_var", "step_count", 0) in atoms
    assert ("get_var", "next_spawn_step", -1) in atoms


# -------------------------------------------------------------------------
# Gate run on the round-1 stochastic emit — P_1 fails
# -------------------------------------------------------------------------

def test_gate_round_1_fails_phi_1():
    """The round-1 stochastic emit fails P_1; the residual carries the
    offending atom (sample_uniform)."""
    residuals = gate(SpaceInvadersR1, [phi_1_no_stochastic])
    assert len(residuals) == 1
    r = residuals[0]
    assert isinstance(r.goal, FootprintExcludeGoal)
    assert r.goal.anchor == "spawn_event.next"
    assert ("sample_uniform",) in r.witness


# -------------------------------------------------------------------------
# Gate run on round-2 off-by-one — P_1 passes, P_3 fails
# -------------------------------------------------------------------------

def test_gate_round_2_off_by_one_fails_phi_2_with_smt_witness():
    """Round-2 off-by-one passes P_1 (no stochastic) but fails P_3
    (modular arithmetic — Z3 returns a counterexample model that
    pinpoints the off-by-one tick)."""
    residuals = gate(
        SpaceInvadersR2OffByOne,
        [phi_1_no_stochastic, _phi_2_goal(SpaceInvadersR2OffByOne)],
    )
    assert len(residuals) == 1
    r = residuals[0]
    assert isinstance(r.goal, ModularArithmeticGoal)
    assert isinstance(r.witness, dict)


# -------------------------------------------------------------------------
# Gate run on round-2 fixed — both P_1 and P_3 pass
# -------------------------------------------------------------------------

def test_gate_round_2_fixed_passes_all_goals():
    """Round-2 with the spawn constant bound to 3 (the SMT
    counterexample's witness) passes both P_1 and P_3 — gate returns
    no residuals, emit is committed."""
    residuals = gate(
        SpaceInvadersR2Fixed,
        [phi_1_no_stochastic, _phi_2_goal(SpaceInvadersR2Fixed)],
    )
    assert residuals == []


# -------------------------------------------------------------------------
# The full round-1 → round-2 trajectory in one test
# -------------------------------------------------------------------------

def test_round_kernel_trajectory_round_1_to_round_2_fixed():
    """The gate as the round-kernel sees three emits in succession.
    Only the third commits."""
    # Round 1 fails phi_1
    r1 = gate(SpaceInvadersR1, [phi_1_no_stochastic])
    assert len(r1) == 1 and isinstance(r1[0].goal, FootprintExcludeGoal)

    # Round 2 (off-by-one) passes phi_1 but fails phi_2
    r2 = gate(
        SpaceInvadersR2OffByOne,
        [phi_1_no_stochastic, _phi_2_goal(SpaceInvadersR2OffByOne)],
    )
    assert len(r2) == 1 and isinstance(r2[0].goal, ModularArithmeticGoal)

    # Round 2-fixed passes both
    r2f = gate(
        SpaceInvadersR2Fixed,
        [phi_1_no_stochastic, _phi_2_goal(SpaceInvadersR2Fixed)],
    )
    assert r2f == []
