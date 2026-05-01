"""Minimal `space_invaders` emits for the gate's running example.

Three @program classes, each capturing one round of MARA's documented
synthesis trajectory on this env:

* :class:`SpaceInvadersR1` — round 1's stochastic spawn rule
  (``sample_uniform(range(1, 21))`` inside the spawn-event predicate).
  Fails P_1 (the spawn guard reads sample_uniform).

* :class:`SpaceInvadersR2OffByOne` — round 2's deterministic emit with
  the constant bound to 4 (the off-by-one fix the synthesizer made
  before the SMT counterexample). Passes P_1; fails P_3 — Z3 finds
  ``t = 3`` as the counterexample tick where the goal demands a spawn
  but the emit predicts none.

* :class:`SpaceInvadersR2Fixed` — round 2 with the constant bound to 3
  (the value extracted from P_3's SMT counterexample via
  ``bind_constant_from_witness(3)``). Passes both P_1 and P_3.

These are deliberately minimal — a counter, a spawn timing variable,
and a spawn-event predicate. Player ship, enemy formation, and Fire
mechanics are out of scope for the gate demonstration; once the
property machine is in shape, those layers become additional state
vars the same gate machinery checks.
"""
from __future__ import annotations

from autumn_py import StateVar, prev, program
from autumn_py._ast_rewrite import symbolic
from autumn_py.ops import get_var, sample_uniform


# -------------------------------------------------------------------------
# Round 1 — stochastic spawn rule, fails P_1
# -------------------------------------------------------------------------

@program(grid_size=16)
class SpaceInvadersR1:
    """Round-1 emit: spawn timing is drawn fresh each tick from
    ``sample_uniform(range(1, 21))``. The challenger of round 0 will
    report 'every seed shows step 4'; the goal extraction then yields
    ``φ_1 = "no sample_uniform in spawn guard"``, which this emit fails."""

    step_count: int = StateVar(int, init=0)
    spawn_event: bool = StateVar(bool, init=False)

    @step_count.next
    def _() -> int:
        return prev(SpaceInvadersR1.step_count) + 1

    @spawn_event.next
    def _() -> bool:
        # Fresh draw each tick — the round-1 stochastic emit
        spawn_step = sample_uniform(tuple(range(1, 21)))
        return get_var("step_count") == spawn_step


# -------------------------------------------------------------------------
# Round 2 — deterministic, off-by-one. Fails P_3.
# -------------------------------------------------------------------------

@program(grid_size=16)
class SpaceInvadersR2OffByOne:
    """Round-2 emit, off-by-one: the synthesizer adopts determinism
    after challenger feedback but binds the spawn constant to 4 (read
    'step 4' literally) instead of 3. Passes P_1 — no stochastic call.
    Fails P_3 — at t=3 the trajectory has spawn(3) true but the emit
    predicts spawn(4) instead."""

    step_count: int = StateVar(int, init=0)
    next_spawn_step: int = StateVar(int, init=4)
    spawn_event: bool = StateVar(bool, init=False)

    @step_count.next
    def _() -> int:
        return prev(SpaceInvadersR2OffByOne.step_count) + 1

    @next_spawn_step.next
    @symbolic
    def _() -> int:
        cur_step = get_var("step_count")
        cur_spawn = prev(SpaceInvadersR2OffByOne.next_spawn_step)
        if cur_step == cur_spawn:
            return cur_spawn + 15
        else:
            return cur_spawn

    @spawn_event.next
    def _() -> bool:
        cur = get_var("step_count")
        spawn = prev(SpaceInvadersR2OffByOne.next_spawn_step)
        return cur == spawn


# -------------------------------------------------------------------------
# Round 2 — fixed (constant bound from SMT counterexample). Passes both.
# -------------------------------------------------------------------------

@program(grid_size=16)
class SpaceInvadersR2Fixed:
    """Round-2 emit after ``bind_constant_from_witness(3)`` retry: the
    spawn constant is now 3 (the SMT counterexample tick from P_3 on
    the off-by-one emit). Passes P_1 (no stochastic) and P_3 (modular
    timing matches t mod 15 == 3 across the bounded horizon)."""

    step_count: int = StateVar(int, init=0)
    next_spawn_step: int = StateVar(int, init=3)
    spawn_event: bool = StateVar(bool, init=False)

    @step_count.next
    def _() -> int:
        return prev(SpaceInvadersR2Fixed.step_count) + 1

    @next_spawn_step.next
    @symbolic
    def _() -> int:
        cur_step = get_var("step_count")
        cur_spawn = prev(SpaceInvadersR2Fixed.next_spawn_step)
        if cur_step == cur_spawn:
            return cur_spawn + 15
        else:
            return cur_spawn

    @spawn_event.next
    def _() -> bool:
        cur = get_var("step_count")
        spawn = prev(SpaceInvadersR2Fixed.next_spawn_step)
        return cur == spawn
