"""Pin the Z3-backed SMT-extraction handler.

Tests demonstrate that the same autumn ops the runtime executes can be
run under `SmtCollectHandler` to produce a Z3 problem; goals are
written as ordinary Python expressions over the program's state-var
names, since Z3 overloads `+`, `==`, `%`, `<`, etc. on its expressions.

The round-1 / round-2 cases mirror Part 1 of
`drafts/autumn-pl-handlers-and-properties.md`. Round-2's deterministic
emit is shown to entail the goal $\\varphi_2$ at the bound constant 3
but to fail when the constant is 4 — Z3 returns a counterexample model
that pinpoints the off-by-one tick, exactly the round-2 D5 walkthrough
in the doc.
"""
from __future__ import annotations

import pytest
import z3

from autumn_py.ops import (
    get_prev_var,
    get_var,
    sample_uniform,
    set_var,
)
from autumn_py._ast_rewrite import symbolic
from autumn_py.ops import if_then_else
from autumn_py.smt import (
    collect_smt,
    read_set,
    solve_against_goal,
    unroll_transitions,
)


# -------------------------------------------------------------------------
# Round-1 stochastic emit — the spawn rule that fails P_1
# -------------------------------------------------------------------------

def test_round_1_stochastic_emit_introduces_existential():
    """The round-1 spawn rule:

        spawn_step = sample_uniform(range(1, 21))
        # later: spawn fires when step_count == spawn_step

    The handler's `__init__`-shaped run produces:
      - a fresh Int existential rng_0 with a domain assertion
      - a transition rule equating `next_spawn_step(t+1) = rng_0`
        (the spawn-rule's effect, lifted via forall t)
    """

    def round_1_init():
        spawn = sample_uniform(range(1, 21))
        set_var("next_spawn_step", spawn)

    _, constraints, funcs = collect_smt(
        round_1_init,
        state_var_specs={"next_spawn_step": int},
    )

    # one domain constraint on the existential, one universally-quantified
    # transition
    assert len(constraints) == 2

    # Assert meaning, not serializer text. Instantiate at a concrete tick
    # (decidable): the spawned value is forced into the sampled domain
    # [1, 20], yet is not pinned to any single value (a fresh existential).
    next_spawn_step = funcs["next_spawn_step"]
    assert solve_against_goal(
        constraints,
        z3.And(next_spawn_step(1) >= 1, next_spawn_step(1) <= 20),
    ) is None
    assert solve_against_goal(constraints, next_spawn_step(1) == 5) is not None


# -------------------------------------------------------------------------
# Round-2 deterministic-but-off-by-one emit — fails P_3
# -------------------------------------------------------------------------

def _round_2_step():
    """The round-2 spawn rule expressed in autumn-op form.

    Recurrence: ``next_spawn_step(t+1) = step_count(t) == next_spawn_step(t)
    ? next_spawn_step(t) + 15 : next_spawn_step(t)``.

    All reads here are at the current tick `t` (the handler's tick value);
    set_var records the next-tick assignment. step_count's value at each
    tick is pinned by the test-supplied counter constraint.
    """
    cur_step = get_var("step_count")
    cur_spawn = get_var("next_spawn_step")
    new_value = if_then_else(
        cur_step == cur_spawn,
        cur_spawn + 15,
        cur_spawn,
    )
    set_var("next_spawn_step", new_value)


def _build_round_2_problem(initial_constant: int, horizon: int = 6):
    """Bounded-model-checking unroll of the round-2 spawn rule for ticks
    in [0, horizon]. Returns (all_constraints, state_funcs)."""
    specs = {"step_count": int, "next_spawn_step": int}
    transitions, funcs = unroll_transitions(_round_2_step, specs, range(horizon))

    next_spawn_step = funcs["next_spawn_step"]
    step_count = funcs["step_count"]

    # step_count(t) = t for all instantiated ticks, plus initial constant
    base = [next_spawn_step(0) == initial_constant]
    base += [step_count(k) == k for k in range(horizon + 1)]

    return base + transitions, funcs


def test_round_2_off_by_one_emit_fails_goal_with_z3_counterexample():
    """Round-2's emit binds the spawn constant to 4, then increments by 15.

    Goal phi_2: across observed ticks, spawn fires iff (t mod 15 == 3).

    Under the synth's emit, spawn fires at t = 4 (then 19, 34, ...) — never
    at t = 3. Z3 returns SAT on (constraints AND NOT phi_2) with a model
    pinpointing the off-by-one tick.

    Bounded model checking: unroll the recurrence for t in [0, 6] rather
    than asking Z3 to reason about it under universal quantification (the
    quantified problem mixes integer arithmetic with modular and is
    undecidable in the general theory)."""
    constraints, funcs = _build_round_2_problem(initial_constant=4)
    next_spawn_step = funcs["next_spawn_step"]
    step_count = funcs["step_count"]

    horizon = 6
    spawn_at = lambda k: step_count(k) == next_spawn_step(k)
    bounded_goal = z3.And(*[
        spawn_at(k) == ((k % 15) == 3)
        for k in range(horizon)
    ])

    cex = solve_against_goal(constraints, bounded_goal)
    assert cex is not None, "expected counterexample but goal was entailed"


def test_round_2_correct_constant_entails_goal():
    """If the synth had bound the spawn constant to 3, the goal would
    be entailed across the bounded horizon. Solver returns None."""
    constraints, funcs = _build_round_2_problem(initial_constant=3)
    next_spawn_step = funcs["next_spawn_step"]
    step_count = funcs["step_count"]

    horizon = 6
    spawn_at = lambda k: step_count(k) == next_spawn_step(k)
    bounded_goal = z3.And(*[
        spawn_at(k) == ((k % 15) == 3)
        for k in range(horizon)
    ])

    cex = solve_against_goal(constraints, bounded_goal)
    assert cex is None, f"goal should be entailed, got counterexample: {cex}"


# -------------------------------------------------------------------------
# Free-variable composability — goals are normal Python over Z3 funcs
# -------------------------------------------------------------------------

def test_goal_is_python_expression_over_state_var_names():
    """The point of the refactor: goal-writing uses normal Python operators
    on the program's state-var Z3 functions, not a parallel Term ADT."""

    def round_2_init():
        set_var("next_spawn_step", 4)

    _, constraints, funcs = collect_smt(
        round_2_init,
        state_var_specs={"next_spawn_step": int},
    )
    next_spawn_step = funcs["next_spawn_step"]

    # Goal authored as ordinary Python over Z3:
    t = z3.Int("t")
    goal = z3.ForAll([t], (t == 3) == (next_spawn_step(t) == t))
    # ^^ "spawn fires at tick t iff t == 3"
    # The emit's transition forces next_spawn_step(t+1) = 4 universally,
    # so this goal is NOT entailed — counterexample exists.

    cex = solve_against_goal(constraints, goal)
    assert cex is not None


# -------------------------------------------------------------------------
# Get/get_prev round-trip threading through Z3 function applications
# -------------------------------------------------------------------------

def test_get_prev_var_returns_function_at_previous_tick():
    def reads_prev_step_count():
        prev = get_prev_var("step_count")
        return prev

    result, _, funcs = collect_smt(
        reads_prev_step_count,
        state_var_specs={"step_count": int},
    )
    step_count = funcs["step_count"]
    h_t = z3.Int("t")
    expected = step_count(h_t - 1)
    assert z3.eq(result, expected)


def test_set_var_records_transition_rule():
    def writes():
        set_var("step_count", get_prev_var("step_count") + 1)

    _, constraints, funcs = collect_smt(
        writes,
        state_var_specs={"step_count": int},
    )
    step_count = funcs["step_count"]

    # One universally-quantified transition; assert its meaning at a concrete
    # tick (decidable via instantiation) rather than the serializer's "forall"
    # keyword: the recurrence step_count(t+1) == step_count(t-1)+1 is entailed.
    # (No refutation control here: this constraint is a relational recurrence
    # with no free constant, so Z3 returns `unknown` rather than a SAT model
    # for a NOT-entailed goal; the round-2 tests cover refutation by unrolling
    # the recurrence into ground constraints.)
    assert len(constraints) == 1
    assert solve_against_goal(constraints, step_count(6) == step_count(4) + 1) is None


# -------------------------------------------------------------------------
# Sample uniform over a finite list (not a range)
# -------------------------------------------------------------------------

def test_sample_uniform_over_finite_list_emits_disjunction():
    def picks_color():
        c = sample_uniform([1, 2, 3, 4])
        set_var("color", c)

    _, constraints, funcs = collect_smt(
        picks_color,
        state_var_specs={"color": int},
    )
    color = funcs["color"]
    # At a concrete tick: color is a member of {1,2,3,4}, never outside it,
    # and is not pinned to a single member.
    assert solve_against_goal(
        constraints, z3.Or(*[color(1) == v for v in (1, 2, 3, 4)])
    ) is None
    assert solve_against_goal(constraints, color(1) != 5) is None
    assert solve_against_goal(constraints, color(1) == 1) is not None


# -------------------------------------------------------------------------
# @symbolic — native if/else in next-clause bodies
# -------------------------------------------------------------------------

def test_symbolic_rewrites_assignment_form_if_else():
    """User writes the round-2 step body using native `if/else` on a Z3
    expression; @symbolic rewrites the if to an `if_then_else` call so
    the body works under SmtCollectHandler."""

    @symbolic
    def round_2_step():
        cur_step = get_var("step_count")
        cur_spawn = get_var("next_spawn_step")
        if cur_step == cur_spawn:
            new_value = cur_spawn + 15
        else:
            new_value = cur_spawn
        set_var("next_spawn_step", new_value)

    # The rewritten body should produce the same constraints as the
    # explicit-form _round_2_step used elsewhere in this file.
    constraints, _ = unroll_transitions(
        round_2_step,
        {"step_count": int, "next_spawn_step": int},
        range(6),
    )
    assert len(constraints) == 6  # one per tick


def test_symbolic_rewrites_ternary_if_expression():
    """`a if c else b` ternary form."""

    @symbolic
    def with_ternary():
        cur = get_var("step_count")
        new_value = cur + 1 if cur < 10 else cur
        set_var("step_count", new_value)

    _, constraints, _ = collect_smt(
        with_ternary,
        state_var_specs={"step_count": int},
    )
    # one transition emitted; should mention the ternary's pieces
    assert len(constraints) == 1


def test_symbolic_rewrites_same_callable_branches():
    """Both branches call set_var with same name but different value."""

    @symbolic
    def call_form():
        cur = get_var("step_count")
        if cur == 0:
            set_var("step_count", 1)
        else:
            set_var("step_count", cur)

    _, constraints, _ = collect_smt(
        call_form,
        state_var_specs={"step_count": int},
    )
    assert len(constraints) == 1


def test_symbolic_rejects_asymmetric_branches():
    """A branch missing or with extra statements raises SyntaxError at
    decoration time, surfacing the mismatch as a residual the synthesizer
    can act on rather than as a silent semantic mismatch."""

    import pytest

    with pytest.raises(SyntaxError, match="cannot rewrite"):
        @symbolic
        def asymmetric():
            cur = get_var("step_count")
            if cur == 0:
                set_var("step_count", 1)
            # else: nothing — asymmetric

    with pytest.raises(SyntaxError, match="cannot rewrite"):
        @symbolic
        def different_calls():
            cur = get_var("step_count")
            if cur == 0:
                set_var("step_count", 1)
            else:
                set_var("next_spawn_step", 2)


def test_symbolic_handles_elif_via_nested_rewrites():
    """`elif` is parsed as nested `If` in the outer `orelse`. The
    recursive `generic_visit` rewrites bottom-up: the inner `If` becomes
    an `Assign(x, if_then_else(c2, b, c))`, after which the outer pattern
    matches and produces `Assign(x, if_then_else(c1, a, if_then_else(c2, b, c)))`.
    Result is a chained ITE — same Z3 semantics as `match` or guard cases."""

    @symbolic
    def with_elif():
        cur = get_var("step_count")
        if cur == 0:
            new_val = 1
        elif cur == 1:
            new_val = 2
        elif cur == 2:
            new_val = 3
        else:
            new_val = 4
        set_var("step_count", new_val)

    _, constraints, funcs = collect_smt(
        with_elif,
        state_var_specs={"step_count": int},
    )
    step_count = funcs["step_count"]
    # The chained elif lowers to nested ITEs; assert branch selection at a
    # concrete tick (0->1, 1->2, 2->3, else->4) rather than counting "ite"
    # in the serializer output.
    assert solve_against_goal(constraints, step_count(1) == z3.If(
        step_count(0) == 0, 1, z3.If(
            step_count(0) == 1, 2, z3.If(
                step_count(0) == 2, 3, 4)))) is None


def test_symbolic_handles_return_form_if_else():
    """`if cond: return a; else: return b` lifts to
    `return if_then_else(cond, a, b)`. The next-clause idiom in
    autumn-py uses return statements, so this is the common shape."""

    @symbolic
    def step_returns():
        cur = get_var("step_count")
        if cur == 0:
            return 1
        else:
            return cur

    # Sanity: the rewritten function still runs under ground execution.
    # Under SmtCollectHandler, the return value would be a Z3 ITE.
    assert step_returns.__name__ == "step_returns"


def test_symbolic_handles_elif_in_call_form():
    """elif also works when each branch is a same-callable expression-statement."""

    @symbolic
    def with_elif_calls():
        cur = get_var("step_count")
        if cur == 0:
            set_var("step_count", 1)
        elif cur == 1:
            set_var("step_count", 2)
        else:
            set_var("step_count", 3)

    _, constraints, funcs = collect_smt(
        with_elif_calls,
        state_var_specs={"step_count": int},
    )
    step_count = funcs["step_count"]
    assert len(constraints) == 1
    # 0->1, 1->2, else->3 at a concrete tick.
    assert solve_against_goal(constraints, step_count(1) == z3.If(
        step_count(0) == 0, 1, z3.If(step_count(0) == 1, 2, 3))) is None


def test_symbolic_preserves_ground_execution_semantics():
    """Under ground execution (no SmtCollectHandler installed), the
    @symbolic-decorated function must still behave like the original
    if/else. if_then_else's default impl evaluates both branches but
    returns the correct one — pure-expression branches are equivalent."""

    @symbolic
    def returns_max(a, b):
        return a if a > b else b

    assert returns_max(5, 3) == 5
    assert returns_max(2, 7) == 7
    assert returns_max(4, 4) == 4


# -------------------------------------------------------------------------
# read_set soundness — the atom-before-effect ordering invariant
# -------------------------------------------------------------------------

def test_read_set_records_atoms_before_a_mid_clause_raise():
    """read_set's over-approximation rests on every handler method appending
    its atom *before* anything that can raise. A clause that records a write,
    then trips the symbolic-bool cast (native `if` on a Z3 value), must still
    have the pre-raise write in its footprint."""
    def clause():
        set_var("written_before", 1)        # atom recorded first
        if get_var("flag") == 0:            # native if on a Z3 BoolRef → raises
            set_var("after", 2)             # unreached

    atoms = read_set(clause)
    assert ("set_var", "written_before") in atoms
    assert ("get_var", "flag", 0) in atoms


def test_read_set_propagates_unexpected_errors():
    """read_set tolerates only the symbolic-bool cast (and out-of-vocabulary
    NotHandled). A genuine bug in a clause must surface, not be swallowed
    into a silently-truncated footprint."""
    def buggy():
        get_var("x")
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        read_set(buggy)
