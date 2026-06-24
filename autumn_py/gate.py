"""Round-kernel gate for the synthesis loop.

``gate(emit_cls, goals)`` checks each typed ``Goal`` against a
``@program``-decorated emit and returns the residuals (the goals it failed).
Dispatch is by ``isinstance`` (``_CHECKERS``); each checker returns
``Residual | None``. Goal shapes: ``FootprintExclude`` (read-set check),
``ModularArithmetic`` (bounded SMT), ``WriteFrame`` (write-set ⊆ allowed),
``TrajectoryInvariant`` (concrete walk). See
``drafts/autumn-pl-handlers-and-properties.md`` Part 1 D4.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable

import z3

from .api import transition_of
from .smt import (
    SMT_SUPPORTED_TYPES,
    read_set,
    solve_against_goal,
    unroll_transitions,
)


# --------------------------------------------------------------------------
# Goal hierarchy and Residual
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Goal:
    """Base class. Subclasses carry shape-specific fields."""
    anchor: str


@dataclass(frozen=True)
class FootprintExcludeGoal(Goal):
    """The lambda at ``anchor``'s read-set must not contain any atom
    matching a pattern in ``exclude``. Patterns may use ``Ellipsis`` as
    a wildcard, e.g. ``("get_var", Ellipsis, -1)`` matches any prev-tick
    variable read."""
    exclude: tuple[tuple, ...]


@dataclass(frozen=True)
class ModularArithmeticGoal(Goal):
    """Bounded-model SMT check.

    ``unroll`` lists the next-clause anchors whose transitions are
    instantiated across ``range(horizon)``; their accumulated constraints
    are conjoined with ``init_constraints(...)`` and checked against the
    conjunction of ``goal_factory(..., k)`` for ``k`` in ``range(horizon)``.
    Both are lambdas whose leading parameters are named after state vars
    (bound to their Z3 functions by ``_call_with_funcs``); ``goal_factory``
    takes the tick ``k`` last.
    """
    unroll: tuple[str, ...]
    init_constraints: Callable[..., list[z3.BoolRef]]
    goal_factory: Callable[..., z3.BoolRef]
    horizon: int = 6


@dataclass(frozen=True)
class TrajectoryInvariantGoal(Goal):
    """Concrete-trajectory invariant.

    Walks the program for ``steps`` ticks, snapshotting every state var.
    ``predicate`` is called per tick with each state-var name bound to a
    per-tick lookup and a final ``t`` (indices clamp to the recorded range)::

        lambda ant, food, t: dist(ant(t+1), food(t)) <= dist(ant(t), food(t))

    Returns a residual on the first failing ``t``; witness is
    ``{"t": t, "snapshot": snapshots[t]}``.
    """
    predicate: Callable[..., Any]
    steps: int = 10


@dataclass(frozen=True)
class WriteFrameGoal(Goal):
    """The decorated body's write-set must be ⊆ ``allowed_writes`` plus the
    state var it is the next-clause for (``foo.next`` may always write
    ``foo``) — catches a next-clause that accidentally writes a sibling var.
    Checked by running it under ``read_set`` and comparing the
    ``("set_var", name)`` atoms against ``allowed_writes ∪ {self_var}``.
    """
    allowed_writes: tuple[str, ...]


@dataclass
class Residual:
    """Failure record handed back to the synthesizer's next emit-proposal."""
    goal: Goal
    witness: Any = None

    def __repr__(self) -> str:
        kind = type(self.goal).__name__
        return f"Residual({kind} on {self.goal.anchor!r}, witness={self.witness!r})"


# --------------------------------------------------------------------------
# select_ast — anchor resolution
# --------------------------------------------------------------------------

def select_ast(emit_cls: type, anchor: str) -> Any:
    """Resolve an anchor to a subterm of `emit_cls`'s `_autumn_spec`.

    * ``"<state_var>.next"`` → ``api.transition_of(sv)``: the shared
      zero-arg transition that runs the clause and emits
      ``set_var(<state_var>, value)``. This is the *same* callable the
      runtime executes in its next-phase, so the gate analyses the literal
      term the runtime commits.
    * ``"<state_var>.init"`` → the StateVar's init value
    * ``"<state_var>"``      → the StateVar object itself
    """
    spec = getattr(emit_cls, "_autumn_spec", None)
    if spec is None:
        raise TypeError(f"{emit_cls.__name__} is not @program-decorated")

    if "." in anchor:
        head, rest = anchor.split(".", 1)
        for sv in spec.state_vars:
            if sv.name == head:
                if rest == "next":
                    return transition_of(sv)
                if rest == "init":
                    return sv.initial_value()
                raise ValueError(f"unknown anchor suffix {rest!r}")
        raise KeyError(f"no state var {head!r} on {emit_cls.__name__}")

    for sv in spec.state_vars:
        if sv.name == anchor:
            return sv
    raise KeyError(f"no state var {anchor!r} on {emit_cls.__name__}")


# --------------------------------------------------------------------------
# Spec extraction for SMT
# --------------------------------------------------------------------------

def _state_var_specs(emit_cls: type) -> dict[str, type]:
    """Build the SMT state-var specs from the program's spec.

    Includes only state vars whose declared type the SMT layer can lift
    (``SMT_SUPPORTED_TYPES`` — int/bool). State vars of other types
    aren't dropped silently — they're absent from the specs, so any
    `set_var` on them under SmtCollectHandler raises a clear
    ``ValueError`` naming the undeclared variable.
    """
    spec = emit_cls._autumn_spec
    return {
        sv.name: sv.type_
        for sv in spec.state_vars
        if sv.type_ in SMT_SUPPORTED_TYPES
    }


# --------------------------------------------------------------------------
# Atom matching for footprint checks
# --------------------------------------------------------------------------

def _atom_matches(atom: tuple, pattern: tuple) -> bool:
    """Tuple-shape match with ``Ellipsis`` as wildcard."""
    if len(atom) != len(pattern):
        return False
    return all(p is Ellipsis or a == p for a, p in zip(atom, pattern))


# --------------------------------------------------------------------------
# Per-shape checker functions — each returns Residual | None
# --------------------------------------------------------------------------

def _check_footprint_exclude(
    emit_cls: type, goal: FootprintExcludeGoal,
) -> Residual | None:
    target = select_ast(emit_cls, goal.anchor)
    atoms = read_set(target)
    violating = sorted(
        a for a in atoms if any(_atom_matches(a, p) for p in goal.exclude)
    )
    if violating:
        return Residual(goal=goal, witness=violating)
    return None


def _record_trajectory(emit_cls: type, steps: int) -> list[dict]:
    """Run ``emit_cls`` under a Runtime for ``steps`` ticks; return
    ``steps + 1`` snapshots — one before each tick, plus one after the
    final tick. Each snapshot is a plain ``dict`` of state-var name →
    value (post-init / post-tick state).

    The Runtime is closed before returning. Uses ``seed=42`` for
    reproducibility; trajectory_invariant goals are deterministic-walk
    by construction (a stochastic program with a trajectory invariant
    that flakes is a real residual the user wants to see).
    """
    from .runtime import Runtime
    snapshots: list[dict] = []
    with Runtime(emit_cls, seed=42) as rt:
        snapshots.append(dict(rt.state.freeze_snapshot()))
        for _ in range(steps):
            rt.step()
            snapshots.append(dict(rt.state.freeze_snapshot()))
    return snapshots


def _check_trajectory_invariant(
    emit_cls: type, goal: TrajectoryInvariantGoal,
) -> Residual | None:
    """Walk the recorded trajectory, evaluate ``goal.predicate`` at each
    tick boundary. Same param-introspection as ``_check_modular``: bare
    parameter names are bound to per-tick lookup functions, the trailing
    ``t`` parameter receives the tick index.

    Each lookup function ``f`` satisfies ``f(k) == snapshots[k][name]``,
    with ``k`` clamped to ``[0, len(snapshots) - 1]`` so out-of-range
    indices (``ant(t-1)`` at ``t=0``) return boundary values instead of
    raising. The lambda is responsible for guarding boundary cases if
    that matters to the property.
    """
    snapshots = _record_trajectory(emit_cls, goal.steps)
    n = len(snapshots)

    def make_lookup(name: str) -> Callable[[int], Any]:
        def f(k: int) -> Any:
            kk = max(0, min(n - 1, int(k)))
            return snapshots[kk][name]
        f.__name__ = f"{name}.trajectory_lookup"
        return f

    # Bind one lookup per state var declared in spec.
    spec = emit_cls._autumn_spec
    var_lookups = {sv.name: make_lookup(sv.name) for sv in spec.state_vars}

    sig = inspect.signature(goal.predicate)
    params = list(sig.parameters.keys())
    if not params or params[-1] != "t":
        raise TypeError(
            f"trajectory_invariant lambda must take a final ``t`` "
            f"parameter; got params {params!r}"
        )
    sv_params = params[:-1]
    bound: list[Any] = []
    for name in sv_params:
        if name not in var_lookups:
            raise NameError(
                f"trajectory_invariant lambda parameter {name!r} is not "
                f"a known state-var name. Available: {sorted(var_lookups)}"
            )
        bound.append(var_lookups[name])

    # Walk t over [0, steps - 1] — the (snapshot[t], snapshot[t+1]) pairs.
    for t in range(goal.steps):
        result = goal.predicate(*bound, t)
        if not result:
            return Residual(
                goal=goal,
                witness={"t": t, "snapshot": dict(snapshots[t])},
            )
    return None


def _check_write_frame(
    emit_cls: type, goal: WriteFrameGoal,
) -> Residual | None:
    """Verify the decorated body's write-set is ⊆ allowed_writes ∪
    {implicitly-decorated state var}.

    The implicit member: if anchor is ``"foo.next"``, then writes to
    ``foo`` are always allowed (the next-clause's job is to compute
    foo's new value). Other writes must be explicitly listed in
    ``allowed_writes`` or the goal fails.
    """
    target = select_ast(emit_cls, goal.anchor)
    atoms = read_set(target)
    actual_writes = {a[1] for a in atoms if a[0] == "set_var"}

    # Implicit allow: the state var the anchor refers to.
    head, _, _ = goal.anchor.partition(".")
    allowed = set(goal.allowed_writes) | {head}

    violating = sorted(actual_writes - allowed)
    if violating:
        return Residual(goal=goal, witness=violating)
    return None


def _check_modular(
    emit_cls: type, goal: ModularArithmeticGoal,
) -> Residual | None:
    specs = _state_var_specs(emit_cls)

    constraints: list[z3.BoolRef] = []
    funcs: dict[str, z3.FuncDeclRef] = {}
    for anchor in goal.unroll:
        target = select_ast(emit_cls, anchor)
        sub_cs, sub_funcs = unroll_transitions(target, specs, range(goal.horizon))
        constraints.extend(sub_cs)
        funcs.update(sub_funcs)

    constraints.extend(_call_with_funcs(goal.init_constraints, funcs))
    bounded_goal = z3.And(*[
        _call_with_funcs(goal.goal_factory, funcs, k)
        for k in range(goal.horizon)
    ])

    cex = solve_against_goal(constraints, bounded_goal)
    if cex is None:
        return None
    return Residual(goal=goal, witness=cex)


def _call_with_funcs(fn: Callable, funcs: dict, *trailing_args) -> Any:
    """Call ``fn`` with state-var Z3 functions bound by parameter name.

    The lambda's leading parameters are named after state vars
    (``lambda x, y, t: x(t+1) >= y(t)``); the gate looks each up in
    ``funcs`` and binds it. Any trailing positional ``*trailing_args``
    (e.g. the tick index) follow the bound functions.
    """
    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())
    sv_params = params[:len(params) - len(trailing_args)]
    bound = []
    for name in sv_params:
        if name not in funcs:
            raise NameError(
                f"invariant/init_constraints lambda parameter {name!r} "
                f"is not a known state-var name. Available: {sorted(funcs)}"
            )
        bound.append(funcs[name])
    return fn(*bound, *trailing_args)


# --------------------------------------------------------------------------
# Module-level dispatch — Goal subclass → checker
# --------------------------------------------------------------------------

_CHECKERS: dict[type, Callable[[type, Any], Residual | None]] = {
    FootprintExcludeGoal:    _check_footprint_exclude,
    ModularArithmeticGoal:   _check_modular,
    WriteFrameGoal:          _check_write_frame,
    TrajectoryInvariantGoal: _check_trajectory_invariant,
}


def gate(
    emit_cls: type,
    goals: list[Goal] | None = None,
) -> list[Residual]:
    """Run goals against `emit_cls`, returning the list of residuals.

    Goal sources, conjoined in this order:

    1. Goals registered on the program via ``@spec(...)``,
       ``@modifies(...)``, or ``@no_stochastic`` (read from
       ``emit_cls._autumn_spec.properties``).
    2. Goals passed in via the ``goals`` parameter (the original API,
       still supported for explicit / synthesized goals not declared
       on the program itself).

    Empty residual list ⇒ emit committed. Non-empty ⇒ rejected; each
    residual is the synthesizer's repair input for the next emit-
    proposal step.
    """
    spec = getattr(emit_cls, "_autumn_spec", None)
    program_goals = list(spec.properties) if spec is not None else []
    user_goals = list(goals) if goals is not None else []
    all_goals = program_goals + user_goals

    residuals: list[Residual] = []
    for goal in all_goals:
        checker = _CHECKERS.get(type(goal))
        if checker is None:
            raise NotImplementedError(
                f"no checker registered for {type(goal).__name__}; "
                f"library-conditional checkers (invariant/existential) "
                f"are not yet wired"
            )
        residual = checker(emit_cls, goal)
        if residual is not None:
            residuals.append(residual)
    return residuals
