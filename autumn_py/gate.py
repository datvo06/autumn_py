"""Round-kernel gate for the synthesis loop.

The gate consumes (1) a `@program`-decorated emit class and (2) a list
of typed `Goal` subclass instances, and returns the list of residuals
naming the goals the emit failed.

Concrete instantiation of `drafts/autumn-pl-handlers-and-properties.md`
Part 1 D4. Each goal subclass carries shape-specific fields; dispatch
is by `isinstance`. Each checker returns ``Residual | None`` —
``None`` means the goal passed.

Currently implemented goal shapes:

* :class:`FootprintExcludeGoal` / :class:`FootprintIncludeGoal` —
  syntactic-dependency checks against the read-set (P_1, P_4, P_6,
  P_8, P_12 in the doc).
* :class:`ModularArithmeticGoal` — bounded-model SMT check via
  ``autumn_py.smt.collect_smt`` and ``solve_against_goal`` (P_3, P_11).

Library-conditional shapes (invariant, existential) wait on the typed
goal/idiom library described in §2.5 of the draft.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import z3

from .ops import set_var
from .smt import collect_smt, read_set, solve_against_goal


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
class FootprintIncludeGoal(Goal):
    """The lambda at ``anchor``'s read-set must contain at least one
    atom matching each pattern in ``include``."""
    include: tuple[tuple, ...]


@dataclass(frozen=True)
class ModularArithmeticGoal(Goal):
    """Bounded-model SMT check.

    ``unroll`` lists the next-clause anchors whose transitions are
    instantiated across ``range(horizon)``; their accumulated
    constraints are conjoined with ``init_constraints(funcs)`` and
    checked against the conjunction of ``goal_factory(funcs, k)`` for
    ``k`` in ``range(horizon)``.
    """
    unroll: tuple[str, ...]
    init_constraints: Callable[[dict], list[z3.BoolRef]]
    goal_factory: Callable[[dict, int], z3.BoolRef]
    horizon: int = 6


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

    * ``"<state_var>.next"`` → a transition wrapper around the registered
      `next`-clause callable: a zero-arg function that runs the clause
      and emits ``set_var(<state_var>, value)`` with the return value.
      Matches the runtime's next-clause-to-state-var commit so handlers
      see the transition.
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
                    if sv._next_fn is None:
                        raise ValueError(f"state var {head!r} has no next clause")
                    return _wrap_as_transition(sv._next_fn, sv.name)
                if rest == "init":
                    return sv.initial_value()
                raise ValueError(f"unknown anchor suffix {rest!r}")
        raise KeyError(f"no state var {head!r} on {emit_cls.__name__}")

    for sv in spec.state_vars:
        if sv.name == anchor:
            return sv
    raise KeyError(f"no state var {anchor!r} on {emit_cls.__name__}")


def _wrap_as_transition(next_fn: Callable, var_name: str) -> Callable:
    """Convert a return-style next-clause into a set_var-style transition,
    matching the runtime's next-clause-to-commit semantics."""
    def transition() -> None:
        value = next_fn()
        set_var(var_name, value)
    transition.__name__ = f"{var_name}_next_transition"
    return transition


# --------------------------------------------------------------------------
# Spec extraction for SMT
# --------------------------------------------------------------------------

_SMT_SUPPORTED_TYPES: frozenset[type] = frozenset({int, bool})


def _state_var_specs(emit_cls: type) -> dict[str, type]:
    """Build the SMT state-var specs from the program's spec.

    Includes only state vars whose declared type is in
    ``_SMT_SUPPORTED_TYPES`` (int/bool). State vars of other types
    aren't dropped silently — they're absent from the specs, so any
    `set_var` on them under SmtCollectHandler raises a clear
    ``ValueError`` naming the undeclared variable.
    """
    spec = emit_cls._autumn_spec
    return {
        sv.name: sv.type_
        for sv in spec.state_vars
        if sv.type_ in _SMT_SUPPORTED_TYPES
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


def _check_footprint_include(
    emit_cls: type, goal: FootprintIncludeGoal,
) -> Residual | None:
    target = select_ast(emit_cls, goal.anchor)
    atoms = read_set(target)
    missing = sorted(
        p for p in goal.include if not any(_atom_matches(a, p) for a in atoms)
    )
    if missing:
        return Residual(goal=goal, witness=missing)
    return None


def _check_modular(
    emit_cls: type, goal: ModularArithmeticGoal,
) -> Residual | None:
    specs = _state_var_specs(emit_cls)

    constraints: list[z3.BoolRef] = []
    funcs: dict[str, z3.FuncDeclRef] = {}
    for anchor in goal.unroll:
        target = select_ast(emit_cls, anchor)
        for k in range(goal.horizon):
            _, sub_cs, sub_funcs = collect_smt(target, specs, tick_value=k)
            constraints.extend(sub_cs)
            funcs.update(sub_funcs)

    constraints.extend(goal.init_constraints(funcs))
    bounded_goal = z3.And(*[goal.goal_factory(funcs, k) for k in range(goal.horizon)])

    cex = solve_against_goal(constraints, bounded_goal)
    if cex is None:
        return None
    return Residual(goal=goal, witness=cex)


# --------------------------------------------------------------------------
# Module-level dispatch — Goal subclass → checker
# --------------------------------------------------------------------------

_CHECKERS: dict[type, Callable[[type, Any], Residual | None]] = {
    FootprintExcludeGoal: _check_footprint_exclude,
    FootprintIncludeGoal: _check_footprint_include,
    ModularArithmeticGoal: _check_modular,
}


def gate(emit_cls: type, goals: list[Goal]) -> list[Residual]:
    """Run every goal against `emit_cls`, returning the list of residuals.

    Empty list ⇒ emit committed. Non-empty ⇒ rejected; each residual is
    the synthesizer's repair input for the next emit-proposal step.
    """
    residuals: list[Residual] = []
    for goal in goals:
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
