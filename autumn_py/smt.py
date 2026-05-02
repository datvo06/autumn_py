"""Symbolic-constraint extraction for next-expressions.

The same evaluator as ground execution, run under a different handler
stack — `SmtCollectHandler` reinterprets the autumn op vocabulary in
Z3's symbolic-expression domain. Each StateVar `name` is encoded as a
Z3 function ``name : Int -> Sort``; the Int argument is the tick. A
read at the handler's symbolic tick `t` produces ``name(t)``; a write
via ``set_var(name, v)`` records the transition rule ``name(t+1) = v``.
At finalization, transition rules are universally quantified over t.

Free variables in the program — the StateVars themselves and the fresh
existentials introduced by ``sample_uniform`` — *are* the symbolic
variables. There is no parallel name-space; constraints are written
over Z3 expressions whose function declarations match the StateVar
names. The user's goal-writing API is just normal Python operators
(`+`, `==`, `%`, ...) on Z3 expressions, because Z3 overloads them all.

This is the architectural pattern from RoboTL's ``solve_init`` adapted
to a finite-domain SMT setting: a collection handler produces (free
variables, constraints), and a backend solver consumes them to return
either an Interpretation (model) or UNSAT.

Limitations
-----------
The handler does not interpret Python's native ``if`` on symbolic
values — Z3 expressions are not directly truthy. Use ``if_then_else``
(a `@defop` exported from this module) for symbolic conditionals; under
the SmtCollectHandler it lowers to ``z3.If``. Under ground execution it
behaves like a normal Python if/else.
"""
from __future__ import annotations

from typing import Any, Callable

import z3
from effectful.ops.semantics import handler
from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.ops.types import NotHandled

from .ops import (
    alloc_obj_id,
    get_click_pos,
    get_prev_var,
    get_var,
    if_then_else,
    is_event_active,
    sample_uniform,
    set_var,
)


# --------------------------------------------------------------------------
# Sort lifting
# --------------------------------------------------------------------------

_SORT_OF: dict[type, z3.SortRef] = {
    int: z3.IntSort(),
    bool: z3.BoolSort(),
}


def _z3_sort(py_type: type) -> z3.SortRef:
    if py_type in _SORT_OF:
        return _SORT_OF[py_type]
    raise TypeError(f"no SMT sort for Python type {py_type!r}")


# --------------------------------------------------------------------------
# SmtCollectHandler
# --------------------------------------------------------------------------

class SmtCollectHandler(ObjectInterpretation):
    """Reinterprets autumn ops as Z3 symbolic-expression construction.

    State vars are encoded as time-indexed functions; the handler's
    symbolic tick `t` is universally quantified at finalization. Free
    existentials (``sample_uniform``, ``alloc_obj_id``) are minted as
    fresh Z3 constants with their domain assertions accumulated.
    """

    def __init__(
        self,
        state_var_specs: dict[str, type],
        *,
        tick_name: str = "t",
        tick_value: int | z3.ExprRef | None = None,
        auto_declare: bool = False,
    ) -> None:
        """`tick_value=None` makes the handler's tick a fresh symbolic Z3
        Int (later universally quantified by `lifted_constraints()`).
        Pass a concrete int / Z3 expression to bind the tick to a value;
        each `set_var` then records an *instantiated* transition at that
        tick, no quantifier needed. Bounded model checking unrolls the
        recurrence by calling collect_smt repeatedly with `tick_value` in
        a range — useful when ForAll-over-Int problems are undecidable.

        `auto_declare`: when True, ``get_var``/``set_var``/``get_prev_var``
        on an undeclared state var register it as ``Int``-typed on the fly
        instead of raising. Used by ``read_set`` (footprint analysis only
        cares about atoms, not types). Off by default so SMT extraction
        complains loudly about typos.
        """
        self.specs = dict(state_var_specs)
        self.state_funcs: dict[str, z3.FuncDeclRef] = {
            name: z3.Function(name, z3.IntSort(), _z3_sort(ty))
            for name, ty in state_var_specs.items()
        }
        self.t = z3.Int(tick_name) if tick_value is None else tick_value
        self._symbolic_tick = tick_value is None
        self._auto_declare = auto_declare
        self.transitions: list[z3.BoolRef] = []
        self.globals: list[z3.BoolRef] = []
        self.existentials: list[z3.ExprRef] = []
        # Atom-set tracking: every op invocation appends the corresponding
        # syntactic atom. Used by `read_set` and the FootprintChecker.
        self.atoms: set[tuple] = set()
        self._fresh_idx = 0

    def _ensure_state_func(self, name: str, default_type: type = int) -> z3.FuncDeclRef:
        """Return the Z3 function for `name`, auto-declaring it if
        ``auto_declare`` was set. Raises if neither found nor auto-declared."""
        if name in self.state_funcs:
            return self.state_funcs[name]
        if self._auto_declare:
            f = z3.Function(name, z3.IntSort(), _z3_sort(default_type))
            self.state_funcs[name] = f
            self.specs[name] = default_type
            return f
        raise ValueError(
            f"set_var on undeclared state var {name!r}; "
            f"add it to state_var_specs (or pass auto_declare=True)"
        )

    # -- core state ops ---------------------------------------------------

    @implements(get_var)
    def _get_var(self, name: str):
        self.atoms.add(("get_var", name, 0))
        if name in self.state_funcs:
            return self.state_funcs[name](self.t)
        if self._auto_declare:
            return self._ensure_state_func(name)(self.t)
        return self._fresh_existential(name, int)

    @implements(get_prev_var)
    def _get_prev_var(self, name: str):
        self.atoms.add(("get_var", name, -1))
        if name in self.state_funcs:
            return self.state_funcs[name](self.t - 1)
        if self._auto_declare:
            return self._ensure_state_func(name)(self.t - 1)
        return self._fresh_existential(name, int)

    @implements(set_var)
    def _set_var(self, name: str, value: Any) -> None:
        self.atoms.add(("set_var", name))
        f = self._ensure_state_func(name)
        self.transitions.append(f(self.t + 1) == value)

    # -- stochasticity & identifiers --------------------------------------

    @implements(sample_uniform)
    def _sample(self, xs: Any):
        self.atoms.add(("sample_uniform",))
        e = z3.Int(f"rng_{self._fresh_idx}")
        self._fresh_idx += 1
        self.existentials.append(e)
        if isinstance(xs, range):
            self.globals.append(z3.And(e >= xs.start, e < xs.stop))
        elif isinstance(xs, (tuple, list)) and xs and all(
            isinstance(v, (int, bool)) for v in xs
        ):
            self.globals.append(z3.Or(*[e == v for v in xs]))
        return e

    @implements(alloc_obj_id)
    def _alloc(self):
        self.atoms.add(("alloc_obj_id",))
        return self._fresh_existential("obj_id", int)

    # -- environment & events --------------------------------------------

    @implements(is_event_active)
    def _event(self, name: str):
        self.atoms.add(("is_event_active", name))
        return z3.Bool(f"event_{name}_t")

    @implements(get_click_pos)
    def _click(self):
        self.atoms.add(("get_click_pos",))
        x = z3.Int("click_x_t")
        y = z3.Int("click_y_t")
        return (x, y)

    # -- symbolic conditional ---------------------------------------------

    @implements(if_then_else)
    def _ite(self, cond, then_b, else_b):
        return z3.If(cond, then_b, else_b)

    # -- helpers ----------------------------------------------------------

    def _fresh_existential(self, prefix: str, py_type: type) -> z3.ExprRef:
        sort = _z3_sort(py_type)
        if sort == z3.IntSort():
            e = z3.Int(f"{prefix}_{self._fresh_idx}")
        elif sort == z3.BoolSort():
            e = z3.Bool(f"{prefix}_{self._fresh_idx}")
        else:
            e = z3.Const(f"{prefix}_{self._fresh_idx}", sort)
        self._fresh_idx += 1
        self.existentials.append(e)
        return e

    # -- finalization ------------------------------------------------------

    def init_constraints(self) -> list[z3.BoolRef]:
        """Constraints from a single-shot run (no universal quantification
        of the tick). Useful when the emit is straight-line `__init__`
        code that runs once at tick 0."""
        return list(self.globals) + list(self.transitions)

    def lifted_constraints(self) -> list[z3.BoolRef]:
        """Constraints with the per-tick transition rules universally
        quantified over t. This is the form the SMT-LIB block in
        `drafts/autumn-pl-handlers-and-properties.md` Part 1 D4 quotes.

        If the handler was constructed with a concrete `tick_value`, the
        transitions are already instantiated and quantification is a
        no-op — same as `init_constraints()`."""
        if not self._symbolic_tick:
            return self.init_constraints()
        out = list(self.globals)
        for c in self.transitions:
            out.append(z3.ForAll([self.t], c))
        return out


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def read_set(
    fn: Callable[[], Any],
    state_var_specs: dict[str, type] | None = None,
) -> frozenset[tuple]:
    """Compute the read-set (set of syntactic atoms) of a callable.

    Atoms are tuples of the form ``(op_name, *args)``:

    * ``("get_var", name, 0)`` — read of state var ``name`` at the current tick.
    * ``("get_var", name, -1)`` — read of state var ``name`` at the previous tick.
    * ``("set_var", name)`` — write to state var ``name``.
    * ``("sample_uniform",)`` — stochastic draw.
    * ``("is_event_active", event_name)`` — event-activation read.

    Implementation: run ``fn`` under ``SmtCollectHandler`` and harvest its
    ``atoms`` accumulator. Side-effecting ops still append their atoms even
    if the resulting Z3 expression is discarded, so this function gives a
    sound over-approximation of the lambda's syntactic dependencies.

    `state_var_specs` defaults to an empty dict — undeclared names get
    fresh existentials, which is fine for footprint analysis since we only
    care about which atoms appear, not their types.
    """
    h = SmtCollectHandler(state_var_specs or {}, auto_declare=True)
    with handler(h):
        try:
            fn()
        except (NotHandled, z3.Z3Exception):
            # The atom was already recorded before the exception fired.
            # Two known-acceptable exception classes:
            #   * NotHandled — an op fired with no concrete-domain handler.
            #   * Z3Exception — typically "Symbolic expressions cannot be
            #     cast to concrete Boolean values," raised when the lambda
            #     uses native Python `if` / `bool()` on a Z3 expression.
            #     The lambda should have used @symbolic / if_then_else for
            #     conditionals; for footprint analysis we only care about
            #     atoms recorded before the failure.
            pass
    return frozenset(h.atoms)


def collect_smt(
    fn: Callable[[], Any],
    state_var_specs: dict[str, type],
    *,
    lifted: bool = True,
    tick_value: int | z3.ExprRef | None = None,
) -> tuple[Any, list[z3.BoolRef], dict[str, z3.FuncDeclRef]]:
    """Run `fn` under `SmtCollectHandler`.

    Returns ``(result, constraints, state_funcs)``:

    - ``result``: whatever `fn` returns (a Z3 expression, a Python value,
      or a tuple of either).
    - ``constraints``: list of Z3 ``BoolRef`` constraints. If ``lifted``
      and `tick_value is None`, transition rules are universally
      quantified over the symbolic tick. If `tick_value` is provided,
      transitions are instantiated at that tick.
    - ``state_funcs``: mapping of state-var names to their Z3 function
      declarations, so callers can reference them when writing goals.
    """
    h = SmtCollectHandler(state_var_specs, tick_value=tick_value)
    with handler(h):
        result = fn()
    constraints = h.lifted_constraints() if lifted else h.init_constraints()
    return result, constraints, dict(h.state_funcs)


def unroll_transitions(
    fn: Callable[[], Any],
    state_var_specs: dict[str, type],
    tick_range: range,
) -> tuple[list[z3.BoolRef], dict[str, z3.FuncDeclRef]]:
    """Bounded-model-checking helper: unroll `fn`'s transitions for each
    tick in `tick_range`, accumulating instantiated constraints.

    Returns ``(constraints, state_funcs)``. The function declarations
    are shared across the unroll (state_funcs[name](k) refers to tick k)
    so callers can write goals that quantify or condition over specific
    ticks within the bound. Use this when a ForAll-over-Int formulation
    is undecidable (e.g., recurrences over modular arithmetic).
    """
    state_funcs: dict[str, z3.FuncDeclRef] = {}
    all_constraints: list[z3.BoolRef] = []
    for k in tick_range:
        _, cs, funcs = collect_smt(
            fn, state_var_specs, tick_value=k,
        )
        if not state_funcs:
            state_funcs = funcs
        all_constraints.extend(cs)
    return all_constraints, state_funcs


def solve_against_goal(
    constraints: list[z3.BoolRef],
    goal: z3.BoolRef,
) -> dict[str, Any] | None:
    """Check whether `goal` is entailed by `constraints`.

    Returns ``None`` if entailed (UNSAT on the negation; the goal holds
    under all models of the constraints). Returns a counterexample
    model — a dict mapping declared variable names to Z3 values — if
    not entailed (SAT model exists where the goal fails).
    """
    s = z3.Solver()
    for c in constraints:
        s.add(c)
    s.add(z3.Not(goal))
    if s.check() == z3.sat:
        m = s.model()
        return {str(d.name()): m[d] for d in m.decls()}
    return None


def to_smt_lib(
    constraints: list[z3.BoolRef],
    goal: z3.BoolRef | None = None,
) -> str:
    """Format constraints (and optionally the negation of `goal`) as an
    SMT-LIB v2 problem string. The output is whatever Z3 produces from
    its built-in serializer — the layout matches Z3's SMT-LIB output, not
    the doc's hand-written figure verbatim, but the *content* is the same
    constraint system."""
    s = z3.Solver()
    for c in constraints:
        s.add(c)
    if goal is not None:
        s.add(z3.Not(goal))
    return s.to_smt2()


