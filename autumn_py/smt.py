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

import ast
import inspect
import textwrap
from typing import Any, Callable

import z3
from effectful.ops.semantics import handler
from effectful.ops.syntax import ObjectInterpretation, defop, implements
from effectful.ops.types import NotHandled

from .ops import (
    alloc_obj_id,
    get_click_pos,
    get_prev_var,
    get_var,
    is_event_active,
    sample_uniform,
    set_var,
)


# --------------------------------------------------------------------------
# Symbolic conditional — the one op we add for symbolic-mode authoring
# --------------------------------------------------------------------------

@defop
def if_then_else(cond, then_branch, else_branch):
    """Three-way op for symbolic conditionals.

    Under ground execution: behaves like Python ``then_branch if cond else else_branch``.
    Under ``SmtCollectHandler``: lowers to ``z3.If(cond, then_branch, else_branch)``.
    """
    if cond:
        return then_branch
    return else_branch


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
    ) -> None:
        """`tick_value=None` makes the handler's tick a fresh symbolic Z3
        Int (later universally quantified by `lifted_constraints()`).
        Pass a concrete int / Z3 expression to bind the tick to a value;
        each `set_var` then records an *instantiated* transition at that
        tick, no quantifier needed. Bounded model checking unrolls the
        recurrence by calling collect_smt repeatedly with `tick_value` in
        a range — useful when ForAll-over-Int problems are undecidable."""
        self.specs = dict(state_var_specs)
        self.state_funcs: dict[str, z3.FuncDeclRef] = {
            name: z3.Function(name, z3.IntSort(), _z3_sort(ty))
            for name, ty in state_var_specs.items()
        }
        self.t = z3.Int(tick_name) if tick_value is None else tick_value
        self._symbolic_tick = tick_value is None
        self.transitions: list[z3.BoolRef] = []
        self.globals: list[z3.BoolRef] = []
        self.existentials: list[z3.ExprRef] = []
        self._fresh_idx = 0

    # -- core state ops ---------------------------------------------------

    @implements(get_var)
    def _get_var(self, name: str):
        if name in self.state_funcs:
            return self.state_funcs[name](self.t)
        return self._fresh_existential(name, int)

    @implements(get_prev_var)
    def _get_prev_var(self, name: str):
        if name in self.state_funcs:
            return self.state_funcs[name](self.t - 1)
        return self._fresh_existential(name, int)

    @implements(set_var)
    def _set_var(self, name: str, value: Any) -> None:
        if name not in self.state_funcs:
            raise ValueError(
                f"set_var on undeclared state var {name!r}; "
                f"add it to state_var_specs"
            )
        f = self.state_funcs[name]
        self.transitions.append(f(self.t + 1) == value)

    # -- stochasticity & identifiers --------------------------------------

    @implements(sample_uniform)
    def _sample(self, xs: Any):
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
        return self._fresh_existential("obj_id", int)

    # -- environment & events --------------------------------------------

    @implements(is_event_active)
    def _event(self, name: str):
        return z3.Bool(f"event_{name}_t")

    @implements(get_click_pos)
    def _click(self):
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


# --------------------------------------------------------------------------
# @symbolic — AST source-rewriting decorator for native if/else
# --------------------------------------------------------------------------

class _SymbolicIfRewriter(ast.NodeTransformer):
    """Rewrites Python's `if/else` statements and `a if c else b` ternaries
    into explicit `if_then_else` calls so the resulting function works
    under `SmtCollectHandler`.

    Supported patterns (raises `SyntaxError` otherwise):

    * **Ternary `IfExp`**: ``x = a if cond else b`` →
      ``x = if_then_else(cond, a, b)``.
    * **Assignment-form `If`**: ``if cond: x = a; else: x = b`` →
      ``x = if_then_else(cond, a, b)``. Both branches must assign to the
      same target.
    * **Same-callable `If`**: ``if cond: f(args1); else: f(args2)`` →
      ``f(args')`` where each differing argument position is wrapped in
      ``if_then_else(cond, args1[i], args2[i])``. Function callable and
      arity must match.

    Asymmetric branches, early returns inside branches, and structurally
    different bodies are unsupported and raise a clear `SyntaxError` with
    the offending line number.
    """

    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
        self.generic_visit(node)
        return ast.Call(
            func=ast.Name(id="if_then_else", ctx=ast.Load()),
            args=[node.test, node.body, node.orelse],
            keywords=[],
        )

    def visit_If(self, node: ast.If) -> ast.AST:
        self.generic_visit(node)
        if len(node.body) != 1 or len(node.orelse) != 1:
            self._fail(node, "both branches must be a single statement")
        b, o = node.body[0], node.orelse[0]
        if isinstance(b, ast.Assign) and isinstance(o, ast.Assign):
            if ast.dump(b.targets[0]) != ast.dump(o.targets[0]) or len(b.targets) != 1 or len(o.targets) != 1:
                self._fail(node, "both branches must assign to the same single target")
            return ast.Assign(
                targets=[b.targets[0]],
                value=ast.Call(
                    func=ast.Name(id="if_then_else", ctx=ast.Load()),
                    args=[node.test, b.value, o.value],
                    keywords=[],
                ),
            )
        if isinstance(b, ast.Expr) and isinstance(o, ast.Expr) and \
                isinstance(b.value, ast.Call) and isinstance(o.value, ast.Call):
            bc, oc = b.value, o.value
            if ast.dump(bc.func) != ast.dump(oc.func):
                self._fail(node, "branches call different callables")
            if len(bc.args) != len(oc.args):
                self._fail(node, "branches call with different arity")
            if [ast.dump(k) for k in bc.keywords] != [ast.dump(k) for k in oc.keywords]:
                self._fail(node, "branches call with different keyword arguments")
            new_args = []
            for i, (ba, oa) in enumerate(zip(bc.args, oc.args)):
                if ast.dump(ba) == ast.dump(oa):
                    new_args.append(ba)
                else:
                    # Differing string-literal args usually mean the two
                    # branches name *different state variables*, which
                    # cannot be made conditional sensibly; reject.
                    if isinstance(ba, ast.Constant) and isinstance(ba.value, str):
                        self._fail(node, f"argument {i} is a differing string "
                                          f"literal — cannot lift to if_then_else")
                    if isinstance(oa, ast.Constant) and isinstance(oa.value, str):
                        self._fail(node, f"argument {i} is a differing string "
                                          f"literal — cannot lift to if_then_else")
                    new_args.append(ast.Call(
                        func=ast.Name(id="if_then_else", ctx=ast.Load()),
                        args=[node.test, ba, oa],
                        keywords=[],
                    ))
            return ast.Expr(value=ast.Call(
                func=bc.func, args=new_args, keywords=bc.keywords,
            ))
        self._fail(node, "branches must both be assignments to the same target, "
                          "or both expression-statements calling the same function")

    @staticmethod
    def _fail(node: ast.If, reason: str) -> None:
        raise SyntaxError(
            f"@symbolic cannot rewrite this if/else (line {node.lineno}): {reason}. "
            f"Supported patterns: (a) `if cond: x = e1; else: x = e2`, "
            f"(b) `if cond: f(...); else: f(...)` with same callable and arity, "
            f"(c) `x = e1 if cond else e2`. "
            f"For other cases, use `if_then_else(cond, e1, e2)` explicitly."
        )


def symbolic(fn: Callable) -> Callable:
    """Decorator that rewrites `if/else` and `a if c else b` in `fn`'s body
    into explicit `if_then_else` calls, so `fn` works under `SmtCollectHandler`
    *and* under ground execution (the rewrite preserves observable semantics
    when both branches are pure-expression-shaped).

    Restrictions
    ------------
    The function's source must be retrievable via `inspect.getsource` —
    works for file-defined functions, fails on lambdas without source.
    See `_SymbolicIfRewriter` for the supported `if`/`else` shapes; other
    shapes raise `SyntaxError` at decoration time, which is the loud-failure
    signal an LLM-emit pipeline should consume as a residual.
    """
    src = textwrap.dedent(inspect.getsource(fn))
    tree = ast.parse(src)
    new_tree = _SymbolicIfRewriter().visit(tree)

    # Strip @symbolic from the rewritten function to avoid infinite recursion
    fn_def = new_tree.body[0]
    if isinstance(fn_def, (ast.FunctionDef, ast.AsyncFunctionDef)):
        fn_def.decorator_list = [
            d for d in fn_def.decorator_list
            if not (isinstance(d, ast.Name) and d.id == "symbolic")
            and not (isinstance(d, ast.Attribute) and d.attr == "symbolic")
        ]

    ast.fix_missing_locations(new_tree)
    code = compile(new_tree, fn.__code__.co_filename, "exec")

    ns = dict(fn.__globals__)
    ns["if_then_else"] = if_then_else
    exec(code, ns)
    return ns[fn.__name__]
