"""AST source-rewriting for native `if/else` in autumn-op-using callables.

The `@symbolic` decorator rewrites Python's `if/else` statements and
`a if c else b` ternaries into explicit `if_then_else` op calls, so the
resulting function works under handlers that can't dispatch on a Python
`bool` derived from a symbolic value (notably `SmtCollectHandler` —
Z3 expressions raise on `__bool__`).

This module is deliberately separate from `autumn_py.smt`: AST rewriting
is a generic Python-syntax concern, not an SMT concern. Effectful's PR
#288 (eb-disassembler) is the cleaner long-term path; until that lands
and grows function-body / statement-level support, source-rewriting is
the pragmatic alternative for code we can read off disk.

Restrictions
------------
The function's source must be retrievable via `inspect.getsource` —
file-defined functions only; lambdas without source fail at decoration.
Supported `if/else` shapes are:

* Ternary: ``x = a if cond else b`` →
  ``x = if_then_else(cond, a, b)``.
* Assignment-form: ``if cond: x = a; else: x = b`` →
  ``x = if_then_else(cond, a, b)``. Both branches must assign to the
  same single target.
* Return-form: ``if cond: return a; else: return b`` →
  ``return if_then_else(cond, a, b)``. Both branches must return a value.
* Same-callable form: ``if cond: f(args1); else: f(args2)`` →
  ``f(...)`` with each differing argument position wrapped in
  ``if_then_else(cond, args1[i], args2[i])``. Function callable and
  arity must match; differing string-literal args (typically state-var
  names) are rejected since names cannot be conditional.

Asymmetric branches and structurally different bodies raise
`SyntaxError` at decoration time.

Caveats
-------
The rewrite preserves observable semantics under ground execution
**only when both branches are pure expressions** (arithmetic, op calls
with the same effect set). Python's `if/else` short-circuits — only
one branch's side effects fire — but `if_then_else` evaluates both
arguments eagerly. The supported patterns above all factor side
effects out of the conditional (assignments to a target, returns, or
a single same-callable call), which keeps the rewrite sound in
practice. Patterns with branch-only side effects (e.g.
``if cond: set_var(...)`` with empty `else`) are rejected.
"""
from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Callable, NoReturn

from .ops import if_then_else


class _SymbolicIfRewriter(ast.NodeTransformer):
    """AST transformer that lifts the supported `if/else` shapes into
    `if_then_else` op calls. See module docstring for the exact patterns."""

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
            if (
                ast.dump(b.targets[0]) != ast.dump(o.targets[0])
                or len(b.targets) != 1
                or len(o.targets) != 1
            ):
                self._fail(node, "both branches must assign to the same single target")
            return ast.Assign(
                targets=[b.targets[0]],
                value=ast.Call(
                    func=ast.Name(id="if_then_else", ctx=ast.Load()),
                    args=[node.test, b.value, o.value],
                    keywords=[],
                ),
            )
        if isinstance(b, ast.Return) and isinstance(o, ast.Return):
            if b.value is None or o.value is None:
                self._fail(node, "both branches' return must have a value")
            assert b.value is not None and o.value is not None  # narrowed for mypy
            return ast.Return(
                value=ast.Call(
                    func=ast.Name(id="if_then_else", ctx=ast.Load()),
                    args=[node.test, b.value, o.value],
                    keywords=[],
                )
            )
        if (
            isinstance(b, ast.Expr)
            and isinstance(o, ast.Expr)
            and isinstance(b.value, ast.Call)
            and isinstance(o.value, ast.Call)
        ):
            bc, oc = b.value, o.value
            if ast.dump(bc.func) != ast.dump(oc.func):
                self._fail(node, "branches call different callables")
            if len(bc.args) != len(oc.args):
                self._fail(node, "branches call with different arity")
            if [ast.dump(k) for k in bc.keywords] != [ast.dump(k) for k in oc.keywords]:
                self._fail(node, "branches call with different keyword arguments")
            new_args: list[ast.expr] = []
            for i, (ba, oa) in enumerate(zip(bc.args, oc.args)):
                if ast.dump(ba) == ast.dump(oa):
                    new_args.append(ba)
                else:
                    if isinstance(ba, ast.Constant) and isinstance(ba.value, str):
                        self._fail(
                            node,
                            f"argument {i} is a differing string literal — "
                            "cannot lift to if_then_else",
                        )
                    if isinstance(oa, ast.Constant) and isinstance(oa.value, str):
                        self._fail(
                            node,
                            f"argument {i} is a differing string literal — "
                            "cannot lift to if_then_else",
                        )
                    new_args.append(
                        ast.Call(
                            func=ast.Name(id="if_then_else", ctx=ast.Load()),
                            args=[node.test, ba, oa],
                            keywords=[],
                        )
                    )
            return ast.Expr(
                value=ast.Call(func=bc.func, args=new_args, keywords=bc.keywords)
            )
        self._fail(
            node,
            "branches must both be assignments to the same target, "
            "both `return`, or both expression-statements calling the same function",
        )

    @staticmethod
    def _fail(node: ast.If, reason: str) -> NoReturn:
        raise SyntaxError(
            f"@symbolic cannot rewrite this if/else (line {node.lineno}): {reason}. "
            f"Supported patterns: assignment-form, return-form, same-callable-form, "
            f"or ternary `a if cond else b`. For other cases, use "
            f"`if_then_else(cond, e1, e2)` explicitly."
        )


def symbolic(fn: Callable) -> Callable:
    """Decorator: rewrite supported `if/else` patterns in `fn`'s body
    into `if_then_else` op calls.

    The rewrite preserves observable semantics on the supported patterns
    (assignment-form, return-form, ternary, same-callable-form) where
    branches are factored to be pure-expression-shaped. See module
    docstring for the full list.
    """
    src = textwrap.dedent(inspect.getsource(fn))
    tree = ast.parse(src)
    new_tree = _SymbolicIfRewriter().visit(tree)

    fn_def = new_tree.body[0]
    if isinstance(fn_def, (ast.FunctionDef, ast.AsyncFunctionDef)):
        fn_def.decorator_list = []

    ast.fix_missing_locations(new_tree)
    code = compile(new_tree, fn.__code__.co_filename, "exec")

    ns = fn.__globals__
    ns.setdefault("if_then_else", if_then_else)
    local_ns: dict = {}
    exec(code, ns, local_ns)
    return local_ns[fn.__name__]
