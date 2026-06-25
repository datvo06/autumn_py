"""Static type inference for next-expressions (annotation-driven).

Run a clause under the type-domain interpretation and it evaluates to a Python
*type* instead of a value. Every op's result type comes from effectful's own
``op.__type_rule__`` — installed as the default for *all* ops by handling the
universal ``apply`` operation (the same mechanism ``effectful.typeof`` uses), so
there is no per-op enumeration. Only ``get_var`` / ``get_prev_var`` are
special-cased: their result type is the *state var's* declared type, which lives
in the env, not on the op signature. (``@obj`` types are real classes, so they
canonicalize like any other; no bespoke TypeVar binder is needed.)
"""
from __future__ import annotations

import typing
from collections import ChainMap
from collections.abc import Mapping
from typing import Any, Callable

from effectful.internals.runtime import interpreter
from effectful.internals.unification import Box
from effectful.ops.semantics import apply

from .ops import get_prev_var, get_var


def _box_if_type(x: Any) -> Any:
    """Normalize an argument for ``__type_rule__``: Box a type token, and lift a
    concrete list/tuple literal to its ``list[elem]`` type — its elements may be
    type tokens (e.g. ``addObj(xs, Particle)`` builds ``[Particle]``), which
    ``nested_type`` can't recurse into. Other values (lambdas, Boxes) pass raw."""
    if isinstance(x, type) or typing.get_origin(x) is not None:
        return Box(x)
    if isinstance(x, (list, tuple)):
        if not x:
            return Box(list)
        elem = x[0]
        return Box(list[elem if isinstance(elem, type) else type(elem)])
    return x


def infer_type(fn: Callable[[], Any], env: Mapping[str, Any]) -> Any:
    """Run zero-arg ``fn`` in the type domain; return the Python type of its
    result. ``env`` maps state-var names to types (read-only, layered)."""
    env = env if isinstance(env, ChainMap) else ChainMap(env)

    def _apply(op, *args, **kwargs):
        if op is get_var or op is get_prev_var:
            name = args[0]
            if name not in env:
                raise NameError(f"type-inference: unbound state var {name!r}")
            return Box(env[name])
        return Box(op.__type_rule__(*(_box_if_type(a) for a in args), **kwargs))

    with interpreter({apply: _apply}):
        result = fn()
    return result.value if isinstance(result, Box) else result


def env_from_program(program_cls) -> ChainMap:
    """Build a read-only type environment from a ``@program`` class: each
    state var's declared ``type_`` (``object`` if unannotated)."""
    spec = getattr(program_cls, "_autumn_spec", None)
    if spec is None:
        raise TypeError(f"{program_cls.__name__} is not @program-decorated")
    return ChainMap({
        sv.name: (sv.type_ if sv.type_ is not None else object)
        for sv in spec.state_vars
        if sv.name is not None
    })
