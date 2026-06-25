"""Static type inference for next-expressions (annotation-driven).

Run a clause under the type-domain interpretation and it evaluates to a Python
*type* instead of a value. The type rules now live on the ``@defop`` signatures
themselves (single source of truth): each op's result type is read off its
return annotation, with any TypeVar bound from the argument types. Only
``get_var`` / ``get_prev_var`` are bespoke — their result type is the *state
var's* declared type, which can't live on the op signature.

(effectful's ``typeof`` / ``op.__type_rule__`` derive the same rules from
signatures, but consume *values* — they call ``nested_type`` internally — so
they don't compose with this type-domain evaluation; and ``unify`` /
``canonicalize`` raise on an ``@obj`` factory element. We bind TypeVars with
``typing.get_args``, which is both simpler and ``@obj``-safe.)
"""
from __future__ import annotations

import inspect
import typing
from collections import ChainMap
from collections.abc import Mapping
from typing import Any, Callable, TypeVar

from effectful.ops.semantics import handler

from .ops import (
    adjPositions_op,
    all_objs,
    alloc_obj_id,
    concat_op,
    filter_op,
    get_click_pos,
    get_prev_var,
    get_var,
    grid_size,
    is_event_active,
    map_op,
    sample_uniform,
    state_has,
)

# Ops whose result type is read off their @defop return annotation.
_ANNOTATION_OPS = (
    sample_uniform, alloc_obj_id, is_event_active, get_click_pos, grid_size,
    state_has, all_objs, map_op, filter_op, concat_op, adjPositions_op,
)


def _as_type(x: Any) -> Any:
    """Normalize an argument to a type token. Most args flowing through the
    type domain are already types; a concrete collection literal in the clause
    (e.g. ``uniformChoice([1, 2, 3])``) is lifted to ``list[<element type>]``."""
    if isinstance(x, (list, tuple)):
        return list[type(x[0])] if x else list
    return x


def _bind(param_ann: Any, arg_type: Any, subs: dict) -> None:
    """Bind TypeVars in ``param_ann`` from ``arg_type`` structurally (via
    ``get_args``, so an ``@obj`` factory element type binds fine where
    effectful's ``canonicalize`` would reject it). First binding wins."""
    if isinstance(param_ann, TypeVar):
        subs.setdefault(param_ann, arg_type)
        return
    for p, a in zip(typing.get_args(param_ann), typing.get_args(arg_type)):
        _bind(p, a, subs)


def _subst(ann: Any, subs: dict) -> Any:
    """Substitute bound TypeVars into ``ann``."""
    if isinstance(ann, TypeVar):
        return subs.get(ann, object)
    args = typing.get_args(ann)
    if not args:
        return ann
    new = tuple(_subst(a, subs) for a in args)
    if new == args:
        return ann
    origin = typing.get_origin(ann)
    return origin[new[0]] if len(new) == 1 else origin[new]


def _annotation_rule(op) -> Callable[..., Any]:
    """Type-domain interpretation of ``op``: its declared return type with any
    TypeVar resolved from the argument types."""
    sig = inspect.signature(op)
    ret = sig.return_annotation
    param_anns = [p.annotation for p in sig.parameters.values()]

    def rule(*arg_types: Any) -> Any:
        subs: dict = {}
        for pa, at in zip(param_anns, arg_types):
            _bind(pa, _as_type(at), subs)
        return _subst(ret, subs)

    return rule


def type_interpretation(env: Mapping[str, Any]) -> dict:
    """Build the type-domain interpretation: ``get_var`` / ``get_prev_var``
    look up the (read-only, layered) env; every other op reads its annotation."""
    env = env if isinstance(env, ChainMap) else ChainMap(env)

    def _lookup(name: str) -> Any:
        if name not in env:
            raise NameError(f"type-inference: unbound state var {name!r}")
        return env[name]

    interp: dict = {get_var: _lookup, get_prev_var: _lookup}
    for op in _ANNOTATION_OPS:
        interp[op] = _annotation_rule(op)
    return interp


def infer_type(fn: Callable[[], Any], env: Mapping[str, Any]) -> Any:
    """Run zero-arg ``fn`` under the type-domain interpretation; return the
    Python type of its result. ``env`` maps state-var names to types."""
    with handler(type_interpretation(env)):
        return fn()


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
