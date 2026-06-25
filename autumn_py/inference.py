"""Static type inference for next-expressions (annotation-driven).

Run a clause under the type-domain interpretation and it evaluates to a Python
*type* instead of a value. The type rules live on the ``@defop`` signatures
(single source of truth): each op's result type comes from effectful's own
``op.__type_rule__``, which strips ``Annotated`` and binds TypeVars via
``unify`` / ``substitute`` over Boxed type-valued args. Only ``get_var`` /
``get_prev_var`` are bespoke — their result type is the *state var's* declared
type, which lives in the env, not on the op signature.

(``@obj`` types are real classes, so they canonicalize like any other type; no
bespoke TypeVar binder is needed.)
"""
from __future__ import annotations

import typing
from collections import ChainMap
from collections.abc import Mapping
from typing import Any, Callable

from effectful.internals.unification import Box
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


def _box_if_type(x: Any) -> Any:
    """Normalize an argument for ``__type_rule__``: Box a type token, and lift a
    concrete list/tuple literal to its ``list[elem]`` type — its elements may be
    type tokens (e.g. ``addObj(xs, Particle)`` builds ``[Particle]``), which
    ``nested_type`` can't recurse into. Other values (lambdas) pass through raw."""
    if isinstance(x, type) or typing.get_origin(x) is not None:
        return Box(x)
    if isinstance(x, (list, tuple)):
        if not x:
            return Box(list)
        elem = x[0]
        return Box(list[elem if isinstance(elem, type) else type(elem)])
    return x


def _annotation_rule(op) -> Callable[..., Any]:
    """Type-domain interpretation of ``op``: effectful derives the result type
    from the op's own signature via ``__type_rule__`` (strips ``Annotated``,
    binds TypeVars via ``unify``)."""
    def rule(*arg_types: Any) -> Any:
        return op.__type_rule__(*(_box_if_type(a) for a in arg_types))
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
