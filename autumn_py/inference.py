"""Static type inference for next-expressions.

Implements the §2 typing rules as a handler-stack computation: an
expression $e$ that evaluates to a value under the standard handler stack
evaluates to a *type token* under the TypeOfHandler stack. This is the
"same evaluator, different handlers" thesis — type inference is one more
non-standard semantics over $\\mathsf{Comp}_A$.

Closes §2.2's $\\texttt{uniformChoice} : \\texttt{List}\\,T \\to T$ gap that
the runtime _check_type alone cannot enforce: the element type $T$ is
threaded through prev / list-builders / list-transformers and surfaces as
the type of a uniformChoice result.
"""
from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import Any, Callable

from effectful.ops.semantics import handler
from effectful.ops.syntax import ObjectInterpretation, implements

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
from .values import Cell, ObjectInstance, Position


# --------------------------------------------------------------------------
# Type tokens
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Type:
    """A type token used during inference.

    `base` is the underlying Python type (int, bool, str, list, Position,
    Cell, ObjectInstance, etc.). For list types, `elem` is the element
    Type. For ObjectInstance-of-class-K, `obj_spec` is K's declared spec.
    """
    base: Any
    elem: "Type | None" = None
    obj_spec: Any = None

    @classmethod
    def of(cls, py_type: Any) -> "Type":
        """Lift a Python type annotation into a Type token."""
        if py_type is None or py_type is type(None):
            return cls(type(None))
        # Parameterised generics: list[T], typing.List[T]
        origin = typing.get_origin(py_type)
        if origin is list:
            args = typing.get_args(py_type)
            elem = cls.of(args[0]) if args else None
            return cls(list, elem=elem)
        # @obj-decorated factory carries a spec
        spec = getattr(py_type, "__autumn_obj_spec__", None)
        if spec is not None:
            return cls(ObjectInstance, obj_spec=spec)
        return cls(py_type)

    @classmethod
    def list_of(cls, elem: "Type") -> "Type":
        return cls(list, elem=elem)

    def __repr__(self) -> str:
        if self.base is list:
            return f"Type[list[{self.elem!r}]]" if self.elem is not None else "Type[list]"
        if self.obj_spec is not None:
            return f"Type[{self.obj_spec.name}]"
        return f"Type[{getattr(self.base, '__name__', self.base)}]"

    @property
    def is_list(self) -> bool:
        return self.base is list

    @property
    def is_obj(self) -> bool:
        return self.base is ObjectInstance


def is_type_token(x: Any) -> bool:
    """Test whether x is a type token (used by stdlib for polymorphic dispatch)."""
    return isinstance(x, Type)


# --------------------------------------------------------------------------
# TypeOfHandler — interprets @defops in the type domain
# --------------------------------------------------------------------------

class TypeOfHandler(ObjectInterpretation):
    """Evaluation under this handler stack returns Type tokens instead of
    values. Each @defop is reinterpreted to return its result's type.

    The handler holds a type environment $\\Gamma$ mapping state-var names
    (and other named scopes) to their declared types.
    """

    def __init__(self, env: dict[str, Type]) -> None:
        self.env = env

    @implements(get_var)
    def _get_var(self, name: str) -> Type:
        if name not in self.env:
            raise NameError(f"type-inference: unbound state var {name!r}")
        return self.env[name]

    @implements(get_prev_var)
    def _get_prev_var(self, name: str) -> Type:
        if name not in self.env:
            raise NameError(f"type-inference: unbound state var {name!r}")
        return self.env[name]

    @implements(sample_uniform)
    def _sample_uniform(self, xs: Any) -> Type:
        # Closes the §2.2 gap: uniformChoice : List T → T
        if isinstance(xs, Type) and xs.is_list and xs.elem is not None:
            return xs.elem
        # Tuple of Type tokens (e.g. uniformChoice over a literal list)
        if isinstance(xs, (tuple, list)) and xs:
            first = xs[0]
            if isinstance(first, Type):
                return first
            # Heterogeneous concrete sample inside a typed lambda — give
            # the most specific common type we can.
            return Type(type(first))
        return Type(object)

    @implements(alloc_obj_id)
    def _alloc(self) -> Type:
        return Type(int)

    @implements(is_event_active)
    def _is_active(self, name: str) -> Type:
        return Type(bool)

    @implements(get_click_pos)
    def _click_pos(self) -> Type:
        return Type(Position)

    @implements(grid_size)
    def _grid_size(self) -> Type:
        return Type(int)

    @implements(state_has)
    def _state_has(self, name: str) -> Type:
        return Type(bool)

    @implements(all_objs)
    def _all_objs(self) -> Type:
        return Type.list_of(Type(ObjectInstance))

    # -- structural list combinators: list type preserved (sound for
    # Autumn's structural uses, where the function's domain and codomain
    # coincide). All the higher-level Autumn list primitives compose these.

    @implements(concat_op)
    def _concat(self, xs: Any, ys: Any) -> Type:
        if isinstance(xs, Type):
            return xs
        return Type(list)

    @implements(filter_op)
    def _filter(self, xs: Any, pred: Any) -> Type:
        if isinstance(xs, Type):
            return xs
        return Type(list)

    @implements(map_op)
    def _map(self, xs: Any, fn: Any) -> Type:
        if isinstance(xs, Type):
            return xs
        return Type(list)

    @implements(adjPositions_op)
    def _adj_positions(self, p: Any) -> Type:
        return Type.list_of(Type(Position))


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def infer_type(fn: Callable[[], Any], env: dict[str, Type]) -> Type:
    """Run zero-arg `fn` under a type-inference handler stack and return
    the type of its result.

    `env` maps state-var names to their declared Type tokens. The function
    is expected to invoke `prev`, `uniformChoice`, and the typed-stdlib
    primitives — these are reinterpreted to thread type tokens through.

    Concrete arithmetic (e.g. `m.bullets + 1`) and pure-Python control
    flow are *not* re-interpreted; they run normally on whatever values
    they see. For straight-line synthesizable next-expressions composed
    of the typed primitives, this is sufficient to close the §2.2 gap.
    """
    with handler(TypeOfHandler(env)):
        return fn()


def env_from_program(program_cls) -> dict[str, Type]:
    """Build a type environment from a @program-decorated class's
    StateVar declarations. Each state var's `type_` becomes a Type
    token in the environment."""
    spec = getattr(program_cls, "_autumn_spec", None)
    if spec is None:
        raise TypeError(f"{program_cls.__name__} is not @program-decorated")
    env: dict[str, Type] = {}
    for sv in spec.state_vars:
        if sv.name is None:
            continue
        env[sv.name] = Type.of(sv.type_) if sv.type_ is not None else Type(object)
    return env
