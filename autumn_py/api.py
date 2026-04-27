from __future__ import annotations

import typing
from dataclasses import dataclass, field
from typing import Any, Callable

from .ops import alloc_obj_id, get_prev_var, get_var, set_var
from .values import Cell, ObjectClassSpec, ObjectInstance, Position


# --------------------------------------------------------------------------
# Lightweight runtime type checker.
#
# Autumn's surface syntax includes (: x T) annotations. The C++ interpreter
# parses and ignores them; we don't. The checker validates state-var init
# values, @obj field values, initializer return types, and next-expression
# return types against their declared annotations.
#
# It is intentionally lightweight: isinstance for ground types, list-only
# for list[...] (no element-type check), special-cased ObjectInstance for
# user-declared object classes, and `object` as a permissive escape hatch.
# --------------------------------------------------------------------------

class TypeMismatch(TypeError):
    """Raised when a value fails its declared Autumn type annotation."""


def _check_type(value: Any, expected: Any, context: str) -> None:
    """Verify that ``value`` is compatible with the declared type ``expected``.

    Raises ``TypeMismatch`` on failure. Does nothing when ``expected`` is
    ``None`` (no annotation) or ``object`` (universal).
    """
    if expected is None or expected is object or expected is Any:
        return

    # Parameterised generics: list[T], tuple[T], etc. We handle list[T] (and
    # typing.List[T]) by checking the container kind and recursively
    # checking each element's type. Other generics fall through.
    origin = typing.get_origin(expected)
    if origin is list:
        if not isinstance(value, list):
            raise TypeMismatch(
                f"{context}: expected list, got {type(value).__name__}"
            )
        args = typing.get_args(expected)
        if args:
            elem_type = args[0]
            for i, elem in enumerate(value):
                _check_type(elem, elem_type, f"{context}[{i}]")
        return

    # @obj-decorated factories carry their spec; treat them as object-class
    # annotations: the value must be an ObjectInstance of that spec.
    obj_spec = getattr(expected, "__autumn_obj_spec__", None)
    if obj_spec is not None:
        if isinstance(value, ObjectInstance) and value.cls is obj_spec:
            return
        raise TypeMismatch(
            f"{context}: expected {obj_spec.name} instance, "
            f"got {type(value).__name__}"
        )

    # Plain Python types — list, int, bool, str, dict, ObjectInstance,
    # Position, Cell, ...
    if isinstance(expected, type):
        if isinstance(value, expected):
            return
        # Permit None as a placeholder for object-valued state vars whose
        # initializer hasn't run yet.
        if value is None and expected in (object,):
            return
        raise TypeMismatch(
            f"{context}: expected {expected.__name__}, "
            f"got {type(value).__name__} (value: {value!r})"
        )

    # Anything else — pass through; we don't enforce more elaborate generics.


# --------------------------------------------------------------------------
# Program-spec record attached to a @program-decorated class.
# --------------------------------------------------------------------------

@dataclass
class OnClause:
    predicate: Any
    body: Callable[[], Any]
    name: str


@dataclass
class ProgramSpec:
    state_vars: list["StateVar"] = field(default_factory=list)
    on_clauses: list[OnClause] = field(default_factory=list)
    obj_classes: list[Any] = field(default_factory=list)
    config: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# StateVar: a descriptor that represents a named piece of per-tick state.
# --------------------------------------------------------------------------

class StateVar:
    """Descriptor bound to a class attribute, representing a named Autumn state var.

    Usage inside a @program class::

        foo = StateVar(list, init=[])

        @foo.next
        def _():
            return ...   # expression producing foo's next-tick value

        # For initial values that require handler-backed effects
        # (e.g. constructing an ObjectInstance, which calls alloc_obj_id):

        bar = StateVar(object)

        @bar.initializer
        def _():
            return Mario(0, Position(7, 15))
    """

    def __init__(self, type_: type | None = None, *, init: Any = None) -> None:
        self.type_ = type_
        self.init = init
        self.name: str | None = None
        self._next_fn: Callable[[], Any] | None = None
        self._init_fn: Callable[[], Any] | None = None

    def __set_name__(self, owner, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"StateVar({self.name!r})"

    def get(self) -> Any:
        assert self.name is not None
        return get_var(self.name)

    def set(self, value: Any) -> None:
        assert self.name is not None
        set_var(self.name, value)

    def next(self, fn: Callable[[], Any]) -> Callable[[], Any]:
        """Register the decorated function as this var's initnext-next expression."""
        self._check_return_annotation(fn, "next-expression")
        self._next_fn = fn
        return fn

    def initializer(self, fn: Callable[[], Any]) -> Callable[[], Any]:
        """Register a zero-arg function producing this var's initial value.
        Invoked by the Runtime's init phase under the persistent handler stack,
        so it may call effects like alloc_obj_id (ObjectInstance construction).
        Takes precedence over ``init=`` when both are set."""
        self._check_return_annotation(fn, "initializer")
        self._init_fn = fn
        return fn

    def initial_value(self) -> Any:
        """Resolve the init value at init-phase time. Prefers the initializer
        callable if one was registered; falls back to the static ``init=``."""
        return self._init_fn() if self._init_fn is not None else self.init

    def _check_return_annotation(self, fn: Callable, kind: str) -> None:
        """If the registered function has a return annotation that conflicts
        with this StateVar's declared type, raise at decoration time."""
        if self.type_ is None or self.type_ is object:
            return
        try:
            hints = typing.get_type_hints(fn)
        except Exception:
            return
        ret = hints.get("return")
        if ret is None or ret is object or ret is Any:
            return
        # Compatible iff same type, or matching parameterised list[T] origin.
        if ret is self.type_:
            return
        sv_origin = typing.get_origin(self.type_)
        ret_origin = typing.get_origin(ret)
        if sv_origin is not None and sv_origin is ret_origin:
            return
        raise TypeMismatch(
            f"{kind} for state var {self.name!r}: function return type "
            f"{getattr(ret, '__name__', ret)!r} does not match "
            f"declared type {getattr(self.type_, '__name__', self.type_)!r}"
        )


def prev(ref) -> Any:
    """Read the previous-tick value of a state var.

    `ref` can be either the StateVar descriptor (preferred, identity-based) or
    a raw attribute name. §2.2: requires the referent to be a declared
    state variable; an unbound descriptor (no `__set_name__` ran) or an
    unknown name raises immediately.
    """
    if isinstance(ref, StateVar):
        if ref.name is None:
            raise TypeMismatch(
                "prev() called on an unbound StateVar (no __set_name__ ran). "
                "Did you forget to put it inside a class body?"
            )
        return get_prev_var(ref.name)
    if isinstance(ref, str):
        return get_prev_var(ref)
    raise TypeError(f"prev() expects a StateVar or str, got {type(ref).__name__}")


# --------------------------------------------------------------------------
# @obj: decorate a class to turn it into an Autumn object factory.
# --------------------------------------------------------------------------

def obj(cls):
    """Class decorator: registers an Autumn object class.

    The decorated class body may declare:

    * ``cell = Cell(...)`` — single-cell object.
    * ``cells = [Cell(...), ...]`` — multi-cell.
    * Typed annotations (``bullets: int``, ``living: bool``, etc.) —
      named per-instance fields. Declaration order is the positional
      argument order for construction.

    Returns a callable factory. The factory accepts positional args in
    the order ``(*fields, origin)`` to mirror the Autumn s-expression
    form ``(Mario 0 (Position 7 15))``. An ``ObjectInstance`` with a
    fresh id is returned.

    A Cell's color may be a ``Callable[[ObjectInstance], str]`` so the
    rendered color can depend on field values (Game of Life's Particle).
    """
    # `from __future__ import annotations` may have stringified the
    # annotations; resolve them to real types via the class's namespace.
    raw_annotations = cls.__dict__.get("__annotations__") or {}
    field_names = tuple(raw_annotations.keys())
    try:
        annotations = typing.get_type_hints(cls)
    except Exception:
        # Fallback: use the raw values (may be strings; _check_type will
        # then no-op on them, which preserves backward compatibility).
        annotations = raw_annotations

    cell_attr = cls.__dict__.get("cell")
    cells_attr = cls.__dict__.get("cells")
    if cells_attr is not None:
        cells: tuple[Cell, ...] = tuple(cells_attr)
    elif cell_attr is not None:
        cells = (cell_attr,)
    else:
        raise ValueError(
            f"@obj class {cls.__name__!r} must define `cell = Cell(...)` "
            f"or `cells = [...]`"
        )

    spec = ObjectClassSpec(name=cls.__name__, cells=cells, field_names=field_names)

    def factory(*args: Any) -> ObjectInstance:
        expected = len(field_names) + 1
        if len(args) != expected:
            raise TypeError(
                f"{cls.__name__}({', '.join(field_names + ('origin',))}) "
                f"expected {expected} positional arg(s), got {len(args)}"
            )
        *field_vals, origin = args
        if not isinstance(origin, Position):
            raise TypeError(
                f"{cls.__name__}: final argument must be a Position, "
                f"got {type(origin).__name__}"
            )
        # Type-check each declared field against its annotation.
        for fname, fval in zip(field_names, field_vals):
            _check_type(
                fval, annotations[fname],
                context=f"@obj {cls.__name__}.{fname}",
            )
        fields = dict(zip(field_names, field_vals))
        return ObjectInstance(
            cls=spec, origin=origin, id=alloc_obj_id(), fields=fields
        )

    factory.__autumn_obj_spec__ = spec  # type: ignore[attr-defined]
    factory.__name__ = cls.__name__
    factory.__qualname__ = cls.__qualname__
    return factory


# --------------------------------------------------------------------------
# @on: register an on-clause.
# --------------------------------------------------------------------------
#
# Autumn programs typically declare many on-clauses. Users write them as
# ``@on(pred) def _(): ...`` repeatedly and rely on shadowing not to matter.
# To preserve all of them regardless of name, ``@on`` pushes each clause onto
# a module-level pending list that ``@program`` drains when the class finishes
# defining. Class bodies execute sequentially, so this works without threads
# or metaclasses.

_pending_on_clauses: list["OnClause"] = []


def on(predicate):
    """Decorate a 0-arg function to register it as an on-clause body.

    Users may reuse the same Python name (e.g. ``def _()``) for every
    on-clause; the registration bypasses ``cls.__dict__`` via an internal
    pending list drained by ``@program``.
    """

    def decorator(fn: Callable[[], Any]) -> Callable[[], Any]:
        fn.__autumn_on_pred__ = predicate  # type: ignore[attr-defined]
        _pending_on_clauses.append(
            OnClause(predicate=predicate, body=fn, name=getattr(fn, "__name__", "_"))
        )
        return fn

    return decorator


# --------------------------------------------------------------------------
# @program: collect state vars, on-clauses, and object classes off a class.
# --------------------------------------------------------------------------

def program(**config):
    """Class decorator. Populates `cls._autumn_spec: ProgramSpec`.

    Drains the module-level on-clause pending list (everything that was
    decorated with @on during this class body's execution) into
    ``spec.on_clauses``.
    """

    def decorator(cls):
        spec = ProgramSpec(config=dict(config))
        for _name, val in list(cls.__dict__.items()):
            if isinstance(val, StateVar):
                spec.state_vars.append(val)
            elif hasattr(val, "__autumn_obj_spec__"):
                spec.obj_classes.append(val)
        spec.on_clauses = list(_pending_on_clauses)
        _pending_on_clauses.clear()
        cls._autumn_spec = spec
        return cls

    return decorator
