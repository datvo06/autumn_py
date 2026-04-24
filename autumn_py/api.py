from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .ops import alloc_obj_id, get_prev_var, get_var, set_var
from .values import Cell, ObjectClassSpec, ObjectInstance, Position


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
        self._next_fn = fn
        return fn

    def initializer(self, fn: Callable[[], Any]) -> Callable[[], Any]:
        """Register a zero-arg function producing this var's initial value.
        Invoked by the Runtime's init phase under the persistent handler stack,
        so it may call effects like alloc_obj_id (ObjectInstance construction).
        Takes precedence over ``init=`` when both are set."""
        self._init_fn = fn
        return fn

    def initial_value(self) -> Any:
        """Resolve the init value at init-phase time. Prefers the initializer
        callable if one was registered; falls back to the static ``init=``."""
        return self._init_fn() if self._init_fn is not None else self.init


def prev(ref) -> Any:
    """Read the previous-tick value of a state var.

    `ref` can be either the StateVar descriptor (preferred, identity-based) or
    a raw attribute name.
    """
    if isinstance(ref, StateVar):
        assert ref.name is not None
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
    annotations = cls.__dict__.get("__annotations__") or {}
    field_names = tuple(annotations.keys())

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
        for name, val in list(cls.__dict__.items()):
            if isinstance(val, StateVar):
                spec.state_vars.append(val)
            elif hasattr(val, "__autumn_obj_spec__"):
                spec.obj_classes.append(val)
        spec.on_clauses = list(_pending_on_clauses)
        _pending_on_clauses.clear()
        cls._autumn_spec = spec
        return cls

    return decorator
