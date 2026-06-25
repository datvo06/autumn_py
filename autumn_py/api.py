from __future__ import annotations

import typing
from dataclasses import dataclass
from dataclasses import field as field
from typing import Any, Callable, Generic, TypeVar, dataclass_transform

from .ops import alloc_obj_id, get_prev_var, get_var, set_var
from .values import Cell, ObjectClassSpec, ObjectInstance, Position

T = TypeVar("T")


# --------------------------------------------------------------------------
# Lightweight runtime type checker.
# --------------------------------------------------------------------------

class TypeMismatch(TypeError):
    """Raised when a value fails its declared Autumn type annotation."""


def _check_type(value: Any, expected: Any, context: str) -> None:
    """Verify that ``value`` is compatible with the declared type ``expected``."""
    if expected is None or expected is object or expected is Any:
        return

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

    obj_spec = getattr(expected, "__autumn_obj_spec__", None)
    if obj_spec is not None:
        if isinstance(value, ObjectInstance) and value.cls is obj_spec:
            return
        raise TypeMismatch(
            f"{context}: expected {obj_spec.name} instance, "
            f"got {type(value).__name__}"
        )

    if isinstance(expected, type):
        if isinstance(value, expected):
            return
        raise TypeMismatch(
            f"{context}: expected {expected.__name__}, "
            f"got {type(value).__name__} (value: {value!r})"
        )


# --------------------------------------------------------------------------
# Spec records attached to a @program-decorated class.
# --------------------------------------------------------------------------

class StateVar(Generic[T]):
    """Spec record for a state-variable declaration, generic in the
    value type ``T``.

    Declared in a ``@program`` class body as a descriptor::

        @program(grid_size=16)
        class MyGame:
            step_count = StateVar(int, init=0)
            enemies = StateVar(list, init=[])

            @step_count.next
            def _():
                return prev(MyGame.step_count) + 1

    ``.next`` registers the var's next-expression; ``.initializer``
    registers a (possibly effect-using) initializer. The generic
    parameter ``T`` flows through ``.get()``, ``.set(v)`` and
    ``prev(MyGame.x)`` so type-checkers track the value type.
    """

    def __init__(
        self,
        type_: type[T] | None = None,
        *,
        init: T | None = None,
        init_fn: Callable[[], T] | None = None,
        name: str | None = None,
    ) -> None:
        self.type_ = type_
        self.init = init
        self.init_fn = init_fn
        self.name = name
        self._next_fn: Callable[[], Any] | None = None

    def __set_name__(self, owner, name: str) -> None:
        # Capture the state-var name from the class-attribute binding.
        if self.name is None:
            self.name = name

    def __repr__(self) -> str:
        return f"StateVar({self.name!r})"

    def initial_value(self) -> Any:
        if self.init_fn is not None:
            return self.init_fn()
        return self.init

    # Arithmetic / comparison / coercion delegate to .get() so the value flows
    # through the handler stack, not the StateVar object. __eq__ / __hash__ are
    # left as identity defaults so StateVar stays hashable / dict-keyable.

    def __add__(self, other):       return self.get() + other
    def __radd__(self, other):      return other + self.get()
    def __sub__(self, other):       return self.get() - other
    def __rsub__(self, other):      return other - self.get()
    def __mul__(self, other):       return self.get() * other
    def __rmul__(self, other):      return other * self.get()
    def __mod__(self, other):       return self.get() % other
    def __floordiv__(self, other):  return self.get() // other
    def __truediv__(self, other):   return self.get() / other
    def __lt__(self, other):        return self.get() < other
    def __gt__(self, other):        return self.get() > other
    def __le__(self, other):        return self.get() <= other
    def __ge__(self, other):        return self.get() >= other
    def __bool__(self):             return bool(self.get())
    def __int__(self):              return int(self.get())
    def __float__(self):            return float(self.get())
    def __index__(self):            return int(self.get())
    def __len__(self):              return len(self.get())

    # --- descriptor API: get / set / next / initializer -----------------

    def get(self) -> T:
        assert self.name is not None
        return get_var(self.name)

    def set(self, value: T) -> None:
        assert self.name is not None
        set_var(self.name, value)

    def next(self, fn: Callable[[], Any]) -> Callable[[], Any]:
        """``@<sv>.next def _(): ...`` — register the var's next-expression.

        Spec metadata attached via ``@spec(...)``/``@modifies(...)``/
        ``@no_stochastic`` is preserved on ``fn.__autumn_spec__`` and
        materialised by ``@program`` (which has the bound state-var
        name; ``__set_name__`` doesn't run until the class body
        finishes, so we can't materialise here yet)."""
        _check_return_annotation(fn, self, f"@{self.name}.next")
        self._next_fn = fn
        return fn

    def initializer(self, fn: Callable[[], Any]) -> Callable[[], Any]:
        """``@<sv>.initializer def _(): ...`` — register an initializer
        that runs under the handler stack (e.g. to construct objects)."""
        _check_return_annotation(
            fn, self, f"initializer for state var {self.name!r}"
        )
        self.init_fn = fn
        return fn


@dataclass
class OnClause:
    predicate: Any
    body: Callable[[], Any]
    name: str


@dataclass
class ProgramSpec:
    state_vars: list[StateVar]
    on_clauses: list[OnClause]
    obj_classes: list[Any]
    config: dict
    properties: list = field(default_factory=list)


# --------------------------------------------------------------------------
# prev() — read previous-tick value of a state var
# --------------------------------------------------------------------------

def prev(ref) -> Any:
    """Read the previous-tick value of a state var.

    `ref` may be: (a) a string name, (b) a `StateVar` spec record (the
    object `@program` mints — typically accessed as ``MyProgram.x``).
    """
    if isinstance(ref, str):
        return get_prev_var(ref)
    if isinstance(ref, StateVar):
        if ref.name is None:
            raise TypeMismatch(
                "prev() called on an unbound StateVar (no __set_name__ ran). "
                "Did you forget to put it inside a class body?"
            )
        return get_prev_var(ref.name)
    raise TypeError(f"prev() expects a str or StateVar, got {type(ref).__name__}")


# --------------------------------------------------------------------------
# transition_of: the shared next-phase transition term.
# --------------------------------------------------------------------------

def transition_of(
    sv: "StateVar", *, validate: Callable[[Any], None] | None = None
) -> Callable[[], Any]:
    """The next-phase transition for ``sv``: evaluate its next-clause and
    commit the result via the ``set_var`` op. The single definition both the
    runtime and the gate (``select_ast`` for ``.next``) run, so the gate
    analyses the term the runtime executes.

    ``validate``, if given, runs on the value before it is committed — the
    runtime passes its type-check, so a mistyped next-expression raises
    before committing; the gate omits it. See
    ``drafts/refactor-design-notes.md``.
    """
    if sv._next_fn is None:
        raise ValueError(f"state var {sv.name!r} has no next clause")
    next_fn = sv._next_fn
    name = sv.name

    def transition() -> Any:
        value = next_fn()
        if validate is not None:
            validate(value)
        set_var(name, value)
        return value

    transition.__name__ = f"{name}_next_transition"
    return transition


# --------------------------------------------------------------------------
# @on: register an on-clause.
# --------------------------------------------------------------------------

_pending_on_clauses: list["OnClause"] = []


def on(predicate):
    """Decorate a 0-arg function to register it as an on-clause body."""

    def decorator(fn: Callable[[], Any]) -> Callable[[], Any]:
        fn.__autumn_on_pred__ = predicate  # type: ignore[attr-defined]
        _pending_on_clauses.append(
            OnClause(predicate=predicate, body=fn, name=getattr(fn, "__name__", "_"))
        )
        return fn

    return decorator


# --------------------------------------------------------------------------
# @obj: decorate a class to turn it into an Autumn object factory.
# --------------------------------------------------------------------------

class AutumnObj:
    """Base class for ``@obj`` classes — a type-checker shim so
    ``Player(Position(8, 14))`` is seen as returning ``ObjectInstance``
    (mypy/Pyright otherwise read the bare class and reject the call; mypy
    #3135). Neither stub runs: ``@obj`` replaces the class with a factory
    that builds ``ObjectInstance`` directly.
    """
    def __new__(cls, *args: object, **kwargs: object) -> "ObjectInstance":  # type: ignore[misc]
        ...
    def __init__(self, *args: object, **kwargs: object) -> None: ...


def obj(cls) -> Callable[..., ObjectInstance]:
    """Class decorator: registers an Autumn object class.

    The decorated class body may declare:

    * ``cell = Cell(...)`` — single-cell object.
    * ``cells = [Cell(...), ...]`` — multi-cell.
    * Typed annotations (``bullets: int``, ``living: bool``, etc.) —
      named per-instance fields. Declaration order is the positional
      argument order for construction.

    Returns a callable factory ``(*fields, origin) -> ObjectInstance``.
    The annotated return type tells type-checkers the factory accepts
    any positional args (the actual signature is dynamic, derived from
    the class's annotations).
    """
    raw_annotations = cls.__dict__.get("__annotations__") or {}
    field_names = tuple(raw_annotations.keys())
    try:
        annotations = typing.get_type_hints(cls)
    except (TypeError, NameError):
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
# @program: walks class-body annotations + defaults to build the spec.
# --------------------------------------------------------------------------

@dataclass_transform(field_specifiers=(field,))
def program(**config):
    """Class decorator. Collects the ``StateVar`` descriptors declared in
    the class body and drains pending ``@on`` registrations into
    ``cls._autumn_spec``::

        @program(grid_size=16)
        class MyGame:
            step_count = StateVar(int, init=0)
            enemies = StateVar(list, init=[])

            @step_count.next
            def _():
                return prev(MyGame.step_count) + 1

    Annotated attributes that aren't ``StateVar`` instances are rejected,
    so it is unambiguous what's a state var vs. a class-level constant.
    """

    def decorator(cls):
        try:
            annotations = typing.get_type_hints(cls)
        except (TypeError, NameError):
            annotations = cls.__dict__.get("__annotations__") or {}

        state_vars: list[StateVar] = []
        seen_names: set[str] = set()

        # Collect StateVar instances; annotated non-StateVar attrs are
        # rejected below so state vars stay explicit.
        for name, val in list(cls.__dict__.items()):
            if isinstance(val, StateVar):
                state_vars.append(val)
                seen_names.add(val.name or name)

        # Reject annotated attributes that aren't wrapped in StateVar.
        for name, ty in annotations.items():
            if name in seen_names:
                continue
            if name not in cls.__dict__:
                continue
            default = cls.__dict__[name]
            if isinstance(default, StateVar) or hasattr(default, "__autumn_obj_spec__"):
                continue
            raise TypeError(
                f"@program {cls.__name__!r}: attribute {name!r} is annotated "
                f"({getattr(ty, '__name__', ty)!r}) with default {default!r} "
                f"but is not a StateVar. State vars must be declared as "
                f"`{name} = StateVar({getattr(ty, '__name__', ty)}, init=...)`. "
                f"For non-state-var class attributes, omit the annotation."
            )

        # Materialise spec goals from __autumn_spec__ on next-clause bodies.
        # Done here, not at @<sv>.next time, because names bind via
        # __set_name__ only after the class body finishes.
        from .properties import _pending_properties, realize_spec_goals
        for sv in state_vars:
            if sv._next_fn is None:
                continue
            autumn_spec = getattr(sv._next_fn, "__autumn_spec__", None)
            if autumn_spec is not None:
                _pending_properties.extend(
                    realize_spec_goals(autumn_spec, f"{sv.name}.next")
                )

        # Collect object factories.
        obj_classes = [v for v in cls.__dict__.values()
                       if hasattr(v, "__autumn_obj_spec__")]

        # Drain pending property goals registered above.
        properties = list(_pending_properties)
        _pending_properties.clear()

        spec = ProgramSpec(
            state_vars=state_vars,
            on_clauses=list(_pending_on_clauses),
            obj_classes=obj_classes,
            config=dict(config),
            properties=properties,
        )
        _pending_on_clauses.clear()
        cls._autumn_spec = spec
        return cls

    return decorator


def _check_return_annotation(fn: Callable, sv: StateVar, context: str) -> None:
    """If ``fn`` has a return annotation, check it's compatible with the
    state var's declared type. ``context`` names the site for the error."""
    if sv.type_ is None or sv.type_ is object:
        return
    try:
        hints = typing.get_type_hints(fn)
    except (TypeError, NameError):
        return
    ret = hints.get("return")
    if ret is None or ret is object or ret is Any:
        return
    if ret is sv.type_:
        return
    sv_origin = typing.get_origin(sv.type_)
    ret_origin = typing.get_origin(ret)
    if sv_origin is not None and sv_origin is ret_origin:
        return
    raise TypeMismatch(
        f"{context}: function return type "
        f"{getattr(ret, '__name__', ret)!r} does not match "
        f"declared type {getattr(sv.type_, '__name__', sv.type_)!r}"
    )
