from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class Position:
    x: int
    y: int

    def __add__(self, other: Position) -> Position:
        return Position(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Position) -> Position:
        return Position(self.x - other.x, self.y - other.y)


@dataclass(frozen=True, slots=True)
class Cell:
    """A grid cell relative to an object's origin.

    ``color`` may be either a string or a callable ``fn(instance) -> str`` so
    object appearance can depend on the instance's field values (e.g. Game of
    Life's Particle whose color tracks ``living``).
    """
    x: int
    y: int
    color: str | Callable[["ObjectInstance"], str]

    def resolve_color(self, inst: "ObjectInstance") -> str:
        c = self.color(inst) if callable(self.color) else self.color
        # §2.3: cell color expressions are typed Str. Enforce at render time.
        if not isinstance(c, str):
            from .api import TypeMismatch  # local import: api ↔ values cycle
            raise TypeMismatch(
                f"cell color of {inst.cls.name} resolved to "
                f"{type(c).__name__} (value: {c!r}); expected str"
            )
        return c


@dataclass(frozen=True, slots=True)
class ObjectClassSpec:
    """Metadata collected from @obj class decoration."""
    name: str
    cells: tuple[Cell, ...]
    field_names: tuple[str, ...]


@dataclass(frozen=True)
class ObjectInstance:
    """Runtime instance of an Autumn object.

    ``fields`` holds the named mutable attributes declared on the class
    (Mario's ``bullets``, Enemy's ``movingLeft``/``lives``, etc.). They are
    readable as ordinary attributes — ``inst.bullets`` — via ``__getattr__``.
    """
    cls: ObjectClassSpec
    origin: Position
    id: int
    fields: dict[str, Any] = field(default_factory=dict)
    alive: bool = True

    def __getattr__(self, name: str) -> Any:
        # Called only when normal attribute lookup misses. Avoid recursion:
        # ``fields`` itself is set by dataclass __init__, so if that attribute
        # missed here we'd hit infinite recursion. Use object.__getattribute__.
        try:
            fields = object.__getattribute__(self, "fields")
        except AttributeError:
            raise AttributeError(name)
        if name in fields:
            return fields[name]
        raise AttributeError(name)

    def rendered_cells(self) -> list[Cell]:
        origin = self.origin
        return [
            Cell(c.x + origin.x, c.y + origin.y, c.resolve_color(self))
            for c in self.cls.cells
        ]

    def with_origin(self, origin: Position) -> "ObjectInstance":
        return replace(self, origin=origin)

    def with_field(self, name: str, value: Any) -> "ObjectInstance":
        return replace(self, fields={**self.fields, name: value})

    def killed(self) -> "ObjectInstance":
        return replace(self, alive=False)


def cell_to_dict(c: Cell) -> dict:
    """Serialization helper. Cell itself no longer carries to_dict since
    ``color`` may be callable until rendered through an ObjectInstance."""
    return {"x": c.x, "y": c.y, "color": c.color}
