"""Autumn native primitives + stdlib helpers, translated to Python.

Each function that needs the RNG or world queries dispatches through the
corresponding @defop; the rest is plain Python. Keep this file a thin
wrapper — if something belongs in the handler layer, put it there.
"""
from __future__ import annotations

import typing
from typing import TYPE_CHECKING, Any, Callable, Iterable

if TYPE_CHECKING:
    from .api import AutumnObj

from .ops import (
    adjPositions_op,
    all_objs,
    concat_op,
    filter_op,
    grid_size,
    map_op,
    sample_uniform,
    state_has,
)
from .values import ObjectInstance, Position


# ---------------------------------------------------------------------------
# Object-list helpers — composed from the structural combinators in ops.py
# (concat_op / filter_op / map_op) so handlers only need to interpret the
# combinators. Instance-form overloads stay direct.
# ---------------------------------------------------------------------------

@typing.overload
def addObj(xs: list, obj_or_list: ObjectInstance | "AutumnObj") -> list: ...
@typing.overload
def addObj(xs: list, obj_or_list: list) -> list: ...

def addObj(xs, obj_or_list):
    """Append to an object list. Second arg may be a single ObjectInstance or
    a list of them (matching Autumn's polymorphic addObj)."""
    if isinstance(obj_or_list, list):
        return concat_op(xs, obj_or_list)
    return concat_op(xs, [obj_or_list])


@typing.overload
def removeObj(first: ObjectInstance) -> ObjectInstance: ...
@typing.overload
def removeObj(first: list, target: ObjectInstance | "AutumnObj" | Callable) -> list: ...

def removeObj(first, target=None):
    """Overloaded:

    * ``removeObj(instance)`` → returns the instance marked dead.
    * ``removeObj(list, target_instance)`` → returns list with target omitted.
    * ``removeObj(list, predicate)`` → returns list with predicate-matching
      instances filtered out.
    """
    if target is None:
        if isinstance(first, ObjectInstance):
            return first.killed()
        raise TypeError(f"removeObj: missing target for {type(first).__name__}")
    if callable(target):
        return filter_op(first, lambda o: not target(o))
    return filter_op(first, lambda o: getattr(o, "id", None) != target.id)


@typing.overload
def updateObj(first: ObjectInstance, name: str, value: Any) -> ObjectInstance: ...
@typing.overload
def updateObj(first: list, fn: Callable, pred: Callable | None = ...) -> list: ...

def updateObj(first, *rest):
    """Overloaded:

    * ``updateObj(list, fn)`` → map fn over each alive instance.
    * ``updateObj(list, fn, predicate)`` → map fn over each alive instance
      where predicate(inst) is truthy; others pass through.
    * ``updateObj(instance, field_name, value)`` → return instance with
      that field replaced.
    """
    if isinstance(first, ObjectInstance):
        name, value = rest
        return first.with_field(name, value)
    fn = rest[0]
    if len(rest) == 1:
        return map_op(first, lambda o: fn(o) if o.alive else o)
    pred = rest[1]
    return map_op(first, lambda o: fn(o) if (o.alive and pred(o)) else o)


def allObjs() -> list[ObjectInstance]:
    """All alive ObjectInstances anywhere in state."""
    return all_objs()


# ---------------------------------------------------------------------------
# Position helpers
# ---------------------------------------------------------------------------

def allPositions(n: int | None = None) -> list[Position]:
    if n is None:
        n = grid_size()
    return [Position(x, y) for x in range(n) for y in range(n)]


def randomPositions(n_or_grid, count: int | None = None) -> list[Position]:
    """``randomPositions(grid_size, count)`` returns ``count`` Positions drawn
    uniformly with replacement from the grid. Matches ``ants.sexp``'s usage."""
    if count is None:
        # Single-arg form: randomPositions(count) — use configured grid.
        count = n_or_grid
        positions = allPositions()
    else:
        positions = allPositions(n_or_grid)
    return [sample_uniform(tuple(positions)) for _ in range(count)]


def adjPositions(p):
    """Cardinal neighbours of position p."""
    return adjPositions_op(p)


def displacement(p1: Position, p2: Position) -> Position:
    return Position(p2.x - p1.x, p2.y - p1.y)


def sqdist(p1: Position, p2: Position) -> int:
    dx, dy = p2.x - p1.x, p2.y - p1.y
    return dx * dx + dy * dy


def _sign(n: int) -> int:
    return (n > 0) - (n < 0)


def unitVector(obj: ObjectInstance, target: ObjectInstance | Position) -> Position:
    """Grid-unit step from obj's origin toward target (clamped to {-1, 0, 1})."""
    t = target.origin if isinstance(target, ObjectInstance) else target
    return Position(_sign(t.x - obj.origin.x), _sign(t.y - obj.origin.y))


def closest(obj: ObjectInstance, candidates: list) -> ObjectInstance | None:
    if not candidates:
        return None
    return min(candidates, key=lambda c: sqdist(obj.origin, _origin_of(c)))


def _origin_of(x: Any) -> Position:
    if isinstance(x, ObjectInstance):
        return x.origin
    if isinstance(x, Position):
        return x
    raise TypeError(f"_origin_of: {type(x).__name__}")


# ---------------------------------------------------------------------------
# Intersection / freeness
# ---------------------------------------------------------------------------

def _cells_of(x) -> list[tuple[int, int]]:
    if isinstance(x, ObjectInstance):
        return [(c.x, c.y) for c in x.rendered_cells()]
    if isinstance(x, Position):
        return [(x.x, x.y)]
    if isinstance(x, list):
        out: list[tuple[int, int]] = []
        for item in x:
            out.extend(_cells_of(item))
        return out
    raise TypeError(f"_cells_of: {type(x).__name__}")


def intersects(a, b) -> bool:
    """Any cell of ``a`` coincident with any cell of ``b``.

    Either argument may be an ObjectInstance, a Position, or a list of
    ObjectInstances.
    """
    a_cells = set(_cells_of(a))
    if not a_cells:
        return False
    b_cells = set(_cells_of(b))
    return bool(a_cells & b_cells)


def isWithinBounds(obj: ObjectInstance) -> bool:
    n = grid_size()
    for c in obj.rendered_cells():
        if not (0 <= c.x < n and 0 <= c.y < n):
            return False
    return True


def _pos_free_ignoring(pos: Position, ignore_ids: set[int]) -> bool:
    for o in allObjs():
        if o.id in ignore_ids:
            continue
        for c in o.rendered_cells():
            if c.x == pos.x and c.y == pos.y:
                return False
    return True


def isFreePos(pos: Position) -> bool:
    return _pos_free_ignoring(pos, ignore_ids=set())


def isFree(obj: ObjectInstance) -> bool:
    ids = {obj.id}
    for c in obj.rendered_cells():
        if not _pos_free_ignoring(Position(c.x, c.y), ids):
            return False
    return True


# ---------------------------------------------------------------------------
# Movement
# ---------------------------------------------------------------------------

def move(obj: ObjectInstance, delta: Position) -> ObjectInstance:
    return obj.with_origin(obj.origin + delta)


def moveLeft(obj: ObjectInstance) -> ObjectInstance:
    return obj.with_origin(Position(obj.origin.x - 1, obj.origin.y))


def moveRight(obj: ObjectInstance) -> ObjectInstance:
    return obj.with_origin(Position(obj.origin.x + 1, obj.origin.y))


def moveUp(obj: ObjectInstance) -> ObjectInstance:
    return obj.with_origin(Position(obj.origin.x, obj.origin.y - 1))


def moveDown(obj: ObjectInstance) -> ObjectInstance:
    return obj.with_origin(Position(obj.origin.x, obj.origin.y + 1))


def moveNoCollision(obj: ObjectInstance, dx: int, dy: int) -> ObjectInstance:
    moved = obj.with_origin(Position(obj.origin.x + dx, obj.origin.y + dy))
    if not isWithinBounds(moved):
        return obj
    if not isFree(moved):
        return obj
    return moved


def moveLeftNoCollision(obj: ObjectInstance) -> ObjectInstance:
    return moveNoCollision(obj, -1, 0)


def moveRightNoCollision(obj: ObjectInstance) -> ObjectInstance:
    return moveNoCollision(obj, 1, 0)


def moveUpNoCollision(obj: ObjectInstance) -> ObjectInstance:
    return moveNoCollision(obj, 0, -1)


def moveDownNoCollision(obj: ObjectInstance) -> ObjectInstance:
    return moveNoCollision(obj, 0, 1)


# ---------------------------------------------------------------------------
# Physics (granular / liquid)
# ---------------------------------------------------------------------------

def nextSolid(obj: ObjectInstance) -> ObjectInstance:
    """Alias for moveDownNoCollision — solid gravity."""
    return moveDownNoCollision(obj)


def nextLiquid(obj: ObjectInstance) -> ObjectInstance:
    """Fluid flow: move down if free, else flow toward the closest free cell
    in the next row (the "hole"). Matches upstream autumnstdlib semantics."""
    gs = grid_size()
    if obj.origin.y != gs - 1:
        below = moveDown(obj)
        if isFree(below):
            return below
    next_y = obj.origin.y + 1
    if next_y >= gs:
        return obj
    holes = [Position(x, next_y) for x in range(gs) if isFreePos(Position(x, next_y))]
    if not holes:
        return obj
    closest_hole = min(
        holes,
        key=lambda p: (p.x - obj.origin.x) ** 2 + (p.y - obj.origin.y) ** 2,
    )
    # Autumn's nextLiquidMoveClosestHole aims for the cell *above* the hole.
    target = Position(closest_hole.x, closest_hole.y - 1)
    step = Position(_sign(target.x - obj.origin.x), _sign(target.y - obj.origin.y))
    moved = obj.with_origin(obj.origin + step)
    if (
        isFreePos(target)
        and isFreePos(moved.origin)
        and isWithinBounds(moved)
    ):
        return moved
    return obj


def adjacentObjs(obj: ObjectInstance, unit_size: int) -> list[ObjectInstance]:
    """Returns every alive object whose origin is within manhattan distance
    ``unit_size`` of ``obj``'s origin. Excludes ``obj`` itself."""
    out: list[ObjectInstance] = []
    for o in allObjs():
        if o.id == obj.id:
            continue
        dx = abs(o.origin.x - obj.origin.x)
        dy = abs(o.origin.y - obj.origin.y)
        if dx + dy <= unit_size:
            out.append(o)
    return out


# ---------------------------------------------------------------------------
# Functional helpers (direct translations of Autumn names for port fidelity)
# ---------------------------------------------------------------------------

def uniformChoice(xs):
    """Sample uniformly from `xs`. Dispatches to `sample_uniform`; under
    TypeOfHandler this returns the element type (closes the §2.2 gap)."""
    if isinstance(xs, (list, tuple)):
        return sample_uniform(tuple(xs))
    return sample_uniform(xs)


def foldl(fn: Callable, init: Any, xs: Iterable) -> Any:
    acc = init
    for x in xs:
        acc = fn(acc, x)
    return acc


def head(xs: list) -> Any:
    return xs[0]


def last(xs: list) -> Any:
    """Autumn's ``tail`` returns the last element (contra Lisp convention).
    We expose it under the unambiguous name ``last`` to avoid the footgun."""
    return xs[-1]


def at(xs: list, i: int) -> Any:
    return xs[i]


def concat(xss: Iterable[Iterable]) -> list:
    out: list = []
    for xs in xss:
        out.extend(xs)
    return out


def in_(item: Any, xs: Iterable[Any]) -> bool:
    return item in xs


def length(xs) -> int:
    return len(xs)


def defined(state_var_or_name) -> bool:
    """True iff the named state var is defined and (for ObjectInstance values)
    still alive. Mirrors Autumn's ``(defined "x")``."""
    from .api import StateVar  # lazy to avoid import cycle
    name = state_var_or_name.name if isinstance(state_var_or_name, StateVar) else state_var_or_name
    return state_has(name)
