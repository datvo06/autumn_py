from typing import Any

from effectful.ops.syntax import defop
from effectful.ops.types import NotHandled


@defop
def get_var(name: str) -> Any:
    raise NotHandled


@defop
def set_var(name: str, value: Any) -> None:
    raise NotHandled


@defop
def get_prev_var(name: str) -> Any:
    raise NotHandled


@defop
def sample_uniform(xs: tuple) -> Any:
    raise NotHandled


@defop
def alloc_obj_id() -> int:
    raise NotHandled


@defop
def is_event_active(name: str) -> bool:
    raise NotHandled


@defop
def get_click_pos() -> tuple[int, int] | None:
    raise NotHandled


@defop
def emit_render_cell(cell: Any) -> None:
    raise NotHandled


@defop
def all_objs() -> list:
    """Return all alive ObjectInstances across every state var."""
    raise NotHandled


@defop
def grid_size() -> int:
    """Return the configured grid size (default 16)."""
    raise NotHandled


@defop
def state_has(name: str) -> bool:
    """Return True iff the state var is defined and its value is truthy
    in Autumn's sense (ObjectInstance alive, or any non-None value)."""
    raise NotHandled


# --------------------------------------------------------------------------
# Structural list combinators as effects.
#
# Higher-level Autumn primitives (addObj, removeObj, updateObj) are written
# as plain Python on top of these — handlers (TypeOfHandler today, future
# symbolic / reduction handlers) only need to interpret the combinators.
#
# The defaults are standard list semantics; type-domain interpretations
# preserve the list type, which is sound for Autumn's structural uses
# (where the function's domain and codomain coincide).
# --------------------------------------------------------------------------

@defop
def map_op(xs, fn):
    """Map fn over xs. Default: standard Python map."""
    return [fn(x) for x in xs]


@defop
def filter_op(xs, pred):
    """Keep items of xs where pred(x) is truthy."""
    return [x for x in xs if pred(x)]


@defop
def concat_op(xs, ys):
    """xs ++ ys."""
    return [*xs, *ys]


@defop
def adjPositions_op(p):
    """Cardinal neighbours of position p as a list of Positions.
    A list *constructor*, not a transformer — kept as its own op."""
    from .values import Position
    return [
        Position(p.x + 1, p.y),
        Position(p.x - 1, p.y),
        Position(p.x, p.y + 1),
        Position(p.x, p.y - 1),
    ]
