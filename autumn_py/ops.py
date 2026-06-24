from collections.abc import Sequence
from typing import Any

from effectful.ops.syntax import defop
from effectful.ops.types import NotHandled

from .values import ObjectInstance, Position


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
def sample_uniform[T](xs: Sequence[T]) -> T:
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
def all_objs() -> list[ObjectInstance]:
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
# Structural list combinators as effects. addObj / removeObj / updateObj are
# plain Python over these, so handlers only interpret the combinators.
# --------------------------------------------------------------------------

@defop
def map_op[T](xs: list[T], fn) -> list[T]:
    """Map fn over xs. Default: standard Python map."""
    return [fn(x) for x in xs]


@defop
def filter_op[T](xs: list[T], pred) -> list[T]:
    """Keep items of xs where pred(x) is truthy."""
    return [x for x in xs if pred(x)]


@defop
def concat_op[T](xs: list[T], ys: list[T]) -> list[T]:
    """xs ++ ys."""
    return [*xs, *ys]


@defop
def adjPositions_op(p: Position) -> list[Position]:
    """Cardinal neighbours of position p as a list of Positions.
    A list *constructor*, not a transformer — kept as its own op."""
    return [
        Position(p.x + 1, p.y),
        Position(p.x - 1, p.y),
        Position(p.x, p.y + 1),
        Position(p.x, p.y - 1),
    ]


@defop
def if_then_else(cond, then_branch, else_branch):
    """Three-way conditional op. Default: ``then_branch`` if ``cond`` else
    ``else_branch``; under ``SmtCollectHandler`` it lowers to ``z3.If``.

    **Both branches are evaluated eagerly** (function-call, not `if/else`,
    semantics), so callers must not put side effects in only one branch.
    """
    if cond:
        return then_branch
    return else_branch
