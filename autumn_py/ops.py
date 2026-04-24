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
