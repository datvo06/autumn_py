from __future__ import annotations

from .ops import get_click_pos, is_event_active
from .values import ObjectInstance


class _EventSentinel:
    """Callable sentinel that resolves via is_event_active(name) when evaluated.

    * ``bool(clicked)`` / ``clicked()`` → True if the event fired this tick.
    * ``clicked(obj)`` → True if the click position falls on any cell of
      ``obj``. Other event sentinels ignore the ``obj`` argument and fall
      back to the boolean form.
    """

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, obj: ObjectInstance | None = None) -> bool:
        if obj is None:
            return bool(is_event_active(self.name))
        if self.name != "clicked":
            return bool(is_event_active(self.name))
        if not is_event_active("clicked"):
            return False
        pos = get_click_pos()
        if pos is None:
            return False
        cx, cy = pos
        for c in obj.rendered_cells():
            if c.x == cx and c.y == cy:
                return True
        return False

    def __bool__(self) -> bool:
        return bool(is_event_active(self.name))

    def __repr__(self) -> str:
        return f"<event:{self.name}>"


clicked = _EventSentinel("clicked")
left = _EventSentinel("left")
right = _EventSentinel("right")
up = _EventSentinel("up")
down = _EventSentinel("down")


class _Click:
    """Pseudo-object exposing click.x / click.y via get_click_pos()."""

    @property
    def x(self) -> int:
        pos = get_click_pos()
        return pos[0] if pos is not None else 0

    @property
    def y(self) -> int:
        pos = get_click_pos()
        return pos[1] if pos is not None else 0

    def __repr__(self) -> str:
        return "<click>"


click = _Click()

EVENT_NAMES: tuple[str, ...] = ("clicked", "left", "right", "up", "down")
