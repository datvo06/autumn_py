from __future__ import annotations

from typing import Mapping

from effectful.ops.types import Interpretation

from ..ops import get_click_pos, is_event_active


def make_event_intp(
    active: frozenset[str],
    click_pos: tuple[int, int] | None,
) -> Interpretation:
    """Return a one-shot Interpretation for this tick's events.

    Events are first-class data: a frozen (active set, click_pos) pair. The
    Runtime materializes this by coproducting it onto the base handler stack at
    the top of step(), and it evaporates at the end of the with-block.
    """

    def _is_active(name: str) -> bool:
        return name in active

    def _click_pos() -> tuple[int, int] | None:
        return click_pos

    return {
        is_event_active: _is_active,
        get_click_pos: _click_pos,
    }
