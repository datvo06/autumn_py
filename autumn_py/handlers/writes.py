from __future__ import annotations

from typing import Any, TYPE_CHECKING

from effectful.ops.syntax import ObjectInterpretation, implements

from ..ops import set_var

if TYPE_CHECKING:
    from .state import StateHandler


class WriteBufferHandler(ObjectInterpretation):
    """Installed only during on-clause dispatch.

    Intercepts set_var and records writes; flush() replays them in
    declaration order into the underlying state via ``apply_buffered``,
    which marks each name in ``on_writes_this_tick`` so the corresponding
    next-expression skips.
    """

    def __init__(self) -> None:
        self.pending: list[tuple[str, Any]] = []

    @implements(set_var)
    def _capture(self, name: str, value: Any) -> None:
        self.pending.append((name, value))

    def flush(self, state_handler: "StateHandler") -> None:
        for name, value in self.pending:
            state_handler.apply_buffered(name, value)
        self.pending.clear()
