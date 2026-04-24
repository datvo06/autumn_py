from __future__ import annotations

from typing import Any, Mapping

from effectful.ops.syntax import ObjectInterpretation, implements

from ..ops import get_prev_var


class PrevStateHandler(ObjectInterpretation):
    """Read-only view of the previous-tick snapshot.

    Installed for the duration of a single step() call, then popped. Because the
    snapshot is an immutable Mapping, prev-reads are pure op dispatch — no
    aliasing with the live state.
    """

    def __init__(self, snapshot: Mapping[str, Any]) -> None:
        self.snapshot = snapshot

    @implements(get_prev_var)
    def _get_prev(self, name: str) -> Any:
        try:
            return self.snapshot[name]
        except KeyError as e:
            raise NameError(f"no previous value for {name!r}") from e
