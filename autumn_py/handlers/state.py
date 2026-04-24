from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

from effectful.ops.syntax import ObjectInterpretation, implements

from ..ops import get_var, set_var


class StateHandler(ObjectInterpretation):
    """Owns the live mutable environment for a Runtime instance.

    Three distinct write paths, each with distinct tracking semantics:

    * **seed** — init-phase, does not touch any overwrite set.
    * **apply_buffered** — on-clause writes flushed at end of on-phase.
      Populates ``on_writes_this_tick`` so the matching next-expr skips.
    * **commit_next** — next-expr result committed at end of next-phase.
      Does not touch ``on_writes_this_tick``.

    Free-form ``set_var`` calls (e.g. inside next-expressions that do sibling
    writes, which Autumn permits) go through ``_set`` and write directly to
    state *without* populating ``on_writes_this_tick`` — matching the C++
    semantics where only on-clause writes suppress the default next update.
    """

    def __init__(self) -> None:
        self._globals: dict[str, Any] = {}
        self.on_writes_this_tick: set[str] = set()

    # --- Effectful op impls -------------------------------------------------

    @implements(get_var)
    def _get(self, name: str) -> Any:
        try:
            return self._globals[name]
        except KeyError as e:
            raise NameError(f"Autumn state variable {name!r} is not defined") from e

    @implements(set_var)
    def _set(self, name: str, value: Any) -> None:
        # Free-form writes (e.g. from next-expression sibling assignments).
        # Does NOT mark the name as on-clause-overwritten; the skip check
        # in the next-phase uses on_writes_this_tick.
        self._globals[name] = value

    # --- Runtime-facing write API (bypasses the set_var op) -----------------

    def seed(self, name: str, value: Any) -> None:
        """Init-phase write."""
        self._globals[name] = value

    def commit_next(self, name: str, value: Any) -> None:
        """Next-expr commit."""
        self._globals[name] = value

    def apply_buffered(self, name: str, value: Any) -> None:
        """On-clause write-buffer flush. Marks the name so the matching
        next-expression skips this tick."""
        self._globals[name] = value
        self.on_writes_this_tick.add(name)

    # --- Read API -----------------------------------------------------------

    def get(self, name: str) -> Any:
        return self._globals[name]

    def has(self, name: str) -> bool:
        return name in self._globals

    def freeze_snapshot(self) -> Mapping[str, Any]:
        return MappingProxyType(dict(self._globals))

    # --- Tick-boundary helpers ---------------------------------------------

    def reset_on_writes(self) -> None:
        self.on_writes_this_tick.clear()
