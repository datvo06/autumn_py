from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

from effectful.ops.syntax import ObjectInterpretation, implements

from ..ops import get_var, set_var


class StateHandler(ObjectInterpretation):
    """Owns the live mutable environment for a Runtime instance.

    Writes land in ``_globals`` two ways, distinguished only by whether
    they mark ``on_writes_this_tick`` (which suppresses the matching
    next-expression this tick):

    * **untracked** — init seeding (``write``), plus every ``set_var`` op
      write (the ``_set`` handler): next-expr commits and the sibling-var
      writes a next-expression may make. Autumn permits next-exprs to write
      sibling vars; as in the C++ semantics, those do not suppress the
      default next update. (On-clause writes are buffered separately —
      ``WriteBufferHandler`` → ``apply_buffered`` — so they don't reach
      ``_set``.)
    * **tracked** — on-clause writes flushed at end of the on-phase
      (``apply_buffered``), which add to ``on_writes_this_tick``.
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

    def write(self, name: str, value: Any) -> None:
        """Direct write used by the runtime for init seeding. (Next-expr
        commits route through the ``set_var`` op — see ``_set`` — not here.)
        Does not touch ``on_writes_this_tick``."""
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
