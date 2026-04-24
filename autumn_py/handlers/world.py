from __future__ import annotations

from typing import TYPE_CHECKING

from effectful.ops.syntax import ObjectInterpretation, implements

from ..ops import all_objs, grid_size, state_has
from ..values import ObjectInstance

if TYPE_CHECKING:
    from ..runtime import Runtime


class WorldHandler(ObjectInterpretation):
    """Implements whole-world queries: allObjs, grid_size, state_has.

    Holds a back-reference to the Runtime so it can iterate state vars and
    read config. Installed once at Runtime construction alongside the other
    persistent handlers.
    """

    def __init__(self, runtime: "Runtime") -> None:
        self.runtime = runtime

    @implements(all_objs)
    def _all(self) -> list[ObjectInstance]:
        out: list[ObjectInstance] = []
        state = self.runtime.state
        for sv in self.runtime.spec.state_vars:
            if not state.has(sv.name):
                continue
            val = state.get(sv.name)
            if isinstance(val, ObjectInstance):
                if val.alive:
                    out.append(val)
            elif isinstance(val, (list, tuple)):
                for v in val:
                    if isinstance(v, ObjectInstance) and v.alive:
                        out.append(v)
        return out

    @implements(grid_size)
    def _grid(self) -> int:
        return int(self.runtime.spec.config.get("grid_size", 16))

    @implements(state_has)
    def _has(self, name: str) -> bool:
        state = self.runtime.state
        if not state.has(name):
            return False
        v = state.get(name)
        if isinstance(v, ObjectInstance):
            return v.alive
        return v is not None
