from __future__ import annotations

from effectful.ops.syntax import ObjectInterpretation, implements

from ..ops import alloc_obj_id


class ObjectAllocHandler(ObjectInterpretation):
    def __init__(self) -> None:
        self._next = 0

    @implements(alloc_obj_id)
    def _alloc(self) -> int:
        i = self._next
        self._next += 1
        return i
