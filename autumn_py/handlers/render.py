from __future__ import annotations

from effectful.ops.syntax import ObjectInterpretation, implements

from ..ops import emit_render_cell
from ..values import Cell


class RenderHandler(ObjectInterpretation):
    def __init__(self) -> None:
        self.cells: list[Cell] = []

    @implements(emit_render_cell)
    def _emit(self, cell: Cell) -> None:
        self.cells.append(cell)

    def reset(self) -> None:
        self.cells.clear()
