from __future__ import annotations

import random
from typing import Any, Iterable

from effectful.ops.syntax import ObjectInterpretation, implements

from ..ops import sample_uniform


class NativeRandomHandler(ObjectInterpretation):
    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    @implements(sample_uniform)
    def _choice(self, xs: Any) -> Any:
        seq = list(xs) if not isinstance(xs, (list, tuple)) else xs
        if not seq:
            raise ValueError("sample_uniform called on empty sequence")
        return self.rng.choice(seq)


class ReplayRandomHandler(ObjectInterpretation):
    """Consumes a pre-recorded sequence of draws; raises when exhausted.

    Used for deterministic tests and (later) C++-parity diffing.
    """

    def __init__(self, draws: Iterable[Any]) -> None:
        self._draws = list(draws)
        self._i = 0

    @implements(sample_uniform)
    def _replay(self, xs: Any) -> Any:  # xs ignored by design
        if self._i >= len(self._draws):
            raise RuntimeError("ReplayRandomHandler draws exhausted")
        draw = self._draws[self._i]
        self._i += 1
        return draw
