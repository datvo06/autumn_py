"""Three rounds of `space_invaders` synth emit, one per file.

Each is a runnable autumn-py program with a player ship, a static
3×4 enemy formation, descending enemy bullets, and ascending player
bullets fired on `clicked`. The classes differ *only* in the spawn-
timing slice (the spawn-event next-clause and, for round 2 forms, the
``next_spawn_step`` state var); the player/enemy/bullet machinery in
``_common.py`` is shared across all three.

* :class:`SpaceInvadersR1` (``r1.py``) — round 1's stochastic spawn rule.
  Fails P_1.
* :class:`SpaceInvadersR2OffByOne` (``r2_off_by_one.py``) — round 2 with
  ``next_spawn_step`` init=4. Passes P_1; fails P_3.
* :class:`SpaceInvadersR2Fixed` (``r2_fixed.py``) — round 2 with init=3
  (the SMT counterexample's witness). Passes both.
"""
from .r1 import SpaceInvadersR1
from .r2_fixed import SpaceInvadersR2Fixed
from .r2_off_by_one import SpaceInvadersR2OffByOne

__all__ = [
    "SpaceInvadersR1",
    "SpaceInvadersR2OffByOne",
    "SpaceInvadersR2Fixed",
]
