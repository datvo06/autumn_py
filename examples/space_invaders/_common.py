"""Shared object classes and helpers for the three `space_invaders` rounds.

The player ship, enemy formation, and bullet types are identical across
rounds. The three rounds differ only in the spawn-timing slice — the
spawn-event next-clause and (for round 2 forms) the `next_spawn_step`
state var. Each round's program file in this package imports from
`_common` for the shared @obj classes and the initial-formation helper.
"""
from __future__ import annotations

from autumn_py import AutumnObj, Cell, Position, obj


@obj
class Player(AutumnObj):
    cell = Cell(0, 0, "blue")


@obj
class Enemy(AutumnObj):
    cell = Cell(0, 0, "red")


@obj
class EnemyBullet(AutumnObj):
    cell = Cell(0, 0, "yellow")


@obj
class PlayerBullet(AutumnObj):
    cell = Cell(0, 0, "lightgreen")


def initial_enemies() -> list:
    """Static 3-row × 4-column formation across the top of the grid."""
    return [Enemy(Position(c, r)) for r in (1, 2, 3) for c in (4, 6, 8, 10)]
