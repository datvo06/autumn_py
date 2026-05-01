"""Three rounds of `space_invaders` synth emit.

Each is a runnable autumn-py program with a player ship, a static
3×4 enemy formation, descending enemy bullets, and ascending player
bullets fired on `clicked`. The classes differ *only* in the spawn-
timing slice (the spawn-event next-clause and, for round 2 onward,
the `next_spawn_step` StateVar):

* :class:`SpaceInvadersR1` — round 1's stochastic spawn rule
  (``sample_uniform(range(1, 21))`` per tick). Fails P_1.

* :class:`SpaceInvadersR2OffByOne` — round 2 deterministic-but-off-by-one
  (``next_spawn_step`` init=4, increments by 15). Passes P_1; fails P_3.

* :class:`SpaceInvadersR2Fixed` — round 2 fixed (init=3, the SMT
  counterexample's witness). Passes both P_1 and P_3.

The gate's $\\varphi_1$ and $\\varphi_2$ in `tests/test_gate.py` anchor on
``"spawn_event.next"``; the player/enemy/bullet machinery doesn't change
those checks but makes each program runnable end-to-end via
``Runtime(SpaceInvadersR2Fixed)``.
"""
from __future__ import annotations

from autumn_py import (
    Cell,
    Position,
    StateVar,
    clicked,
    left,
    obj,
    on,
    prev,
    program,
    right,
)
from autumn_py._ast_rewrite import symbolic
from autumn_py.ops import get_var, sample_uniform
from autumn_py.stdlib import (
    addObj,
    moveDown,
    moveLeftNoCollision,
    moveRightNoCollision,
    moveUp,
    removeObj,
    updateObj,
)


# -------------------------------------------------------------------------
# Object classes (shared across all three rounds)
# -------------------------------------------------------------------------

@obj
class Player:
    cell = Cell(0, 0, "blue")


@obj
class Enemy:
    cell = Cell(0, 0, "red")


@obj
class EnemyBullet:
    cell = Cell(0, 0, "yellow")


@obj
class PlayerBullet:
    cell = Cell(0, 0, "lightgreen")


def _initial_enemies() -> list:
    """Static 3-row × 4-column formation across the top of the grid."""
    return [Enemy(Position(c, r)) for r in (1, 2, 3) for c in (4, 6, 8, 10)]


# -------------------------------------------------------------------------
# Round 1 — stochastic spawn rule. Fails P_1.
# -------------------------------------------------------------------------

@program(grid_size=16)
class SpaceInvadersR1:
    """Round-1 emit: spawn timing is drawn fresh each tick from
    ``sample_uniform(range(1, 21))``."""

    step_count: int = StateVar(int, init=0)
    spawn_event: bool = StateVar(bool, init=False)

    player = StateVar(object)
    enemies = StateVar(list, init=[])
    enemy_bullets = StateVar(list, init=[])
    player_bullets = StateVar(list, init=[])

    @player.initializer
    def _():
        return Player(Position(8, 14))

    @enemies.initializer
    def _():
        return _initial_enemies()

    @step_count.next
    def _() -> int:
        return prev(SpaceInvadersR1.step_count) + 1

    @spawn_event.next
    def _() -> bool:
        spawn_step = sample_uniform(tuple(range(1, 21)))
        return get_var("step_count") == spawn_step

    @enemies.next
    def _() -> list:
        return prev(SpaceInvadersR1.enemies)

    @enemy_bullets.next
    @symbolic
    def _() -> list:
        descended = updateObj(prev(SpaceInvadersR1.enemy_bullets), moveDown)
        descended = removeObj(descended, lambda b: b.origin.y > 15)
        if prev(SpaceInvadersR1.spawn_event):
            return addObj(descended, EnemyBullet(Position(8, 1)))
        else:
            return descended

    @player_bullets.next
    def _() -> list:
        ascended = updateObj(prev(SpaceInvadersR1.player_bullets), moveUp)
        return removeObj(ascended, lambda b: b.origin.y < 0)

    @on(left)
    def _():
        SpaceInvadersR1.player.set(moveLeftNoCollision(prev(SpaceInvadersR1.player)))

    @on(right)
    def _():
        SpaceInvadersR1.player.set(moveRightNoCollision(prev(SpaceInvadersR1.player)))

    @on(clicked)
    def _():
        p = prev(SpaceInvadersR1.player)
        SpaceInvadersR1.player_bullets.set(
            addObj(prev(SpaceInvadersR1.player_bullets), PlayerBullet(p.origin))
        )


# -------------------------------------------------------------------------
# Round 2 — deterministic, off-by-one. Fails P_3.
# -------------------------------------------------------------------------

@program(grid_size=16)
class SpaceInvadersR2OffByOne:
    """Round-2 emit, off-by-one: ``next_spawn_step`` init=4 (synth read
    challenger's "step 4" literally instead of inferring t mod 15 == 3)."""

    step_count: int = StateVar(int, init=0)
    next_spawn_step: int = StateVar(int, init=4)
    spawn_event: bool = StateVar(bool, init=False)

    player = StateVar(object)
    enemies = StateVar(list, init=[])
    enemy_bullets = StateVar(list, init=[])
    player_bullets = StateVar(list, init=[])

    @player.initializer
    def _():
        return Player(Position(8, 14))

    @enemies.initializer
    def _():
        return _initial_enemies()

    @step_count.next
    def _() -> int:
        return prev(SpaceInvadersR2OffByOne.step_count) + 1

    @next_spawn_step.next
    @symbolic
    def _() -> int:
        cur_step = get_var("step_count")
        cur_spawn = prev(SpaceInvadersR2OffByOne.next_spawn_step)
        if cur_step == cur_spawn:
            return cur_spawn + 15
        else:
            return cur_spawn

    @spawn_event.next
    def _() -> bool:
        return get_var("step_count") == prev(SpaceInvadersR2OffByOne.next_spawn_step)

    @enemies.next
    def _() -> list:
        return prev(SpaceInvadersR2OffByOne.enemies)

    @enemy_bullets.next
    @symbolic
    def _() -> list:
        descended = updateObj(prev(SpaceInvadersR2OffByOne.enemy_bullets), moveDown)
        descended = removeObj(descended, lambda b: b.origin.y > 15)
        if prev(SpaceInvadersR2OffByOne.spawn_event):
            return addObj(descended, EnemyBullet(Position(8, 1)))
        else:
            return descended

    @player_bullets.next
    def _() -> list:
        ascended = updateObj(prev(SpaceInvadersR2OffByOne.player_bullets), moveUp)
        return removeObj(ascended, lambda b: b.origin.y < 0)

    @on(left)
    def _():
        SpaceInvadersR2OffByOne.player.set(
            moveLeftNoCollision(prev(SpaceInvadersR2OffByOne.player))
        )

    @on(right)
    def _():
        SpaceInvadersR2OffByOne.player.set(
            moveRightNoCollision(prev(SpaceInvadersR2OffByOne.player))
        )

    @on(clicked)
    def _():
        p = prev(SpaceInvadersR2OffByOne.player)
        SpaceInvadersR2OffByOne.player_bullets.set(
            addObj(prev(SpaceInvadersR2OffByOne.player_bullets), PlayerBullet(p.origin))
        )


# -------------------------------------------------------------------------
# Round 2 fixed — constant bound from SMT counterexample. Passes both.
# -------------------------------------------------------------------------

@program(grid_size=16)
class SpaceInvadersR2Fixed:
    """Round-2 emit after ``bind_constant_from_witness(3)``: spawn
    constant is now 3 (the SMT counterexample tick from P_3 on the
    off-by-one emit). Passes P_1 (no stochastic) and P_3 (modular
    timing matches t mod 15 == 3 across the bounded horizon)."""

    step_count: int = StateVar(int, init=0)
    next_spawn_step: int = StateVar(int, init=3)
    spawn_event: bool = StateVar(bool, init=False)

    player = StateVar(object)
    enemies = StateVar(list, init=[])
    enemy_bullets = StateVar(list, init=[])
    player_bullets = StateVar(list, init=[])

    @player.initializer
    def _():
        return Player(Position(8, 14))

    @enemies.initializer
    def _():
        return _initial_enemies()

    @step_count.next
    def _() -> int:
        return prev(SpaceInvadersR2Fixed.step_count) + 1

    @next_spawn_step.next
    @symbolic
    def _() -> int:
        cur_step = get_var("step_count")
        cur_spawn = prev(SpaceInvadersR2Fixed.next_spawn_step)
        if cur_step == cur_spawn:
            return cur_spawn + 15
        else:
            return cur_spawn

    @spawn_event.next
    def _() -> bool:
        return get_var("step_count") == prev(SpaceInvadersR2Fixed.next_spawn_step)

    @enemies.next
    def _() -> list:
        return prev(SpaceInvadersR2Fixed.enemies)

    @enemy_bullets.next
    @symbolic
    def _() -> list:
        descended = updateObj(prev(SpaceInvadersR2Fixed.enemy_bullets), moveDown)
        descended = removeObj(descended, lambda b: b.origin.y > 15)
        if prev(SpaceInvadersR2Fixed.spawn_event):
            return addObj(descended, EnemyBullet(Position(8, 1)))
        else:
            return descended

    @player_bullets.next
    def _() -> list:
        ascended = updateObj(prev(SpaceInvadersR2Fixed.player_bullets), moveUp)
        return removeObj(ascended, lambda b: b.origin.y < 0)

    @on(left)
    def _():
        SpaceInvadersR2Fixed.player.set(
            moveLeftNoCollision(prev(SpaceInvadersR2Fixed.player))
        )

    @on(right)
    def _():
        SpaceInvadersR2Fixed.player.set(
            moveRightNoCollision(prev(SpaceInvadersR2Fixed.player))
        )

    @on(clicked)
    def _():
        p = prev(SpaceInvadersR2Fixed.player)
        SpaceInvadersR2Fixed.player_bullets.set(
            addObj(prev(SpaceInvadersR2Fixed.player_bullets), PlayerBullet(p.origin))
        )
