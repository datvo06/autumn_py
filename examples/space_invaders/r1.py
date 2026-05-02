# mypy: disable-error-code="misc"
"""Round 1 emit: stochastic spawn rule.

Spawn timing is drawn fresh each tick from ``sample_uniform(range(1, 21))``.
Fails P_1 (the spawn-event next-clause's read-set contains the
``("sample_uniform",)`` atom).
"""
from __future__ import annotations

from autumn_py import (
    Position,
    StateVar,
    clicked,
    left,
    on,
    prev,
    program,
    right,
)
from autumn_py._ast_rewrite import symbolic
from autumn_py.ops import sample_uniform
from autumn_py.stdlib import (
    addObj,
    moveDown,
    moveLeftNoCollision,
    moveRightNoCollision,
    moveUp,
    removeObj,
    updateObj,
)

from ._common import EnemyBullet, Player, PlayerBullet, initial_enemies


@program(grid_size=16)
class SpaceInvadersR1:
    step_count = StateVar(int, init=0)
    spawn_event = StateVar(bool, init=False)
    player = StateVar(object)
    enemies = StateVar(list, init=[])
    enemy_bullets = StateVar(list, init=[])
    player_bullets = StateVar(list, init=[])

    @player.initializer
    def _() -> object:
        return Player(Position(8, 14))

    @enemies.initializer
    def _() -> list:
        return initial_enemies()

    @step_count.next
    def _() -> int:
        return prev(SpaceInvadersR1.step_count) + 1

    @spawn_event.next
    def _() -> bool:
        spawn_step = sample_uniform(tuple(range(1, 21)))
        return SpaceInvadersR1.step_count.get() == spawn_step

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
        SpaceInvadersR1.player.set(
            moveLeftNoCollision(prev(SpaceInvadersR1.player))
        )

    @on(right)
    def _():
        SpaceInvadersR1.player.set(
            moveRightNoCollision(prev(SpaceInvadersR1.player))
        )

    @on(clicked)
    def _():
        p = prev(SpaceInvadersR1.player)
        SpaceInvadersR1.player_bullets.set(
            addObj(prev(SpaceInvadersR1.player_bullets), PlayerBullet(p.origin))
        )
