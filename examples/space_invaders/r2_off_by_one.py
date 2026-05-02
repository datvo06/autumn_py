# mypy: disable-error-code="misc"
"""Round 2 emit, off-by-one: deterministic, but ``next_spawn_step``
init=4 (synth read challenger's "step 4" literally instead of inferring
``t mod 15 == 3``).

Passes P_1 (no `sample_uniform` in spawn-event next-clause).
Fails P_3 — Z3 returns ``t = 3`` as the counterexample tick where the
goal demands a spawn but the emit predicts none.
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
class SpaceInvadersR2OffByOne:
    step_count = StateVar(int, init=0)
    next_spawn_step = StateVar(int, init=4)
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
        return prev(SpaceInvadersR2OffByOne.step_count) + 1

    @next_spawn_step.next
    @symbolic
    def _() -> int:
        cur_step = SpaceInvadersR2OffByOne.step_count.get()
        cur_spawn = prev(SpaceInvadersR2OffByOne.next_spawn_step)
        if cur_step == cur_spawn:
            return cur_spawn + 15
        else:
            return cur_spawn

    @spawn_event.next
    def _() -> bool:
        return (SpaceInvadersR2OffByOne.step_count.get()
                == prev(SpaceInvadersR2OffByOne.next_spawn_step))

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
