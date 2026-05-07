# mypy: disable-error-code="misc"
"""Round 2 emit fixed: spawn constant is now 3 (the SMT counterexample's
witness from P_3 on the off-by-one emit, applied via
``bind_constant_from_witness(3)``).

Passes both P_1 (no `sample_uniform`) and P_3 (modular timing matches
``t mod 15 == 3`` across the bounded horizon). The emit commits.
"""
from __future__ import annotations

from autumn_py import (
    Position,
    StateVar,
    clicked,
    left,
    no_stochastic,
    on,
    prev,
    program,
    right,
    spec,
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
class SpaceInvadersR2Fixed:
    step_count = StateVar(int, init=0)
    next_spawn_step = StateVar(int, init=3)
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
        return prev(SpaceInvadersR2Fixed.step_count) + 1

    @next_spawn_step.next
    @symbolic
    def _() -> int:
        cur_step = SpaceInvadersR2Fixed.step_count.get()
        cur_spawn = prev(SpaceInvadersR2Fixed.next_spawn_step)
        if cur_step == cur_spawn:
            return cur_spawn + 15
        else:
            return cur_spawn

    @spawn_event.next
    @spec(
        # The spawn-decision predicate must be deterministic — no stochastic
        # ops in the spawn-event next-clause's read-set.
        no_stochastic = True,
        # The spawn predicate writes spawn_event only.
        modifies = (spawn_event,),
        # Across the bounded horizon: spawn_event(t+1) iff (t mod 15 == 3).
        # Lambda parameters bind by name to the corresponding state vars'
        # Z3 functions; no funcs[] indirection needed.
        invariant = lambda step_count, next_spawn_step, spawn_event, t:
            spawn_event(t + 1) == ((t % 15) == 3),
        unroll = (next_spawn_step, spawn_event),
        init_constraints = lambda step_count, next_spawn_step, spawn_event: [
            *(step_count(k) == k for k in range(8)),
            next_spawn_step(0)  == 3,
            next_spawn_step(-1) == 3,
        ],
        horizon = 6,
    )
    def _() -> bool:
        return (SpaceInvadersR2Fixed.step_count.get()
                == prev(SpaceInvadersR2Fixed.next_spawn_step))

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
