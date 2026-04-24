"""Port of Autumn.cpp/tests/mario.sexp.

Mario falls under gravity (moveDown every tick), arrow keys move him
horizontally, up-arrow jumps (only when standing on something).
Clicking fires a bullet upward that disappears when it hits a step.
Collecting a coin removes it and grants a bullet. The enemy patrols
horizontally and is killed after taking ``enemyLives`` hits.
"""
from __future__ import annotations

from autumn_py import (
    Cell,
    Position,
    StateVar,
    click,
    clicked,
    left,
    obj,
    on,
    prev,
    program,
    right,
    up,
)
from autumn_py.stdlib import (
    addObj,
    defined,
    intersects,
    moveDown,
    moveDownNoCollision,
    moveLeft,
    moveLeftNoCollision,
    moveNoCollision,
    moveRight,
    moveRightNoCollision,
    moveUp,
    removeObj,
    updateObj,
)


@obj
class Mario:
    bullets: int
    cell = Cell(0, 0, "red")


@obj
class Step:
    cells = [
        Cell(-1, 0, "darkorange"),
        Cell(0, 0, "darkorange"),
        Cell(1, 0, "darkorange"),
    ]


@obj
class Coin:
    cell = Cell(0, 0, "gold")


@obj
class Enemy:
    movingLeft: bool
    lives: int
    cells = [
        Cell(-1, 0, "blue"), Cell(0, 0, "blue"), Cell(1, 0, "blue"),
        Cell(-1, 1, "blue"), Cell(0, 1, "blue"), Cell(1, 1, "blue"),
    ]


@obj
class Bullet:
    cell = Cell(0, 0, "mediumpurple")


@program(grid_size=16)
class MarioGame:
    mario = StateVar(object)
    steps = StateVar(list)
    coins = StateVar(list)
    enemy = StateVar(object)
    bullets = StateVar(list, init=[])
    enemyLives = StateVar(int, init=1)

    # ---- initializers (run under the handler stack) -----------------------

    @mario.initializer
    def _():
        return Mario(0, Position(7, 15))

    @steps.initializer
    def _():
        return [Step(Position(4, 13)), Step(Position(8, 10)), Step(Position(11, 7))]

    @coins.initializer
    def _():
        return [Coin(Position(4, 12)), Coin(Position(7, 4)), Coin(Position(11, 6))]

    @enemy.initializer
    def _():
        return Enemy(True, 1, Position(7, 0))

    # ---- default next-expressions ----------------------------------------

    @mario.next
    def _():
        m = prev(MarioGame.mario)
        return moveDown(m) if intersects(moveDown(m), prev(MarioGame.coins)) else moveDownNoCollision(m)

    @steps.next
    def _():
        return prev(MarioGame.steps)

    @coins.next
    def _():
        return prev(MarioGame.coins)

    @enemy.next
    def _():
        e = prev(MarioGame.enemy)
        if e is None or not e.alive:
            return e
        return moveLeft(e) if e.movingLeft else moveRight(e)

    @bullets.next
    def _():
        def _advance(b):
            return removeObj(b) if intersects(moveUp(b), prev(MarioGame.steps)) else moveUp(b)
        return updateObj(prev(MarioGame.bullets), _advance)

    @enemyLives.next
    def _():
        return prev(MarioGame.enemyLives)

    # ---- enemy patrol: bounce at x=1 (going left) / x=14 (going right) ---

    @on(lambda: defined("enemy") and prev(MarioGame.enemy).origin.x == 1)
    def _():
        e = prev(MarioGame.enemy)
        MarioGame.enemy.set(moveRight(updateObj(e, "movingLeft", False)))

    @on(lambda: defined("enemy") and prev(MarioGame.enemy).origin.x == 14)
    def _():
        e = prev(MarioGame.enemy)
        MarioGame.enemy.set(moveLeft(updateObj(e, "movingLeft", True)))

    # ---- controls --------------------------------------------------------

    @on(left)
    def _():
        m = prev(MarioGame.mario)
        if intersects(moveLeft(m), prev(MarioGame.coins)):
            MarioGame.mario.set(moveLeft(m))
        else:
            MarioGame.mario.set(moveLeftNoCollision(m))

    @on(right)
    def _():
        m = prev(MarioGame.mario)
        if intersects(moveRight(m), prev(MarioGame.coins)):
            MarioGame.mario.set(moveRight(m))
        else:
            MarioGame.mario.set(moveRightNoCollision(m))

    # Jump only when standing on something (moveDownNoCollision is a no-op).
    @on(lambda: up() and moveDownNoCollision(prev(MarioGame.mario)) == prev(MarioGame.mario))
    def _():
        MarioGame.mario.set(moveNoCollision(prev(MarioGame.mario), 0, -4))

    # ---- coin collection -------------------------------------------------

    @on(lambda: intersects(prev(MarioGame.mario), prev(MarioGame.coins)))
    def _():
        m = prev(MarioGame.mario)
        cs = prev(MarioGame.coins)
        MarioGame.coins.set(removeObj(cs, lambda o: intersects(o, m)))
        MarioGame.mario.set(
            moveDownNoCollision(updateObj(m, "bullets", m.bullets + 1))
        )

    # ---- shooting --------------------------------------------------------

    @on(lambda: clicked() and prev(MarioGame.mario).bullets > 0)
    def _():
        m = prev(MarioGame.mario)
        MarioGame.bullets.set(addObj(prev(MarioGame.bullets), Bullet(m.origin)))
        MarioGame.mario.set(
            moveDownNoCollision(updateObj(m, "bullets", m.bullets - 1))
        )

    # ---- enemy vs bullets ------------------------------------------------

    @on(lambda: defined("enemy") and intersects(prev(MarioGame.enemy), prev(MarioGame.bullets)))
    def _():
        e = prev(MarioGame.enemy)
        bs = prev(MarioGame.bullets)
        MarioGame.bullets.set(removeObj(bs, lambda b: intersects(b, e)))
        if prev(MarioGame.enemyLives) == 1:
            MarioGame.enemy.set(removeObj(e))
        else:
            MarioGame.enemy.set(moveLeft(e) if e.movingLeft else moveRight(e))
        MarioGame.enemyLives.set(prev(MarioGame.enemyLives) - 1)
