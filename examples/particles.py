"""Python port of Autumn.cpp/tests/particles.sexp.

Original s-expression::

    (program
      (= GRID_SIZE 16)
      (object Particle (Cell 0 0 "blue"))
      (: particles (List Particle))
      (= particles (initnext (list)
        (updateObj (prev particles)
                   (--> obj (Particle (uniformChoice (adjPositions (.. obj origin))))))))
      (on clicked (= particles (addObj (prev particles)
                                        (Particle (Position (.. click x) (.. click y)))))))
"""
from autumn_py import Cell, Position, StateVar, click, clicked, obj, on, prev, program
from autumn_py.stdlib import addObj, adjPositions, updateObj, uniformChoice


@obj
class Particle:
    cell = Cell(0, 0, "blue")


@program(grid_size=16)
class Particles:
    particles = StateVar(list, init=[])

    @particles.next
    def _():
        return updateObj(
            prev(Particles.particles),
            lambda o: Particle(uniformChoice(adjPositions(o.origin))),
        )

    @on(clicked)
    def _():
        Particles.particles.set(
            addObj(
                prev(Particles.particles),
                Particle(Position(click.x, click.y)),
            )
        )
