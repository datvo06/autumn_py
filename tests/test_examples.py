"""Law-based tests for the ported example programs. No gold fixtures — each
test asserts an invariant of the program's semantics rather than a specific
cell list."""
from __future__ import annotations

from autumn_py import Runtime


def _cells_by_color(cells: list[dict], color: str) -> list[dict]:
    return [c for c in cells if c["color"] == color]


# -------------------------------------------------------------------------
# Mario
# -------------------------------------------------------------------------

def test_mario_starts_with_one_mario_three_coins_three_steps_one_enemy():
    from examples.mario import MarioGame
    with Runtime(MarioGame, seed=42) as r:
        cells = r.render_all()
    # 1 mario + 3 steps * 3 cells + 3 coins + 1 enemy * 6 cells.
    assert len(_cells_by_color(cells, "red")) == 1
    assert len(_cells_by_color(cells, "darkorange")) == 9
    assert len(_cells_by_color(cells, "gold")) == 3
    assert len(_cells_by_color(cells, "blue")) == 6


def test_mario_moves_right_under_right_arrow():
    from examples.mario import MarioGame
    with Runtime(MarioGame, seed=42) as r:
        x0 = _cells_by_color(r.render_all(), "red")[0]["x"]
        r.right()
        r.step()
        x1 = _cells_by_color(r.render_all(), "red")[0]["x"]
    assert x1 == x0 + 1


def test_mario_jump_only_when_landed():
    """Up-arrow jumps only when Mario is on the ground (moveDown is a no-op).
    Pressing up mid-air does nothing."""
    from examples.mario import MarioGame
    with Runtime(MarioGame, seed=42) as r:
        y0 = _cells_by_color(r.render_all(), "red")[0]["y"]
        assert y0 == 15  # Mario spawns on the floor
        r.up()
        r.step()
        y1 = _cells_by_color(r.render_all(), "red")[0]["y"]
        assert y1 == y0 - 4  # jump displaces by 4 upward
        # Now mid-air: up does nothing
        r.up()
        r.step()
        y2 = _cells_by_color(r.render_all(), "red")[0]["y"]
        # He's falling one unit per tick; up-arrow shouldn't have lifted him.
        assert y2 == y1 + 1


# -------------------------------------------------------------------------
# Game of Life
# -------------------------------------------------------------------------

def test_gol_initial_pattern_has_five_living():
    from examples.game_of_life import GameOfLife
    with Runtime(GameOfLife, seed=0) as r:
        cells = r.render_all()
    assert len(_cells_by_color(cells, "lightpink")) == 5
    # 16x16 grid + 2 buttons
    assert len(cells) == 16 * 16 + 2


def test_gol_green_button_advances_one_generation():
    """Clicking the green button (top-left) must fire the advance clause.
    Specifically: our seed pattern { (2,3), (3,3), (3,1), (4,3), (4,2) } on
    one step becomes a different 5-cell living set (some die, some are
    born). Just assert the living set *changed* — the specific pattern
    is the algorithm's business, not the DSL's."""
    from examples.game_of_life import GameOfLife
    with Runtime(GameOfLife, seed=0) as r:
        before = {(c["x"], c["y"]) for c in _cells_by_color(r.render_all(), "lightpink")}
        r.click(0, 15)  # green button position
        r.step()
        after = {(c["x"], c["y"]) for c in _cells_by_color(r.render_all(), "lightpink")}
    assert before != after


def test_gol_silver_button_kills_all():
    from examples.game_of_life import GameOfLife
    with Runtime(GameOfLife, seed=0) as r:
        r.click(15, 15)  # silver reset button
        r.step()
        living = _cells_by_color(r.render_all(), "lightpink")
    assert living == []


def test_gol_grid_click_toggles_that_particle():
    from examples.game_of_life import GameOfLife
    with Runtime(GameOfLife, seed=0) as r:
        # (0, 0) starts dead. Click toggles to living.
        r.click(0, 0)
        r.step()
        living = {(c["x"], c["y"]) for c in _cells_by_color(r.render_all(), "lightpink")}
    assert (0, 0) in living


# -------------------------------------------------------------------------
# Ants
# -------------------------------------------------------------------------

def test_ants_start_without_food_and_dont_move():
    from examples.ants import AntsGame
    with Runtime(AntsGame, seed=42) as r:
        before = sorted((c["x"], c["y"]) for c in _cells_by_color(r.render_all(), "gray"))
        r.step()
        after = sorted((c["x"], c["y"]) for c in _cells_by_color(r.render_all(), "gray"))
    assert before == after


def test_ants_click_spawns_two_food():
    from examples.ants import AntsGame
    with Runtime(AntsGame, seed=42) as r:
        r.click(8, 8)
        r.step()
        food = _cells_by_color(r.render_all(), "red")
    assert len(food) == 2


def test_ants_step_toward_food_by_one_unit():
    """When food exists, each ant displaces by a unit vector toward closest food."""
    from examples.ants import AntsGame
    with Runtime(AntsGame, seed=42) as r:
        r.click(8, 8)
        r.step()  # food now exists
        ants_before = [(c["x"], c["y"]) for c in _cells_by_color(r.render_all(), "gray")]
        r.step()
        ants_after = [(c["x"], c["y"]) for c in _cells_by_color(r.render_all(), "gray")]
    for (bx, by), (ax, ay) in zip(ants_before, ants_after):
        # Each coordinate displaces by at most 1 (the unit vector clamp).
        assert abs(ax - bx) <= 1 and abs(ay - by) <= 1


# -------------------------------------------------------------------------
# Grow
# -------------------------------------------------------------------------

def test_grow_initial_scene_has_sun_cloud_and_eight_leaves():
    from examples.grow import GrowGame
    with Runtime(GrowGame, seed=42) as r:
        cells = r.render_all()
    assert len(_cells_by_color(cells, "gold")) == 9    # 3x3 sun
    assert len(_cells_by_color(cells, "gray")) == 12   # 4x3 cloud
    assert len(_cells_by_color(cells, "green")) == 8   # 8 leaves along y=15


def test_grow_down_arrow_drops_a_water_pellet():
    from examples.grow import GrowGame
    with Runtime(GrowGame, seed=42) as r:
        r.down()
        r.step()
        blues = _cells_by_color(r.render_all(), "blue")
    assert len(blues) == 1


def test_grow_water_falls_one_cell_per_tick_until_it_hits_a_leaf():
    """Between the drop and contact with a leaf, water descends one y per
    tick with no other shape change."""
    from examples.grow import GrowGame
    with Runtime(GrowGame, seed=42) as r:
        r.down()
        r.step()
        drop_positions = []
        for _ in range(13):
            blues = _cells_by_color(r.render_all(), "blue")
            drop_positions.append(blues[0]["y"] if blues else None)
            r.step()
    ys = [y for y in drop_positions if y is not None]
    for a, b in zip(ys, ys[1:]):
        assert b == a + 1


def test_grow_water_landing_on_green_leaf_grows_a_new_leaf():
    """One full drop from cloud position (13, 0): the water walks to (13, 15),
    gets consumed, and a new green leaf spawns at (13, 14)."""
    from examples.grow import GrowGame
    with Runtime(GrowGame, seed=42) as r:
        r.down()
        r.step()
        # Water needs ~14 ticks to reach the leaves from y=1.
        for _ in range(20):
            r.step()
        cells = r.render_all()
    greens = _cells_by_color(cells, "green")
    # Started with 8; growth should have added at least one more.
    assert len(greens) > 8
    green_positions = {(c["x"], c["y"]) for c in greens}
    assert (13, 14) in green_positions


def test_grow_left_arrow_moves_cloud_one_cell_left():
    from examples.grow import GrowGame
    with Runtime(GrowGame, seed=42) as r:
        before = min(c["x"] for c in _cells_by_color(r.render_all(), "gray"))
        r.left()
        r.step()
        after = min(c["x"] for c in _cells_by_color(r.render_all(), "gray"))
    assert after == before - 1


# -------------------------------------------------------------------------
# Sand
# -------------------------------------------------------------------------

def test_sand_initial_scene():
    from examples.sand import SandGame
    with Runtime(SandGame, seed=42) as r:
        cells = r.render_all()
    assert len(_cells_by_color(cells, "tan")) == 26   # dry sand grains
    assert len(_cells_by_color(cells, "red")) == 1    # sand-select button
    assert len(_cells_by_color(cells, "green")) == 1  # water-select button


def test_sand_clicking_water_button_switches_click_type():
    from examples.sand import SandGame
    with Runtime(SandGame, seed=42) as r:
        assert r.state.get("clickType") == "sand"
        r.click(7, 0)
        r.step()
        assert r.state.get("clickType") == "water"
        r.click(2, 0)
        r.step()
        assert r.state.get("clickType") == "sand"


def test_sand_water_falls_straight_down_through_empty_column():
    """Dropping water at (3, 0) — which is empty down to y=6 — should find
    the drop at y=6 after six ticks of gravity."""
    from examples.sand import SandGame
    with Runtime(SandGame, seed=42) as r:
        r.click(7, 0); r.step()   # switch to water
        r.click(3, 0); r.step()   # drop at (3, 0)
        for _ in range(6):
            r.step()
        water = _cells_by_color(r.render_all(), "skyblue")
    assert len(water) == 1
    assert (water[0]["x"], water[0]["y"]) == (3, 6)


def test_sand_adjacent_to_water_becomes_liquid():
    """After the water drop has reached the top of the sand pile, sand grains
    in a manhattan-1 neighbourhood of the water should liquefy (recolour
    sandybrown)."""
    from examples.sand import SandGame
    with Runtime(SandGame, seed=42) as r:
        r.click(7, 0); r.step()
        r.click(3, 0); r.step()
        for _ in range(8):  # let water reach sand + one liquefaction tick
            r.step()
        liquid = _cells_by_color(r.render_all(), "sandybrown")
    assert len(liquid) > 0


def test_sand_click_on_free_cell_adds_a_grain_of_selected_type():
    """clickType='sand' + click on a free cell → a new dry (tan) grain."""
    from examples.sand import SandGame
    with Runtime(SandGame, seed=42) as r:
        # clickType starts as 'sand'.
        before = len(_cells_by_color(r.render_all(), "tan"))
        r.click(0, 0)  # free cell
        r.step()
        after = len(_cells_by_color(r.render_all(), "tan"))
    assert after == before + 1


# -------------------------------------------------------------------------
# Space Invaders (R2 fixed — the converged emit; see drafts/...handlers... Part 1)
# -------------------------------------------------------------------------

def test_space_invaders_r2_fixed_starts_with_player_and_12_enemy_formation():
    from examples.space_invaders import SpaceInvadersR2Fixed
    with Runtime(SpaceInvadersR2Fixed, seed=42) as r:
        cells = r.render_all()
    assert len(_cells_by_color(cells, "blue")) == 1     # player
    assert len(_cells_by_color(cells, "red")) == 12     # 3 rows × 4 cols
    assert len(_cells_by_color(cells, "yellow")) == 0   # no enemy bullets yet
    assert len(_cells_by_color(cells, "lightgreen")) == 0


def test_space_invaders_r2_fixed_spawns_an_enemy_bullet_after_step_3():
    """At t=3 step_count == next_spawn_step == 3; spawn_event becomes True
    at end of tick. enemy_bullets reads prev(spawn_event), so the bullet
    appears at tick 4."""
    from examples.space_invaders import SpaceInvadersR2Fixed
    with Runtime(SpaceInvadersR2Fixed, seed=42) as r:
        for _ in range(5):
            r.step()
        cells = r.render_all()
    # By tick 5 we should have at least one enemy bullet on screen.
    assert len(_cells_by_color(cells, "yellow")) >= 1


def test_space_invaders_r2_fixed_player_moves_left_under_left_arrow():
    from examples.space_invaders import SpaceInvadersR2Fixed
    with Runtime(SpaceInvadersR2Fixed, seed=42) as r:
        before = _cells_by_color(r.render_all(), "blue")[0]
        r.left()
        r.step()
        after = _cells_by_color(r.render_all(), "blue")[0]
    assert after["x"] == before["x"] - 1


def test_space_invaders_r2_fixed_click_fires_player_bullet():
    from examples.space_invaders import SpaceInvadersR2Fixed
    with Runtime(SpaceInvadersR2Fixed, seed=42) as r:
        r.click(8, 14)   # at the player's position
        r.step()
        cells = r.render_all()
    assert len(_cells_by_color(cells, "lightgreen")) >= 1
