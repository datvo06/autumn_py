"""Pin down the Autumn semantic: when an on-clause writes a state var this tick,
its default next-expression is skipped. Only on-clause-untouched vars re-run
their next-exprs."""
from __future__ import annotations

from autumn_py import Runtime, StateVar, clicked, on, program


def test_on_clause_write_skips_next_expr():
    calls = {"counter_next": 0, "untouched_next": 0}

    @program(grid_size=4)
    class P:
        counter = StateVar(int, init=0)
        untouched = StateVar(int, init=0)

        @counter.next
        def _():
            calls["counter_next"] += 1
            # Would add 100 every tick if allowed to run.
            return P.counter.get() + 100

        @untouched.next
        def _():
            calls["untouched_next"] += 1
            return P.untouched.get() + 1

        @on(clicked)
        def _click():
            # On-clause overwrites counter → counter's next-expr must skip.
            P.counter.set(42)

    r = Runtime(P, seed=0)

    # Tick 1: no click → both next-exprs run.
    r.step()
    assert r.state.get("counter") == 100
    assert r.state.get("untouched") == 1
    assert calls == {"counter_next": 1, "untouched_next": 1}

    # Tick 2: click → counter is overwritten by on-clause to 42.
    # counter_next must NOT fire; untouched_next must still fire.
    r.click(0, 0)
    r.step()
    assert r.state.get("counter") == 42
    assert r.state.get("untouched") == 2
    assert calls == {"counter_next": 1, "untouched_next": 2}

    # Tick 3: no click again → counter_next resumes (from the 42 the click left).
    r.step()
    assert r.state.get("counter") == 142
    assert r.state.get("untouched") == 3
    assert calls == {"counter_next": 2, "untouched_next": 3}


def test_overwritten_set_resets_between_ticks():
    """A write from on-clause N must not suppress the next-expr on tick N+1."""

    @program()
    class P:
        x = StateVar(int, init=0)

        @x.next
        def _():
            return P.x.get() + 1

        @on(clicked)
        def _():
            P.x.set(99)

    r = Runtime(P)
    r.click(0, 0)
    r.step()                         # click commits 99, next-expr skipped
    assert r.state.get("x") == 99
    r.step()                         # no click; next-expr must fire now
    assert r.state.get("x") == 100


def test_next_expr_may_write_sibling_state_vars():
    """Autumn permits a next-expression to set sibling state vars. Those
    writes land directly in _globals (via StateHandler._set), do not
    populate on_writes_this_tick, and become visible to later next-exprs
    via get_var. This test pins that permissive semantic."""

    @program()
    class P:
        a = StateVar(int, init=0)
        b = StateVar(int, init=0)

        @a.next
        def _():
            # Side-effect b while computing a's new value.
            P.b.set(P.b.get() + 10)
            return P.a.get() + 1

        @b.next
        def _():
            # Read b (which a's next-expr just incremented) and add 1.
            return P.b.get() + 1

    r = Runtime(P)
    r.step()
    # a: 0 -> 1. b: 0 -> (a's next sets b to 10) -> (b's next reads 10, adds 1 -> 11).
    assert r.state.get("a") == 1
    assert r.state.get("b") == 11
