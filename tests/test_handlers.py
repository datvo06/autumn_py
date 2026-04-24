from __future__ import annotations

import pytest
from effectful.ops.semantics import handler

from autumn_py.handlers import (
    NativeRandomHandler,
    ObjectAllocHandler,
    PrevStateHandler,
    RenderHandler,
    ReplayRandomHandler,
    StateHandler,
    WriteBufferHandler,
    make_event_intp,
)
from autumn_py.ops import (
    alloc_obj_id,
    emit_render_cell,
    get_click_pos,
    get_prev_var,
    get_var,
    is_event_active,
    sample_uniform,
    set_var,
)
from autumn_py.values import Cell


def test_state_handler_roundtrip():
    st = StateHandler()
    with handler(st):
        set_var("x", 7)
        assert get_var("x") == 7
        set_var("x", 8)
        assert get_var("x") == 8
    assert st.get("x") == 8
    # Free-form set_var does NOT populate on_writes_this_tick — only the
    # write-buffer flush (apply_buffered) does. Skip-check semantics are
    # covered behaviorally in tests/test_overwrite_skips_next.py.
    assert "x" not in st.on_writes_this_tick


def test_state_handler_missing_raises():
    st = StateHandler()
    with handler(st):
        with pytest.raises(NameError):
            get_var("missing")


def test_prev_handler_reads_snapshot():
    snap = {"a": 1, "b": 2}
    with handler(PrevStateHandler(snap)):
        assert get_prev_var("a") == 1
        assert get_prev_var("b") == 2


def test_prev_snapshot_is_isolated_from_live_state():
    st = StateHandler()
    st.seed("a", 1)
    snap = st.freeze_snapshot()
    with handler(st):
        set_var("a", 99)
    with handler(PrevStateHandler(snap)):
        assert get_prev_var("a") == 1


def test_write_buffer_captures_and_flushes():
    st = StateHandler()
    st.seed("a", 0)
    buf = WriteBufferHandler()
    with handler(st):
        with handler(buf):
            set_var("a", 1)
            set_var("a", 2)
            set_var("b", 3)
            # Writes must not yet be visible in the underlying state.
            assert st.get("a") == 0
            assert not st.has("b")
        buf.flush(st)
    assert st.get("a") == 2  # last-writer-wins
    assert st.get("b") == 3
    # Only writes that flow through apply_buffered (on-clause flush) land
    # in on_writes_this_tick.
    assert st.on_writes_this_tick == {"a", "b"}


def test_native_random_deterministic_from_seed():
    a = NativeRandomHandler(seed=42)
    b = NativeRandomHandler(seed=42)
    seq = [1, 2, 3, 4, 5]
    with handler(a):
        ra = [sample_uniform(seq) for _ in range(10)]
    with handler(b):
        rb = [sample_uniform(seq) for _ in range(10)]
    assert ra == rb


def test_replay_random_consumes_in_order():
    draws = [10, 20, 30]
    h = ReplayRandomHandler(draws=draws)
    with handler(h):
        assert sample_uniform([1, 2, 3]) == 10
        assert sample_uniform([1, 2, 3]) == 20
        assert sample_uniform([1, 2, 3]) == 30
        with pytest.raises(RuntimeError):
            sample_uniform([1, 2, 3])


def test_object_alloc_monotonic():
    h = ObjectAllocHandler()
    with handler(h):
        ids = [alloc_obj_id() for _ in range(5)]
    assert ids == [0, 1, 2, 3, 4]


def test_event_intp_reports_active_events_and_click_pos():
    intp = make_event_intp(frozenset({"clicked", "left"}), (3, 4))
    with handler(intp):
        assert is_event_active("clicked") is True
        assert is_event_active("left") is True
        assert is_event_active("right") is False
        assert get_click_pos() == (3, 4)


def test_render_handler_collects_cells():
    h = RenderHandler()
    with handler(h):
        emit_render_cell(Cell(0, 0, "red"))
        emit_render_cell(Cell(1, 2, "blue"))
    assert len(h.cells) == 2
    h.reset()
    assert h.cells == []
