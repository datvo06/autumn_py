from __future__ import annotations

import contextlib
from typing import Any

from effectful.ops.semantics import handler

from .api import ProgramSpec, TypeMismatch, _check_type
from .events import EVENT_NAMES
from .handlers import (
    NativeRandomHandler,
    ObjectAllocHandler,
    PrevStateHandler,
    RenderHandler,
    StateHandler,
    WorldHandler,
    WriteBufferHandler,
    make_event_intp,
)
from .ops import emit_render_cell
from .values import ObjectInstance, cell_to_dict


class Runtime:
    """Drives a @program-decorated Autumn class through init and per-tick steps.

    Handler composition:
      * Persistent handlers installed for the lifetime of the Runtime:
          StateHandler, NativeRandomHandler, ObjectAllocHandler, RenderHandler.
      * Per-tick handlers layered inside step():
          PrevStateHandler (frozen snapshot of globals before the tick),
          make_event_intp(...) (frozen dict of this tick's events),
          WriteBufferHandler (buffers set_var during on-clauses).

    Next-expressions are evaluated in declaration order under the live
    state handler only (no write buffer, no guard). Autumn permits
    next-exprs to side-effect sibling state vars via ``set_var``; those
    writes land directly in ``_globals`` without populating
    ``on_writes_this_tick``, so they do not suppress other next-exprs.
    """

    def __init__(self, program_cls, *, seed: int = 42) -> None:
        spec: ProgramSpec | None = getattr(program_cls, "_autumn_spec", None)
        if spec is None:
            raise TypeError(
                f"{program_cls.__name__} is not decorated with @program()"
            )
        self.spec = spec
        self.program_cls = program_cls

        self.state = StateHandler()
        self.random = NativeRandomHandler(seed=seed)
        self.alloc = ObjectAllocHandler()
        self.render = RenderHandler()
        self.world = WorldHandler(self)

        self._stack = contextlib.ExitStack()
        self._stack.enter_context(handler(self.state))
        self._stack.enter_context(handler(self.random))
        self._stack.enter_context(handler(self.alloc))
        self._stack.enter_context(handler(self.render))
        self._stack.enter_context(handler(self.world))

        self._event_queue: list[dict] = []

        self._init_phase()

    # ----------------------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------------------

    def _init_phase(self) -> None:
        """Seed each state var with its init value. Initializers that require
        effects (e.g. ObjectInstance construction via alloc_obj_id) run here
        with the full persistent handler stack already installed."""
        for sv in self.spec.state_vars:
            assert sv.name is not None
            value = sv.initial_value()
            _check_type(value, sv.type_, context=f"initial value of state var {sv.name!r}")
            self.state.seed(sv.name, value)

    def close(self) -> None:
        self._stack.close()

    def __enter__(self) -> "Runtime":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ----------------------------------------------------------------------
    # Event input
    # ----------------------------------------------------------------------

    def click(self, x: int, y: int) -> None:
        self._event_queue.append({"kind": "click", "x": int(x), "y": int(y)})

    def left(self) -> None:
        self._event_queue.append({"kind": "left"})

    def right(self) -> None:
        self._event_queue.append({"kind": "right"})

    def up(self) -> None:
        self._event_queue.append({"kind": "up"})

    def down(self) -> None:
        self._event_queue.append({"kind": "down"})

    # ----------------------------------------------------------------------
    # The tick loop
    # ----------------------------------------------------------------------

    def step(self) -> None:
        snap = self.state.freeze_snapshot()
        active, click_pos = self._drain_events()
        event_intp = make_event_intp(active, click_pos)

        self.state.reset_on_writes()
        write_buf = WriteBufferHandler()

        with handler(PrevStateHandler(snap)):
            with handler(event_intp):
                with handler(write_buf):
                    for clause in self.spec.on_clauses:
                        pred = clause.predicate
                        result = pred() if callable(pred) else pred
                        # §2.3: on-clause predicates must be Bool. Coerce
                        # bool subclasses (the event-sentinel __bool__'s
                        # result is bool by definition; lambdas may return
                        # non-bool truthy/falsy — reject those).
                        if not isinstance(result, bool):
                            raise TypeMismatch(
                                f"on-clause predicate {clause.name!r} returned "
                                f"{type(result).__name__} (value: {result!r}); "
                                f"expected bool"
                            )
                        if result:
                            clause.body()
                write_buf.flush(self.state)

            # Next-phase: iterate state vars in declaration order. Skip a
            # var whose name was written by an on-clause this tick
            # (on_writes_this_tick). Otherwise evaluate its next-expression
            # and commit. A next-expression MAY call set_var on sibling
            # vars; those writes pass through StateHandler._set to
            # _globals directly, do not populate on_writes_this_tick, and
            # become visible to later next-exprs via get_var.
            for sv in self.spec.state_vars:
                if sv._next_fn is None:
                    continue
                if sv.name in self.state.on_writes_this_tick:
                    continue
                value = sv._next_fn()
                _check_type(value, sv.type_, context=f"next-expression of state var {sv.name!r}")
                self.state.commit_next(sv.name, value)

    def _drain_events(self) -> tuple[frozenset[str], tuple[int, int] | None]:
        active: set[str] = set()
        click_pos: tuple[int, int] | None = None
        for ev in self._event_queue:
            kind = ev["kind"]
            if kind == "click":
                active.add("clicked")
                click_pos = (ev["x"], ev["y"])
            elif kind in EVENT_NAMES:
                active.add(kind)
        self._event_queue.clear()
        return frozenset(active), click_pos

    # ----------------------------------------------------------------------
    # Rendering
    # ----------------------------------------------------------------------

    def render_all(self) -> list[dict]:
        """Return rendered cells in draw order as a list of {x, y, color} dicts.

        The walk routes every cell through the emit_render_cell op, so a
        custom RenderHandler can filter, reorder, or enrich the output.
        """
        self.render.reset()
        for sv in self.spec.state_vars:
            if not self.state.has(sv.name):
                continue
            for inst in _iter_instances(self.state.get(sv.name)):
                if not inst.alive:
                    continue
                for cell in inst.rendered_cells():
                    emit_render_cell(cell)
        return [cell_to_dict(c) for c in self.render.cells]


def _iter_instances(value: Any):
    if isinstance(value, ObjectInstance):
        yield value
    elif isinstance(value, (list, tuple)):
        for v in value:
            if isinstance(v, ObjectInstance):
                yield v
