"""Property annotations for `@program`-decorated classes.

Borrows Dafny's pattern of attaching specifications inline with code:
each next-clause carries its spec via a single ``@spec(...)`` decorator
(or smaller standalone decorators like ``@modifies(...)`` /
``@no_stochastic`` for compound-spec ergonomics).

The decorator records spec metadata onto the function (``__autumn_spec__``).
StateVar.next / @on(...) recognise the metadata and mint corresponding
Goal subclass instances against their anchor, registering them on a
module-level ``_pending_properties`` list. ``@program`` drains this
list into ``cls._autumn_spec.properties``; ``gate(emit_cls)`` reads
from there when no explicit goals are passed.

Annotations supported in this module:

* ``@spec(...)`` — bundle: ``no_stochastic`` / ``modifies`` /
  ``invariant`` / ``unroll`` / ``init_constraints`` / ``horizon``.
* ``@no_stochastic`` — sugar for ``@spec(no_stochastic=True)``.
* ``@modifies(*allowed_writes)`` — sugar for ``@spec(modifies=...)``.

Compose by stacking — ``@spec(...)`` and ``@modifies(...)`` on the same
function merge their fields. Last decorator wins on conflicts.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable

# Module-level pending list; @program drains into _autumn_spec.properties.
# Same pattern as _pending_on_clauses / _pending_next_clauses in api.py.
_pending_properties: list = []


_VALID_MONOTONE = {"nondecreasing", "nonincreasing", "strict_increasing", "strict_decreasing"}


@dataclass(frozen=True)
class Spec:
    """A bundled spec for a next-clause or on-clause body.

    Each field is optional; only populated fields contribute goals.

    Fields
    ------
    no_stochastic : bool
        The decorated body's read-set must not contain a
        ``("sample_uniform", ...)`` atom. Mints a ``FootprintExcludeGoal``.
    modifies : tuple[str | StateVar, ...] | None
        The decorated body's write-set must be a subset of
        ``allowed_writes`` (plus the implicitly-decorated state var).
        Mints a ``WriteFrameGoal``. Entries may be strings or StateVar
        references — StateVar refs are resolved to ``.name`` at
        goal-mint time (after class-body's ``__set_name__`` binds names).
    invariant : Callable[[FuncMap, int], z3.BoolRef] | None
        SMT-checkable invariant: ``(funcs, k) -> Z3 BoolRef`` producing
        the per-tick goal predicate at tick k. Conjoined across the
        bounded horizon. Mints a ``ModularArithmeticGoal``. ``funcs``
        is a dict-like keyed on either string state-var names *or*
        StateVar references — both ``funcs["x"]`` and ``funcs[x]``
        resolve to the same Z3 function.
    unroll : tuple[str | StateVar, ...] | None
        Anchors whose transitions are unrolled by the SMT goal. Strings
        like ``"x.next"`` or bare StateVar references (which resolve to
        ``"<name>.next"``). Defaults to ``(self_anchor,)``. Used only
        with ``invariant``.
    init_constraints : Callable[[FuncMap], list[z3.BoolRef]] | None
        Z3 init constraints (state-var inits, prev-anchors, counters).
        Used only with ``invariant``.
    horizon : int
        Bounded model checking horizon for ``invariant``. Default 6.
    monotone : str | None
        One of ``_VALID_MONOTONE``. (Future work — currently validated
        but does not mint a goal.)
    """
    no_stochastic: bool = False
    modifies: tuple | None = None
    invariant: Callable[[Any, int], Any] | None = None
    unroll: tuple | None = None
    init_constraints: Callable[[Any], list] | None = None
    horizon: int = 6
    monotone: str | None = None

    def __post_init__(self) -> None:
        # Imported lazily to avoid circular import: properties → api → properties
        from .api import StateVar
        if self.modifies is not None:
            if not isinstance(self.modifies, tuple):
                raise TypeError(
                    f"Spec.modifies must be a tuple of state-var names or "
                    f"StateVar references (got {type(self.modifies).__name__})"
                )
            for w in self.modifies:
                if not isinstance(w, (str, StateVar)):
                    raise TypeError(
                        f"Spec.modifies entries must be str or StateVar; "
                        f"got {type(w).__name__}: {w!r}"
                    )
        if self.unroll is not None:
            if not isinstance(self.unroll, tuple):
                raise TypeError(
                    f"Spec.unroll must be a tuple of anchor strings or "
                    f"StateVar references (got {type(self.unroll).__name__})"
                )
            for u in self.unroll:
                if not isinstance(u, (str, StateVar)):
                    raise TypeError(
                        f"Spec.unroll entries must be str or StateVar; "
                        f"got {type(u).__name__}: {u!r}"
                    )
        if self.monotone is not None and self.monotone not in _VALID_MONOTONE:
            raise ValueError(
                f"Spec.monotone must be one of {sorted(_VALID_MONOTONE)}; "
                f"got {self.monotone!r}"
            )
        if self.horizon < 1:
            raise ValueError(f"Spec.horizon must be >= 1; got {self.horizon}")
        if self.unroll is not None and self.invariant is None:
            raise ValueError(
                "Spec.unroll is only meaningful with an invariant; "
                "received unroll without invariant"
            )

    def merge(self, other: "Spec") -> "Spec":
        """Right-biased merge — `other`'s populated fields override self.
        Used when ``@spec(...)`` and ``@modifies(...)`` stack."""
        return Spec(
            no_stochastic=other.no_stochastic or self.no_stochastic,
            modifies=other.modifies if other.modifies is not None else self.modifies,
            invariant=other.invariant if other.invariant is not None else self.invariant,
            unroll=other.unroll if other.unroll is not None else self.unroll,
            init_constraints=(
                other.init_constraints if other.init_constraints is not None
                else self.init_constraints
            ),
            horizon=other.horizon if other.horizon != 6 else self.horizon,
            monotone=other.monotone if other.monotone is not None else self.monotone,
        )


def spec(**kwargs) -> Callable[[Callable], Callable]:
    """Decorator: attach a structured ``Spec`` to a next-clause / on-clause body.

    Validates fields at decoration time (unknown kwarg → TypeError;
    invalid value → ValueError). Composes with prior ``@spec(...)`` /
    ``@modifies(...)`` / ``@no_stochastic`` decorators on the same function
    via ``Spec.merge``.

    Goals are not minted yet — the spec is only attached as
    ``fn.__autumn_spec__``. The actual minting happens when StateVar.next
    or @on(...) sees the function and knows its anchor.
    """
    s = Spec(**kwargs)

    def decorator(fn: Callable) -> Callable:
        existing = getattr(fn, "__autumn_spec__", None)
        if existing is not None:
            s_merged = existing.merge(s)
        else:
            s_merged = s
        fn.__autumn_spec__ = s_merged  # type: ignore[attr-defined]
        return fn

    return decorator


def no_stochastic(fn: Callable) -> Callable:
    """Sugar for ``@spec(no_stochastic=True)``. Composes with other specs."""
    return spec(no_stochastic=True)(fn)


def modifies(*allowed_writes: str) -> Callable[[Callable], Callable]:
    """Sugar for ``@spec(modifies=allowed_writes)``. Composes with other specs.

    Usage: ``@modifies("step_count")`` or ``@modifies("a", "b")``.
    """
    return spec(modifies=tuple(allowed_writes))


def _resolve_var_name(ref) -> str:
    """Convert a str-or-StateVar ref into the state-var's string name."""
    from .api import StateVar
    if isinstance(ref, str):
        return ref
    if isinstance(ref, StateVar):
        if ref.name is None:
            raise ValueError(
                f"StateVar reference has no bound name yet — make sure "
                f"the StateVar is a class attribute (so __set_name__ runs) "
                f"before being used in a Spec."
            )
        return ref.name
    raise TypeError(f"expected str or StateVar; got {type(ref).__name__}: {ref!r}")


def _resolve_anchor(ref) -> str:
    """Convert a str-or-StateVar ref into an anchor string. StateVar refs
    resolve to ``"<name>.next"`` (the most common use case)."""
    from .api import StateVar
    if isinstance(ref, str):
        return ref
    if isinstance(ref, StateVar):
        return f"{_resolve_var_name(ref)}.next"
    raise TypeError(f"expected str or StateVar; got {type(ref).__name__}: {ref!r}")


def realize_spec_goals(spec_obj: Spec, anchor: str) -> list:
    """Translate a Spec at a known anchor into a list of Goal instances.

    StateVar references in ``modifies``/``unroll`` are resolved here
    (after class-body's ``__set_name__`` has bound names).
    """
    from .gate import (
        FootprintExcludeGoal,
        ModularArithmeticGoal,
        WriteFrameGoal,
    )

    goals: list = []
    if spec_obj.no_stochastic:
        goals.append(FootprintExcludeGoal(
            anchor=anchor,
            exclude=(("sample_uniform",),),
        ))
    if spec_obj.modifies is not None:
        resolved_writes = tuple(_resolve_var_name(w) for w in spec_obj.modifies)
        goals.append(WriteFrameGoal(
            anchor=anchor,
            allowed_writes=resolved_writes,
        ))
    if spec_obj.invariant is not None:
        if spec_obj.unroll is not None:
            resolved_unroll = tuple(_resolve_anchor(u) for u in spec_obj.unroll)
        else:
            resolved_unroll = (anchor,)
        goals.append(ModularArithmeticGoal(
            anchor=anchor,
            unroll=resolved_unroll,
            init_constraints=spec_obj.init_constraints or (lambda *args: []),
            goal_factory=spec_obj.invariant,
            horizon=spec_obj.horizon,
        ))
    # spec_obj.monotone is validated but doesn't mint a goal yet — future work.
    return goals
