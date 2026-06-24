# Refactor design notes

Rationale for a few non-obvious decisions in `autumn_py`, kept out of the
source so the code can stay terse. Each entry: the decision, why, and the
alternatives rejected.

## `@symbolic` binds its op under a private global in the *live* module namespace

`autumn_py/_ast_rewrite.py` — `_IF_THEN_ELSE_GLOBAL = "__autumn_if_then_else__"`.

The `@symbolic` rewriter lifts native `if`/ternaries into calls of the
`if_then_else` op. The rewritten function needs that op resolvable at call
time. Two independent constraints shape how we inject it:

- **Private name, not bare `if_then_else`.** Binding the bare name would be a
  collision land mine: a user-defined `if_then_else` in the same module would
  silently win (the rewrite would call theirs) or be clobbered. Targeting a
  private `__autumn_if_then_else__` that only the rewriter emits removes the
  collision in both directions.
- **Live module globals, not a snapshot copy.** A `@symbolic` next-clause
  body references its enclosing `@program` class (e.g. `SpaceInvadersR2Fixed`),
  which is bound to the module only *after* the class body finishes — i.e.
  after this decorator runs. Exec'ing into a copy of `fn.__globals__` taken at
  decoration time would miss that later binding and raise `NameError` at call
  time. So we bind into the live `fn.__globals__`.

Rejected: the reviewer-suggested "exec into a fresh `{**fn.__globals__, ...}`
copy" — unsound here for the late-binding reason above (it failed exactly that
way in testing). Residual cost: one idempotent private key is written into the
module globals. A zero-pollution version would bind the op as a default
argument on the rewritten function (a local, no globals write), but that needs
AST signature surgery; not worth it for one private symbol.

## `transition_of` is the single next-phase transition term, shared by runtime and gate

`autumn_py/api.py` — `transition_of(sv, *, validate=None)`.

The rule "a next-clause's return value becomes `set_var(<var>, value)`" used
to be encoded twice: the runtime committed off-op (a direct `_globals` write)
while the gate analysed an on-op `set_var` re-derivation. They agreed only by
coincidence. Now both run the one `transition_of` callable, so the term the
gate analyses under `read_set` / `SmtCollectHandler` is literally the term the
runtime executes — gate soundness is structural rather than a property of two
hand-kept copies staying in sync.

The optional `validate` hook keeps the runtime's commit transactional. The
runtime passes its type-check as `validate`, which runs on the computed value
*before* `set_var`, so a mistyped next-expression raises without the bad value
ever reaching state. The gate omits `validate` — a symbolic value isn't a
concrete instance to type-check, and the SMT path must stay untouched.

(The validator guards only this next-expression's own commit; a clause that
`set_var`s a *sibling* var still commits that sibling through the op before
`validate` fires. That is the documented sibling-write semantics, not a
transaction over the whole state.)

## `read_set` catches nothing — a complete footprint, or a loud failure

`autumn_py/smt.py` — `read_set`.

`read_set` runs a clause under `SmtCollectHandler` and returns the syntactic
atoms it touched. The footprint goals (`@no_stochastic`, `@modifies`) rely on
it being **complete**: a missed `sample_uniform` or `set_var` past some
failure point would silently false-pass the goal.

So `read_set` swallows nothing. A clause either evaluates end-to-end (a
provably complete footprint) or it raises — native `if`/`bool()` on a symbolic
value raises `z3.Z3Exception`; an op with no symbolic interpretation raises
`NotHandled`. Both propagate, signalling "this clause can't be footprinted
as-is": decorate it with `@symbolic` (which lifts `if` into `if_then_else`, so
both branches' atoms are collected) or extend `SmtCollectHandler` to interpret
the op.

Rejected:
- Returning the atoms collected *before* the raise (the old behaviour). It
  produces a partial, unsound footprint — exactly the false-pass above.
- Distinguishing the "tolerable" Z3 bool-cast from a real Z3 bug by matching
  the exception message (`"Boolean" in str(e)`). Neither sound (a message
  change breaks it) nor complete (can't reliably tell the two apart), and it
  still left the truncation unsoundness in place.
