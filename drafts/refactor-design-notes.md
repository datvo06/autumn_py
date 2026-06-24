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

## Module rationale (kept out of the source docstrings)

The per-module design context, so the source docstrings can state only what
each module *is*:

- **`smt.py`** — mirrors RoboTL's `solve_init` pattern adapted to a
  finite-domain SMT setting: a collection handler produces (free variables,
  constraints) and a backend solver consumes them to return a model or UNSAT.
  Encoding: each StateVar `name` is a Z3 function `name : Int -> Sort`; a read
  at tick `t` is `name(t)`, a write `set_var(name, v)` records `name(t+1) = v`,
  and the per-tick transition rules are universally quantified over `t` at
  finalization. The StateVars and the `sample_uniform` existentials *are* the
  symbolic variables — no parallel namespace — so goals are plain Python
  operators over Z3 expressions.
- **`properties.py`** — borrows Dafny's inline-spec ergonomics (a spec sits
  next to the code it constrains), but the mechanism is a decorator →
  `Spec` → `Goal` → `gate()` pipeline, not Dafny's WP→Boogie→Z3. See the
  Dafny/JML/decorator comparison if revisiting the design.
- **`gate.py`** — concrete instantiation of
  `drafts/autumn-pl-handlers-and-properties.md` Part 1 D4. Goal shapes map to
  the doc's properties (FootprintExclude ≈ P_1/P_4/P_6/P_8/P_12;
  ModularArithmetic ≈ P_3/P_11). Library-conditional shapes (invariant,
  existential) wait on the typed goal/idiom library in §2.5 of that draft.
- **`inference.py`** — the §2 typing rules as a handler-stack computation
  ("same evaluator, different handlers": a term evaluates to a *type token*
  under `TypeOfHandler`). Closes §2.2's `uniformChoice : List T -> T` gap that
  the runtime `_check_type` alone can't enforce.
- **`_ast_rewrite.py`** — a pragmatic stand-in until effectful PR #288
  (eb-disassembler) grows statement-level support; until then we source-rewrite
  code we can read off disk. The lift is observably sound only because every
  supported shape factors side effects out of the conditional (assignments,
  returns, or one same-callable call) — `if_then_else` evaluates *both*
  branches eagerly, unlike Python's short-circuiting `if`.

## Spec-system design vs. LemmaScript

A design comparison of two ways to attach machine-checkable specs to code, and
where each is *leaner*. "Leaner" is multi-axis — it inverts depending on whether
you count code-you-write, surface-you-depend-on, or work-you-still-owe a human.
(LemmaScript claims are from its README, github.com/midspiral/LemmaScript, and
may not reflect internals.)

**The two systems.** *LemmaScript*: specs are `//@` comments in TypeScript
(`requires`/`ensures`/`invariant`/`decreases`/`type`); a transpiler lowers them
to **Dafny or Lean 4** and the *target prover* discharges them (Dafny's
Boogie→Z3, or Lean's tactics), with the human supplying proofs in a companion
file. *`autumn_py`*: specs are **decorators** (`@spec`/`@modifies`/
`@no_stochastic`) that attach a live `Spec` to a next-clause; `@program`
realizes them into typed `Goal`s; `gate()` discharges each **in-process** by
footprint (`read_set`), bounded SMT (Z3 unroll), or a concrete walk — no human
proofs.

| Dimension | LemmaScript | `autumn_py` | Leaner |
|---|---|---|---|
| Authoring surface | `//@` comment DSL | Python decorators (live objects) | `autumn_py` — reuses host syntax |
| Ingestion / parser | bespoke `//@` parser | none — interpreter evaluates decorators | **`autumn_py`** |
| Checking code *you write* | almost none — transpile + delegate | ~1.1k LOC of bespoke checkers¹ | **LemmaScript** |
| External surface *you depend on* | transpiler + Dafny (Boogie+Z3) or Lean 4 + tactics | `effectful`; `z3-solver` optional² | **`autumn_py`** — by a wide margin |
| Verification power | full inductive functional verification | bounded SMT + syntactic + concrete | LemmaScript (the cost of leanness) |
| Human proof effort | writes lemmas/tactics | none — auto-discharged within power | **`autumn_py`** |
| Specs as data | comments → text, erased | live, composable (`Spec.merge`), introspectable | `autumn_py` (capability) |
| Add a new spec kind | extend DSL **and** transpilation | add a `Goal` + checker + `Spec` field, in-repo | **`autumn_py`** |
| Drift / failure mode | comments invisible to `tsc` | decorators are real code, can't be ignored | `autumn_py` |
| Operational cost | transpiler + heavyweight prover per check | run Python; Z3 only for `invariant` | **`autumn_py`** |

¹ `properties.py` 254 + `gate.py` 386 + `smt.py` 323 + `_ast_rewrite.py` 159 ≈
**1,122 LOC**, self-contained in-repo (plus `inference.py` 220 for type
inference). ² `pyproject.toml`: runtime dep is `effectful` alone; `z3-solver`
is an optional extra needed only for `ModularArithmeticGoal`; footprint and
trajectory goals need no solver.

**Where the leanness inverts.** Separate two kinds of lean:

> **`autumn_py` is the leaner *system*** — fewer moving parts, ~1.5
> dependencies, no proof burden, specs-as-data. **LemmaScript is the leaner
> *checker*** — it offloads verification to a real prover and writes almost
> none itself, at the cost of a large trusted base and human proofs.

`autumn_py` minimizes *trusted surface and operational cost* but reimplements a
small bespoke checker and buys only **bounded** guarantees. LemmaScript
minimizes *verification code it owns* but stands on a heavyweight prover and a
human in the loop, for **unbounded, inductive** guarantees. They optimize
opposite columns of the same table — pick by whether you want the leaner system
to embed/operate (`autumn_py`) or the leaner path to a strong guarantee
(LemmaScript).

## Open design items

Captured here so they aren't lost; none is a cleanup — each is a real decision.

- **Bounded → unbounded `@spec(invariant=...)`.** `ModularArithmeticGoal`
  unrolls the recurrence to a fixed `horizon` (`gate._check_modular`), so it
  proves the invariant only up to that depth — where LemmaScript inherits
  unbounded soundness from the prover. Closing the gap *without* leaving the
  lean-system column means adding an inductive step to the gate: prove
  `init ⟹ inv(0)` and `inv(k) ∧ transition ⟹ inv(k+1)` (k-induction with a
  small `k` for strength) over the Z3 encoding already produced by
  `collect_smt`, instead of/in addition to the bounded conjunction. This is a
  new goal shape (or a flag on the existing one), not a tweak.
- **`@symbolic` zero-pollution (EB-10 residual).** The private-name fix removed
  the collision land mine but still writes one idempotent private key into the
  decorated module's globals. A zero-write version would bind the op as a
  default argument on the rewritten function (a local, baked into
  `__defaults__`) instead of a global — needs AST signature surgery (inject a
  keyword-only param, rewrite calls to it). Judged not worth the complexity for
  one private symbol; recorded in case that calculus changes.
