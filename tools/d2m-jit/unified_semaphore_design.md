# Supporting semaphore ops in unified-split kernels

Design notes for safely handling `semaphore_set` / `semaphore_inc` /
`semaphore_wait(reset=...)` once `@d2m.kernel` is unified-only. Companion to the
`TODO(#unified-semaphores)` in `lib/Dialect/D2M/Utils/DMAUtils.cpp`.

## The insight that simplifies everything

All the split threads (compute = TRISC, datamovement = BRISC/NCRISC) run on the
**same Tensix core and share L1**. A semaphore is just an L1 address. So:

- **Observation is free.** Any thread can `semaphore_wait` on a semaphore another
  thread mutates — they poll the same L1 word. Replicating a *wait* across threads
  is already safe and is exactly what the split does today (the simple all_gather
  replicates `semaphore_wait(end_sem)` into both compute and DM and it works).
- **Only two things actually break:** (1) a *mutation* (`inc`/`set`, and the reset
  half of `wait`-with-reset) ending up wrong — either executed N times (e.g.
  `device_synchronize`, which ScheduleDMA clones across DM threads) or, for
  explicit `semaphore_inc`/`set`, *misplaced onto the compute thread* (today's
  `split-unified-thread-v2` moves them there — they're neither `isDMAOp` nor
  `isReplicated` — where TRISC has no NOC/fabric access); and (2) *ordering* — a
  thread acting before/after the mutation when the split dropped the program-order
  edge that guaranteed it.

So "put the mutation on exactly one DM thread" solves (1). The remaining question
is (2): re-establishing the cross-thread happens-before edges that program order
gave us for free in the unified kernel.

**Implemented so far (the Option-D mitigation):** `split-unified-thread-v2` now
classifies `semaphore_inc`/`set` as DM-resident (kept on datamovement, erased
from compute), and `ScheduleDMA` keeps any kernel containing an explicit
`semaphore_inc`/`set` on a *single* DM thread (no NOC split) so the mutation runs
exactly once. This covers the inc-then-wait pattern and end-of-kernel resets that
no other thread re-waits on. It does **not** yet handle (2) for a reset that a
*different* thread concurrently waits on — that still needs the exit-barrier /
per-edge work below.

## First, decompose `wait`-with-reset away

`semaphore_wait(S, N, reset=v)` is just `semaphore_wait(S, N)` followed by
`semaphore_set(S, v)` on the same thread. Lower it to exactly that as an **early
step, before the split** (it's what D2MToTTKernel already does at the very end —
just hoist it earlier). This eliminates the reset-bearing wait variant entirely:
the split never sees it, and the verifier's special wait-with-reset branch goes
away too. What's left is the plain wait (a pure read) plus a plain `set` (a
mutation), both of which fall under the two general rules below — no special case.

The only thing that carries over is that the decomposed `set` is an ordinary
mutation, so it obeys the ordinary mutation rules: pinned to one owner, and
ordered after any replicated waits on the same semaphore (handled by the general
edge mechanism / the exit barrier — see below). In the common case where the
reset only preps the owner's own next iteration and no other thread re-waits on
`S`, program order on the owner thread already supplies that edge and no extra
barrier is needed.

## Taxonomy (drives the transform)

After the decomposition above there are only two cases:

| Op | Under replication | Transform |
|---|---|---|
| `semaphore_wait` (no reset) | safe (pure read) | replicate to every thread that must block |
| `semaphore_inc` / `semaphore_set` | **N× mutation** | pin to one owner thread, order vs observers |
| `device_synchronize` | N× fabric inc | pin to one owner thread (done) |

## Approaches for the cross-thread ordering (the real question)

### A. Lean on the existing CB dataflow edges (what `device_synchronize` does now)

The barrier is pinned to the store-owning DM thread and is transitively ordered
w.r.t. the other threads through the `push/pop/wait/reserve` CB handshakes that
the split already inserts (the writer's send is gated by the barrier on its own
thread; reader/compute are gated through CBs). Cost: zero new machinery. Limit:
only correct when the semaphore op's required ordering *coincides* with an
existing CB edge. It does for the barrier; it won't in general (e.g. a mid-kernel
`set` that compute must observe but that has no data dependency).

### B. Local-semaphore arrive/wait barrier (the general primitive)

Allocate a local semaphore `L`. At the sync point, each non-owner thread does
`inc(L)` ("arrived"); the owner does `wait(L, num_other_threads)`, performs the
global mutation, then (for reset) `set(L,0)`. This is the sanctioned tt-metal
cross-RISC sync (`noc_semaphore_inc/wait`, with the fencing those provide —
important, since raw L1 writes between RISCs aren't ordered without it). It
directly implements "local-barrier the other threads with the semaphore thread."
Cost: a local sem + a handshake per edge. This is the workhorse; A is an
optimization over it.

### C. CB-token barrier (reuse the infra you already trust)

Model each required ordering edge as a 1-element dummy CB: the owner `push`es
after the mutation, observers `wait`/`pop` before they depend on the
post-mutation state. It's local semaphores under the hood, but it rides the
existing CB scheduler/allocator and the split's existing handshake-insertion
code, so it may be *less new code* than B even though it's the same primitive.
Downside: synthetic CBs are a bit of an abstraction stretch and cost a CB slot.

### D. Don't split the DM region for semaphore-bearing kernels (the cheap correct fallback)

If a kernel contains explicit `set`/`inc`/`wait`-reset, have `ScheduleDMA` keep
datamovement on a **single** DM thread (skip the NOC-processor split). Then every
semaphore op is naturally on one thread, compute still gets its own thread + CB
handshakes, and you only ever replicate *waits*. This is the proven-working shape
(the simple unified all_gather). You lose dual-NOC datamovement parallelism — but
CCL datamovement is fabric-bound, so that's usually not the bottleneck. Strictly
weaker than A+B but a one-conditional change and obviously correct.

## How to place the edges

The split already maps each original op to its per-thread clones, so original
**program order is recoverable**. Two granularities:

- **Coarse (ship first):** bracket the generic with one entry barrier and one exit
  barrier, pin all mutations to the owner in original relative order, decompose
  `wait`-reset. Correct, over-synchronizes slightly.
- **Precise (later):** build a happens-before graph from program order; emit a
  local-sem edge only where a mutation on the owner is observed by another thread
  *and* no existing CB edge already covers it (i.e. A subsumes B per-edge). This
  keeps the common CCL path barrier-free.

## Recommendation

Tiered, and it slots into the two passes we already have:

0. **Decompose `wait`-with-reset → `wait` + `set` early** (before the split).
   Removes the only special-case op; everything downstream sees just waits and
   mutations.

1. **Now / minimal:** keep the permissive verifier; in `ScheduleDMA`, extend the
   `device_synchronize` pinning to *all* mutating sem ops (`inc`/`set`), choosing
   the owner the same way (the DM thread owning the related store/CB). Add **one
   exit local-barrier** before any owner-side `set`, so a reset that preps the
   next iteration can't fire while another thread is still waiting on that
   semaphore (covers `semaphore_set(start_sem,0)`-style resets, the only explicit
   mutation in the current kernels). This is small and makes today's CCL kernels
   correct, not just accepted.

2. **Then:** generalize to Approach B with program-order-derived edges, falling
   back to A where a CB edge already encodes the order. Use **D as the safety
   valve** — if the dependency analysis can't prove an edge is covered, refuse to
   NOC-split that kernel rather than emit a subtly-wrong barrier.

3. **Primitive choice:** prefer **B** (explicit local semaphores) for clarity, but
   prototype with **C** if reusing the CB handshake-insertion code lets you land it
   faster — they're the same hardware mechanism.

With `wait`-with-reset decomposed away, the whole model is just two rules:
**replicate waits, pin mutations** — plus an ordering edge from a pinned mutation
to any thread that observes it (a reset that another thread re-waits on is the
case to cover with a test).

---

## Producer-done signals (matmul → core_read gather) — empirical update

Wiring `core_read` into a fused matmul→all_gather surfaced a concrete instance
of the placement problem, and an empirical hardware constraint that rules out
one of the obvious fixes.

### The kernel and the bug

```python
c = a @ b                                 # per-core matmul tile (compute thread)
semaphore_inc(ready, 1, core=[0, 0])      # "my tile is ready" -> injector
if injector:
    semaphore_wait(ready, NUM_CORES)
    for j: core_read(g[j], c, core=[0, j]); ...  # gather every core's tile (DM)
```

Compiles and runs (no hang) but returns `inf`. After the split, each producer
core's `semaphore_inc(ready)` lands on the **DM** thread right after the *input*
DMA — **before** the compute-thread matmul has written `c`. So `ready` reaches
`NUM_CORES` while tiles are still uncomputed, and the injector's `core_read`
gathers stale/uninitialized L1.

The matmul output `c` lowers to a **raw, unsynchronized shared buffer**
(`get_arg`, `cb_layout`, *no* reserve/push/wait/pop). There is no compute→DM
handshake on it at all — nothing the readiness signal can be ordered behind.

### Dead-end: place the inc on the compute thread

The tempting fix — route a `compute_signal`-tagged inc onto the compute thread
(after the matmul) — was prototyped (discardable `d2m.compute_signal` attr +
`collectOpsToErase` routing). The IR was correct (inc placed after
`tile_matmul_block`), but the **kernel does not compile**:

```
dataflow_api_common.h: error: 'NOC_INDEX' was not declared in this scope
dataflow_api.h: error: redefinition of 'get_absolute_logical_x()' ...
```

**TRISC (compute) has no NOC.** A NOC `semaphore_inc` pulls in `dataflow_api.h`,
which needs `NOC_INDEX` — undefined in the compute environment. There is no NOC
semaphore API in any `*trisc*` firmware. So a NOC semaphore op simply cannot be
emitted from the compute kernel. (This is *why* tt-metal's
`MatmulFusedOpSignaler` signals from the **writer/DM** kernel, not the matmul
compute kernel — the writer does `cb_wait_front` on the matmul output CB, then
the NOC inc.) The prototype was reverted.

### Correct fix: per-core compute→DM fence on the matmul output CB

Keep the readiness inc on the **DM** thread (it must, for the NOC), but **fence**
it behind the local matmul, matching tt-metal:

1. Make the matmul output a real compute→DM handshake CB on **every** producer
   core: compute `reserve`/`push`, DM `wait`/`pop`. Today it is a raw shared
   buffer because its only data-consumer (`core_read`) is **cross-core on the
   injector** — producer cores have no local consumer, so the handshake is never
   synthesized. The fence must be driven by the *readiness inc itself* acting as
   a local DM consumer of the matmul output, not by the cross-core `core_read`.
2. Order the `semaphore_inc(ready)` after that local DM `wait`. Now `ready`
   counts *completed* matmuls; the injector's cross-core `core_read` is safe
   (the `ready` semaphore covers the cross-core edge; the local fence covers the
   per-core compute→DM edge).

To associate the inc with the matmul output it fences, the cleanest surface is
an explicit fence operand on `semaphore_inc` (e.g. `semaphore_inc(ready, 1,
core=[0,0], after=c)`) so the dependency is in the IR; `split-unified-thread-v2`
then treats the inc as a local DM consumer of `c`'s CB and synthesizes the
handshake. This is the "ordering edge from a pinned mutation to the producing
thread" from the Recommendation above, specialized to producer-done signals.

**Status:** characterized and validated on device (the `inf` repro); the fence
is not yet implemented. `core_read`/`core_write` themselves are correct — this
is purely the semaphore-ordering edge for fused producer→consumer kernels.
