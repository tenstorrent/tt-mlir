# D2M `SplitUnifiedThread` Redesign

Status: Draft / for review
Owner: nsmith
Branch: `nsmith/split-threads`

## 1. Purpose

Replace the current monolithic `D2MSplitUnifiedThread` pass
(`lib/Dialect/D2M/Transforms/SplitUnifiedThread.cpp`) with a sequence of three
small, single-responsibility passes. The primary objective is to **reduce
complexity**: the current pass simultaneously infers synchronization scopes,
wraps compute in a `synchronized_region` op, converts data-movement ops to
explicit-CB form, inserts compute-side CB sync ops, and prunes dead ops per
thread — all in one `matchAndRewrite`.

Secondary objectives that fall out of the decomposition:

- **Eliminate the `synchronized_region` op and the entire scope-inference
  machinery.** It exists only to delimit "where compute lives" so that CB
  wait/pop/reserve/push can be placed. In the new design that scope is derived
  directly and locally from the compute thread after splitting, so the op and
  concept are unnecessary.
- Make thread assignment an explicit, inspectable property (an attribute)
  rather than something re-derived from op-type heuristics at each stage.

## 2. Current architecture (what we are replacing)

`D2MSplitUnifiedThreadRewriter::matchAndRewrite` (one `GenericOp` with a single
`#d2m.thread<unified>` region) does, in order:

1. `wrapComputeInSynchronizedRegion` — walks the region, finds the outermost
   ops enclosing `D2MGenericRegionComputeOpTrait` ops, expands/merges those
   ranges up to `SynchronizableOpInterface` boundaries, traces each
   `memref.load`/`store`/`TileMatmulBlockOp` operand back to a CB
   (`traceComputeMemrefToCB`), and wraps the range in a `SynchronizedRegionOp`
   carrying `consumers`/`producers` operand lists. This is the single largest
   source of complexity and the only reason `synchronized_region` exists.
2. Builds a new 2-region `GenericOp` (`datamovement`, `compute`) and clones
   **all** unified ops into **both** regions.
3. `processSharedBufferPairs` + `insertCBOpsForCompute` — insert
   wait/pop/reserve/push into the compute region, with special cases for
   aliased load/store pairs and fan-out (one wait before first consumer, one
   pop after last).
4. `eraseAliasedLoadStoreOps` + `convertDMAToExplicitCBForm` — on the DMA
   region, erase aliased (no-DMA) load/stores and convert the rest of
   `remote_load`/`remote_store`/`local_copy` to explicit-CB form.
5. `eraseDMAOpsInComputeBlock` + `eraseDeadOps` (×2) — prune each region using
   op-type heuristics (`ShardDMAOpInterface`, `DeviceSynchronizeOp`,
   `SemaphoreWaitOp`) to decide what survives where.
6. `unwrapSynchronizedRegion` — flatten the `synchronized_region` ops back out.

### Why it is complex

- **Scope inference is implicit and global.** Steps 1 and 3 must agree on what
  constitutes a "synchronization scope," and the agreement is encoded in two
  places (the wrap walk and the CB-insertion walk), each with its own
  cross-nest guard producing near-identical diagnostics.
- **`synchronized_region` is scaffolding.** It is created in step 1 only to be
  destroyed in step 6; nothing downstream consumes it.
- **Thread membership is re-derived three times** (clone-to-both, then
  `eraseDMAOpsInComputeBlock`, then `eraseDeadOps` with type heuristics)
  instead of being decided once.
- **Aliased vs. streaming is handled in multiple branches**
  (`processSharedBufferPairs`, `insertCBOpsForCompute`'s
  `aliasedLoadProducer`/`aliasedStoreConsumer` flags, `eraseAliasedLoadStoreOps`).

## 3. Key invariants the redesign must preserve

These are extracted from the existing tests and dialect semantics. The new
flow must keep all of them.

1. **DMA-side remote/local ops are self-synchronizing.** An explicit-CB-form
   `remote_load` embeds reserve+push for the CB it fills; `remote_store` embeds
   wait+pop for the CB it drains; `local_copy` embeds wait+pop on its source CB
   and reserve+push on its destination CB. These are lowered later by
   `LowerLoadStoreOpsToDMA`. Therefore the DMA thread needs **no** explicit
   `d2m.wait/pop/reserve/push`.
2. **The compute thread is the only place that needs explicit CB sync ops**,
   for the CBs its compute ops read (consume) or write (produce).
3. **Streaming load** (DRAM→CB): DMA produces the CB; compute consumes it →
   compute gets `wait … pop`.
4. **Streaming store** (CB→DRAM): compute produces the CB; DMA consumes it →
   compute gets `reserve … push`.
5. **Aliased load/store** (`d2m.operand_alias`, data already in L1, no DMA
   transfer): the CB protocol still must be balanced, but there is no DMA peer.
   Compute therefore performs **both** halves it is missing:
   - aliased load → compute `reserve … push` (stand in for the absent DMA
     producer) in addition to `wait … pop` for its own read;
   - aliased store → compute `wait … pop` (stand in for the absent DMA
     consumer) in addition to `reserve … push` for its own write.
6. **Shared buffer pair, both remote** (load and store reference the same CB
   operand): both ops live on DMA, share one `get_cb`, compute is empty.
7. **Fan-out**: a CB consumed by multiple compute ops *in the same block* gets
   exactly one `wait` before the first consumer and one `pop` after the last —
   never one pair per consumer (would deadlock a depth-limited CB).
8. **Cross-nest fan-out is rejected** with a diagnostic, not miscompiled.
9. **`local_copy` stays entirely on DMA.** Intermediate scratch CBs that only
   flow load→copy→…→copy→compute keep their sync on DMA; compute only
   waits on the CB it actually reads.
10. **Scratch buffers** (`d2m.scratch_buffer`) and **reduction scaler buffers**
    are not treated as synchronized CBs.
11. **`semaphore_wait` (no reset) is replicated into both threads**, preserving
    relative order. `semaphore_wait` *with* reset in a unified region is an
    error (`checkForIllegalSemaphoreOps`).
12. Multicast parameters on `remote_load` and `preallocated_semaphores` /
    semaphore operands on `remote_store` are preserved through the explicit-CB
    conversion.

## 4. Proposed design

Three passes, run back-to-back in the pipeline at the current
`createD2MSplitUnifiedThread()` slot (between `D2MHoistCBAllocs` and
`D2MPreallocateMcastSemaphores` in `D2MPipelines.cpp`):

```
d2m-assign-threads      (Pass 1: annotate + lower to explicit CB form; no split)
d2m-split-threads       (Pass 2: mechanical region split by attribute)
d2m-insert-compute-cb   (Pass 3: compute-local CB sync insertion)
```

Names are provisional. A single umbrella pipeline entry
(`createD2MSplitUnifiedThread()` retained as the public name, internally
scheduling the three) keeps the pipeline diff minimal and the lit RUN lines
stable for callers that invoke the whole thing.

### Core idea

> **The thread a leaf op is assigned to encodes its synchronization
> responsibility.** Once each effectful op carries a thread tag, splitting is
> mechanical, and compute-side CB sync is a purely local walk of the compute
> block. No global scope inference, no `synchronized_region`.

The crucial reframing of the aliased case: instead of erasing aliased
`remote_load`/`remote_store` on the DMA side and special-casing them in compute
CB insertion, **assign aliased ops to the compute thread**. They then survive
into the compute block as `SynchronizableOpInterface` producer/consumer
markers, and Pass 3's generic rule produces the correct extra reserve/push or
wait/pop with no special branch. Pass 3 erases them once consumed.

### Pass 1 — `d2m-assign-threads`

Operates in place on the single unified region. Two responsibilities:

**(a) Lower data-movement ops to explicit-CB form.** Convert each implicit-form
`remote_load`, `remote_store`, and `local_copy` to explicit-CB form by
introducing/reusing a `get_cb(operandIdx)` (via the existing
`d2m::getOrCreateCB`) for each local buffer they touch. This is independent of
splitting and is exactly the body of today's `convertDMAToExplicitCBForm`,
applied to the unified region instead of the post-split DMA block. Multicast /
semaphore / `preallocated_semaphores` attributes are carried over verbatim
(invariant 12).

Aliased load/store (`isAliasedLoad`/`isAliasedStore`) are **not** converted to
explicit CB form — they carry no DMA transfer. They are left as-is and tagged
compute (see below); Pass 3 consumes and erases them.

**(b) Tag each leaf op with its thread.** Add an attribute
`d2m.thread = #d2m.thread<datamovement | compute>` to each *effectful leaf*:

| Op | Thread |
|----|--------|
| streaming `remote_load` / `remote_store` (explicit-CB) | `datamovement` |
| `local_copy` | `datamovement` |
| other `ShardDMAOpInterface` / `DeviceSynchronizeOp` | `datamovement` |
| aliased `remote_load` / `remote_store` | `compute` |
| ops with `D2MGenericRegionComputeOpTrait` and their enclosing `linalg.generic` | `compute` |
| `semaphore_wait` (no reset) | replicated → tag `both` |

Structural / pure ops (`arith.constant`, `scf.for`, `d2m.core_index`,
`view_layout`, `memref.alloc`, `get_cb`, index arithmetic) are **left
untagged**; Pass 2 replicates them to both threads and lets DCE drop the unused
copies. This matches the current "clone all to both, then prune" behaviour but
makes the decision explicit and one-time.

Pass 1 also runs `checkForIllegalSemaphoreOps` (invariant 11) and the scratch /
reduction-scaler exclusions (invariant 10) when classifying compute CB usage —
though note these only matter to Pass 3; see §6.

Output: still one unified region, semantically unchanged, now annotated and in
explicit-CB form for data movement.

### Pass 2 — `d2m-split-threads`

Dead-simple mechanical transform. For a `GenericOp` whose single region is
`#d2m.thread<unified>`:

1. Create a new 2-region `GenericOp` (`datamovement`, `compute`), mapping the
   semaphore block arguments into both blocks (unchanged from today).
2. Clone every unified op into **both** blocks.
3. In the `datamovement` block, erase ops tagged `compute`. In the `compute`
   block, erase ops tagged `datamovement`. Ops tagged `both` or untagged stay
   in both.
4. Run DCE per block to drop now-dead structural ops (e.g. `core_index`,
   `arith.addi`, empty `scf.for` bodies) and unused `get_cb`s.
5. Strip the `d2m.thread` leaf attributes (no longer needed).

No CB insertion, no scope reasoning, no op-type heuristics for survival — the
attribute is the single source of truth. This replaces today's
`eraseDMAOpsInComputeBlock` + `eraseDeadOps` heuristic pruning.

### Pass 3 — `d2m-insert-compute-cb`

Analyses the **compute region in isolation** and inserts CB sync ops.

> **Correction (discovered during stage-1 implementation).** The original plan
> below assumed Pass 3 could find every compute CB user by walking for
> `SynchronizableOpInterface` ops. That is true for the *high-level* compute
> form (`linalg.generic` — which has a `SynchronizableOpInterface` external
> model — and `tile_tilize_block`/`tile_untilize_block`), which is what the
> existing lit tests feed. **But in the real pipeline the compute body reaching
> this pass is already lowered** to `scf.for` nests containing
> `memref.load`/`memref.store` into `#dst` plus `d2m.tile_*` ops (and, for
> matmul, `TileMatmulBlockOp`). Those `scf.for` nests are *not*
> `SynchronizableOpInterface` ops, so a pure interface walk would miss them and
> emit no CB sync — a silent miscompile that the linalg-only lit tests would not
> catch. This lowered form is precisely why `synchronized_region` /
> `traceComputeMemrefToCB` exist: to aggregate a raw loop nest's CB usage into a
> single interface-bearing unit. A lowered-form regression test
> (`split_unified_thread_lowered_loops.mlir`, captured from the real pipeline)
> now guards this.

Revised algorithm. For each compute block, build per-CB ordered
consumer/producer lists from **two** sources, then place ops uniformly:

1. **Interface-bearing units** (`linalg.generic`, `tile_*_block`, and the
   surviving aliased `remote_load`/`remote_store` markers): use `isConsumer`/
   `isProducer` on their operands (today's `insertCBOpsForCompute` already does
   this).
2. **Raw compute spans** (top-level `scf.for` nests / bare `TileMatmulBlockOp`
   that contain `D2MGenericRegionComputeOpTrait` ops but are not themselves
   interface ops): compute their consumed/produced CBs by tracing
   `memref.load`/`memref.store` (and `TileMatmulBlockOp` A/B/output) back to the
   CB via `traceComputeMemrefToCB`. The span's outer op is the placement anchor.
   This is the analysis half of today's `wrapComputeInSynchronizedRegion`,
   reused **without** materializing a `synchronized_region` op. Scratch and
   reduction-scaler buffers are excluded (invariant 10).
3. For each CB with consumers: verify all consumer anchors share a common parent
   block (`commonParentBlock`); otherwise emit the cross-nest diagnostic
   (invariant 8). This subsumes the old global "multiple synchronization scopes"
   guard — in the cross-nest fan-out case the consumer anchors land in distinct
   loop-body blocks, so `commonParentBlock` already rejects it (the diagnostic
   text changes; see D3). Insert one `wait` before the first consumer anchor and
   one `pop` after the last; rewrite that anchor's CB uses to the wait result
   (invariant 7, fan-out).
4. For each CB with producers: same common-block check; insert one `reserve`
   before the first producer anchor and one `push` after the last; rewrite uses
   to the reserve result.
5. Erase any surviving aliased `remote_load`/`remote_store` markers. An aliased
   load is a *producer* (reserve/push) and the compute op reading it is a
   *consumer* (wait/pop) — giving invariant 5's "reserve, push, wait, … , pop"
   with no special-case code. Symmetrically for aliased store.

There is no `synchronized_region` op and no wrap/unwrap dance; the "scope" is the
span between the first and last consumer/producer anchor of each CB. The
form-specific analysis (interface vs. raw-span tracing) is retained as internal
helpers — the *op and concept* are removed, which is the stated goal.

> **Use-threading note.** For raw lowered spans the CB is reached through a
> `memref.collapse_shape`; rewriting the CB use to the `wait`/`reserve` result
> must follow that chain (today's `synchronized_region` threaded it via the
> region block argument). This is the fiddliest part of the rewrite and is
> covered by the lowered-loops regression test.

## 5. Attribute schema

One new op attribute, set by Pass 1, stripped by Pass 2:

- `d2m.thread` : `ThreadAttr` (reuse the existing `#d2m.thread<…>`), values
  `datamovement`, `compute`, or `both`.

No new ops, no new types. We **remove** `D2M_SynchronizedRegionOp` and its
utilities (`wrapInSynchronizedRegion`, `unwrapSynchronizedRegion`,
`isPurelyDerivedOp`) once Pass 3 lands. `SynchronizableOpInterface` itself is
retained — Pass 3 still uses `isProducer`/`isConsumer`.

> Open question (D1): do we need an attribute at all, or can Pass 2/3 re-derive
> thread membership from op type (as today)? Recommendation: keep the explicit
> attribute. It is the mechanism that lets Pass 2 be "dead simple" and lets the
> aliased-as-compute trick work without Pass 3 inspecting the DMA region. The
> cost is one attribute set/strip.

## 6. Worked mapping of existing tests

Demonstrating semantic equivalence for every current test case.

| Test (file) | CBs and roles | New-flow result |
|----|----|----|
| 1 streaming load + mcast | in: DMA-produces, compute-reads; out: compute-produces, DMA-stores | DMA: `remote_load … into cb` (mcast preserved), `remote_store`; compute: `wait`, linalg, `pop`, plus `reserve/push` for out CB. ✓ |
| 2 streaming store | in aliased→compute; out: compute-produces, DMA-stores | DMA: `remote_store … from cb`; compute: `reserve`, linalg, `push`. ✓ |
| 3 / 14b full load+store | in DMA-produces; out compute-produces | DMA: load+store explicit; compute: `wait`,`reserve`,linalg,`push`,`pop`. ✓ (14b confirms no dependence on `d2m.blocking_loop`; the new flow never reads that attr for scoping.) |
| 4 aliased load/store | cb0,cb1 aliased loads→compute; cb2 aliased store→compute | DMA empty of remote ops; compute: per-CB reserve/push (from aliased loads) + wait/pop (from linalg reads) + reserve/push (linalg write of cb2) + wait/pop (aliased store of cb2). ✓ |
| 5 non-unified | — | not matched; unchanged. ✓ |
| 6 multiple loads | cb0,cb1 DMA-produce; cb_out aliased store→compute | DMA: two loads; compute: two waits, linalg, two pops (+ reserve/push/wait/pop for cb_out). ✓ |
| 7 shared pair both remote | one CB, DMA-produces and DMA-consumes | both ops share `get_cb(2)` on DMA; compute empty. ✓ |
| 8 load remote / store local(aliased) | cb DMA-produces (load), aliased store→compute | DMA: load; compute: `wait`,`pop` (aliased store consumer). ✓ |
| 9 load local(aliased) / store remote | aliased load→compute, store DMA-consumes | DMA: store from `get_cb(2)`; compute: `reserve`,`push`. ✓ |
| 10 L1→L1 shared | one CB DMA-produces and DMA-consumes | DMA: load+store; compute empty. ✓ |
| 11 load→copy→compute | cb_buf DMA-produce+DMA-consume(copy); scratch DMA-produce(copy)+compute-read; cb_out compute-produce+DMA-store | DMA: load, copy, store; compute: `wait` scratch, linalg, `pop` (+reserve/push cb_out). Source pop "deferred to DM" is automatic — compute never touches cb_buf. ✓ |
| 12 copy chain | all intermediates DMA-only; final scratch2 compute-read | DMA: load+copy+copy+store; compute: wait scratch2, linalg, pop. ✓ |
| 13 copy→store | all DMA | DMA: load+copy+store; compute empty. ✓ |
| 14 compute→copy→compute | cb_buf DMA-produce; compute_scratch compute-produce+DMA-consume(copy); copy_dst DMA-produce(copy)+compute-read; cb_out compute-produce+DMA-store | DMA: load, copy, store; compute: wait/reserve/linalg/push/pop ×2. ✓ |
| fanout (file 2) | cb_in DMA-produce, two compute consumers same nest | one `wait` before first, one `pop` after last. ✓ |
| fanout_unsupported | cb_in consumers in two distinct nests | `commonParentBlock` fails → diagnostic. ✓ (Message text will change; see §8.) |
| semaphore_wait ×2 | as labelled | `semaphore_wait` tagged `both`, replicated, order preserved; CB ops as above. ✓ |

## 7. Risks and open questions

- **D1 (attribute vs. re-derivation)** — see §5. Recommend explicit attribute.
- **D2 (instruction ordering / CHECK churn) — RESOLVED: semantic equivalence,
  not byte-identical.** Pass 3 places `wait`/`reserve` at the *first*
  producer/consumer and `pop`/`push` after the *last*, the same policy as today,
  so ordering should match in the common cases. Where the new single-pass
  ordering legitimately differs (e.g. the aliased "reserve, push, wait, …"
  interleaving in test 4, produced today by `processSharedBufferPairs` running
  before `insertCBOpsForCompute`), we **update the CHECK lines**. The hard
  requirement is CB-protocol correctness (every reserve has a matching push,
  every wait a matching pop, fan-out collapses to one pair), not textual
  identity.
- **D3 (diagnostic wording)** — the cross-nest error message moves from
  `wrapComputeInSynchronizedRegion` to Pass 3. `fanout_unsupported` and any
  scope-related diagnostics' `expected-error` strings must be updated to the new
  text. The current pass also emits "compute ops span multiple synchronization
  scopes" from the wrap step; Pass 3's equivalent is the per-CB
  `commonParentBlock` check. We should keep a single, clear message.
- **D4 (shared-CB pairing across different operands) — RESOLVED: not allowed.**
  A CB is never shared between different operands. Each remote/local op converts
  via its own local-buffer operand index, and two ops share a `get_cb` only when
  they reference the *same* operand (e.g. shared pair tests 7/10). Pass 1 may
  assert this invariant. The stale "use the output operand's CB" comment in the
  current pass should be dropped.
- **D5 (intermediate IR validity) — RESOLVED: atomic pipeline step.** After
  Pass 1 the unified region contains explicit-CB-form remote ops *and* thread
  attributes but is not yet split. Passes 1–3 run as an atomic internal pipeline
  step (verify only at the end), so we do not loosen op verifiers to accept
  explicit-CB-form ops inside a `#d2m.thread<unified>` region. The public
  `createD2MSplitUnifiedThread()` schedules all three.
- **D6 (`local_copy` in compute)** — confirm `local_copy` is never legitimately
  a compute op; the design assumes it is always `datamovement`.

## 8. Test plan

### 8.1 Regression (must continue to pass)

Run all existing pass tests unchanged where possible; update only CHECK
ordering / diagnostic text per decisions D2/D3:

- `test/ttmlir/Dialect/D2M/Transforms/split_unified_thread.mlir` (14 cases)
- `…/split_unified_thread_fanout.mlir`
- `…/split_unified_thread_fanout_unsupported.mlir` (update `expected-error`
  text to the new Pass 3 message)
- `…/split_unified_thread_semaphore_wait.mlir`
- Indirect: `lower_load_store_ops_sharded_to_interleaved.mlir`,
  `insert_dst_register_access_scheduled_fallback.mlir`,
  `allocate_output_alloc_in_compute_loop.mlir`
- End-to-end: `test/python/.../test_matmul.py` and `check-ttmlir`, plus
  `check-perf` (resnet/yolo_v8/segformer) if `TTMLIR_ENABLE_OPMODEL=ON`, to
  catch silicon-affecting regressions.

Command: `cmake --build ${BUILD_DIR} --target check-ttmlir`.

### 8.2 New per-pass unit tests

Because each pass is independently invocable, add focused lit tests that pin the
contract at each boundary. Provisional RUN lines:

**Pass 1 (`d2m-assign-threads`)** — `split_unified_assign_threads.mlir`:
- streaming load: assert `remote_load` is in explicit-CB form with a `get_cb`
  and carries `d2m.thread = #d2m.thread<datamovement>`; region still single &
  `unified`; mcast attrs preserved.
- streaming store: `remote_store` tagged `datamovement`.
- aliased load/store: op left in implicit form, tagged
  `d2m.thread = #d2m.thread<compute>`.
- compute op (`linalg.generic` + `tile_exp`): tagged `compute`.
- `semaphore_wait` (no reset): tagged `both`; with reset: `expected-error`.
- `local_copy`: explicit-CB form, tagged `datamovement`.
- negative: non-unified generic untouched.

**Pass 2 (`d2m-split-threads`)** — `split_unified_split_threads.mlir`:
Feed pre-annotated IR (output shape of Pass 1) and assert:
- two regions `[datamovement, compute]` created; semaphore block args mapped to
  both.
- ops tagged `datamovement` only in region 0; `compute` only in region 1;
  `both` in both.
- structural ops (`core_index`, `arith`, empty loops) DCE'd from the region that
  no longer uses them.
- `d2m.thread` attributes stripped from the output.
- no CB sync ops inserted by this pass (CHECK-NOT `d2m.wait/pop/reserve/push`
  beyond those already present).

**Pass 3 (`d2m-insert-compute-cb`)** — `split_unified_insert_compute_cb.mlir`:
Feed a split-but-unsynchronized compute region and assert:
- single consumer → one `wait`/`pop`; single producer → one `reserve`/`push`.
- fan-out (two consumers, one nest) → exactly one `wait`/`pop` pair (port the
  `CHECK-NOT d2m.wait %[[IN]]` style assertions from the existing fanout test).
- cross-nest consumers → `expected-error` (new message).
- aliased load marker present → emits reserve/push + wait/pop and erases the
  marker; aliased store marker → wait/pop + the linalg's reserve/push.
- scratch / reduction-scaler buffers → no CB ops emitted for them.

### 8.3 Equivalence harness (recommended during development)

For each function in the existing `.mlir` corpus, diff the final IR of the old
single pass vs. the new 3-pass pipeline (normalising SSA names). Any diff that
is not a known/justified ordering change (D2) or message change (D3) is a bug.
This gives high confidence before we delete `synchronized_region`.

### 8.4 Cleanup verification

After Pass 3 lands and tests are green, the split flow no longer creates or
consumes `synchronized_region` (`traceComputeMemrefToCB` is also gone, replaced
by `traceCBUse`). **However, the op cannot simply be deleted from the dialect:**
a `grep` shows other consumers independent of thread splitting —

- `D2MMarkSynchronizedBuffers` (live pipeline pass): its lit fixtures
  (`mark_synchronized_buffers.mlir`) use `synchronized_region` to express CB
  producer/consumer roles for `tile_matmul_block` / `tile_reduce_sum` (compute
  that `getCBUsageInfo` cannot otherwise see, since those ops are not
  `SynchronizableOpInterface`). The pass itself only calls `getCBUsageInfo`, so
  it does not reference the op type, but its tests rely on the op.
- `TestD2MGenericAnalysis.cpp` unit test `CanWrapAndUnwrapSynchronizedRegion`
  directly exercises `wrapInSynchronizedRegion` / `unwrapSynchronizedRegion`.

**Done.** `D2M_SynchronizedRegionOp` and `wrapInSynchronizedRegion` /
`unwrapSynchronizedRegion` / `isPurelyDerivedOp` are deleted, along with the
`CanWrapAndUnwrapSynchronizedRegion` unit test. The `mark_synchronized_buffers`
fixtures were rewritten from synthetic `synchronized_region` to `linalg.generic`
(matmul / reduce) — the form that actually reaches `MarkSynchronizedBuffers`
(which runs before tile-matmul lowering), so the test now matches the real
pipeline and still covers the accumulating-compute → single-buffer path.
`traceComputeMemrefToCB` is also gone (replaced by `traceCBUse`). No
`synchronized_region` references remain outside historical comments.

## 9. Implementation order (suggested)

1. ✅ **Done.** Land Pass 1 + 2 behind the existing public pass name, keeping the
   compute-CB insertion (the `synchronized_region` bridge) temporarily in Pass
   2's tail, so pipeline output is unchanged. (`d28bfc67b9`)
2. ✅ **Done.** Add the lowered-loops regression test — the real-pipeline compute
   form was uncovered by lit. (`2c8ffbca58`)
3. ✅ **Done.** Move explicit-CB lowering into Pass 1 (`d2m-assign-threads`), and
   rework the aliased handling to be **peer-independent** so it survives the
   conversion (and is reusable by Pass 3): `eraseAliasedLoadStoreOps` drops any
   aliased op from the DMA thread; `insertStandaloneAliasedCBs` replaces
   `processSharedBufferPairs`, emitting wait/pop (aliased store) or reserve/push
   (aliased load) for aliased ops whose CB has no compute-side peer. (`8889e2ace1`)
4. **Next.** Replace Pass 2's bridge with the real compute-local CB insertion
   (§4 Pass 3): keep the cross-scope guard, keep the analysis (interface ops +
   raw-span tracing), drop the `synchronized_region` op. The peer-independent
   aliased handling from step 3 is already in place.
5. ✅ **Done.** Replace Pass 2's bridge with the compute-local CB insertion
   (`cc5abf6d15`): cross-scope guard kept as `checkComputeSyncScope`, analysis
   kept as interface-ops + raw-span `traceCBUse`, `synchronized_region` no longer
   created. Validated against linalg-form + lowered-loops + fanout lit tests, and
   `simple_eltwise`/`simple_max` through the full pipeline to flatbuffer.
6. ✅ **Done.** Deleted the `synchronized_region` op + utils + unit test, rewrote
   the `mark_synchronized_buffers` fixtures (`8f1f6f3ad9`). See §8.4.

7. ✅ **Done.** Split the compute-CB insertion into its own pass
   `d2m-insert-compute-cb` (`5671b6e0e9`), completing the 3-pass decomposition:
   `assign-threads` → `split-threads` (purely mechanical) → `insert-compute-cb`.

   - **Verifier constraint:** `remote_load`/`store` must live on a datamovement
     (or unified) region, so aliased remote ops cannot persist in the compute
     region across a pass boundary. They are assigned to the **datamovement**
     thread (where remote ops belong); `insert-compute-cb` inspects them there —
     they reference the same CB values the compute region uses — supplies the
     missing compute-side half, and erases them.
   - **Tags load-bearing (partial):** `split-threads`'s compute-side cleanup is
     now tag-driven (drop `datamovement`-tagged ops + trivial DCE). The
     datamovement-side cleanup stays heuristic (dead-op elimination), because a
     lowered compute nest's structural ops (`memref.load/store` into `#dst`,
     `scf.for`, `acquire_dst`) are untagged and side-effecting — removing them
     from the DM thread by tag alone isn't possible without replicating span
     analysis into `assign-threads`. So full tag-driven split is *not* a clean
     win; the heuristic is retained where it is genuinely needed.

> **Note on tags becoming load-bearing.** Pass 2 still ignores the `d2m.thread`
> tags (it clones-all-to-both + heuristic-erases). Making the split purely
> tag-driven is a further simplification that can land with or after step 4.
