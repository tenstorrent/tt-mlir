# Moving `distributed_rms_norm` sharding from the workaround to the optimizer (opt-2)

Handoff notes for the layout-optimizer developers. Branch:
`mvasiljevic/5738-distributed-rmsnorm-rulebook`.

> **Status: working on Wormhole and Blackhole.** All pieces are in and validated: lit tests
> pass, and the llama-3.1-70B 1-layer opt-2 run on qb2 (Blackhole, 2×2 mesh, `--pcc-decode`)
> **passes** (decode PCC ≥ 0.94). The Blackhole fix required, beyond the rulebook + gating +
> op-model proxy, two more things described below: modeling the fused op's real input
> constraint in the op-model, and a full-bbox/even prune in reshard-candidate generation.

## Blackhole bring-up (what it took beyond the base recipe)

The base recipe (rulebook + opt<2 gating + op-model proxy) worked on Wormhole but produced an
inconsistent config on Blackhole: an interleaved-DRAM input paired with a sharded program
config, crashing the fused kernel (`bad optional access` / `Bad StatusOr access: INTERNAL 13`,
sometimes a segfault depending on stale L1). Three defects, all now fixed on this branch:

1. **Invalid output hint (`getOutputHints` divisibility).** The hint accepted *any* full-bbox
   width-sharded rectangle without checking the core count divides the width-tile count. For
   the 4096-wide (128-tile) decode norm it emitted an **11×2 = 22-core** shard (22 ∤ 128 →
   padded to 22×6 = 132). Fixed: `getOutputHints` now keeps only even divisor shards
   (`isEvenWidthShard`), so the emitted output/config is a valid **1×8 = 8-core** shard
   (8×16 = 128).

2. **Op-model too permissive (`OpModel<DistributedRMSNormOp>`).** The proxy queried plain
   `::ttnn::rms_norm`, which *accepts interleaved input*, so the beam search legally chose the
   cheaper interleaved-input / sharded-output config — which the fused kernel cannot run.
   Fixed: the op-model now rejects any non-width-sharded-L1 input, matching the fused op's real
   constraint, so that config is illegal and the search must width-shard the input.

3. **Valid shard crowded out of reshard candidates (shared reshard generation).**
   `generateReshardCandidates` keeps only the top `maxReshardCandidatesPerType` (=4) width
   shards **by grid volume**, then filters. On the 11-wide grid the *valid* `1×8` shard
   (volume 8) was crowded out of the top-4 by higher-volume shards that the op-model then
   rejected, leaving only the interleaved fallback. Fixed with an **opt-in** rulebook flag
   `OpRuleBook::requiresFullBboxShardedInput()` (default false; `DistributedRMSNormOp` returns
   true): when set, `generateReshardCandidates` prunes — *before* the volume cap — any shard
   that would leave uninitialized cells in the kernel's bounding-box reduction rectangle,
   i.e. **phantom cores** (non-full-bbox, via `isFullBboxSharded`) and, for width shards, a
   **padded tail** (uneven division). Zero blast radius for other ops (the prune only runs
   when the op opts in). This is the same physical invariant the runtime validator enforces
   (tt-metal#50979 / tt-xla#5738): the fused reduction reads the whole bbox rectangle, so every
   cell must hold real data.

Also fixed an operand-indexing bug: with `residual` absent (`operandSegmentSizes <1,1,0,…>`),
flat operand 2 is the f32 `stats` tensor, not residual — so the width-shard filter must not be
applied to it. The filter now constrains only operand 0 (input, width-sharded L1) and
operand 1 (weight, ROW_MAJOR).

### Remaining perf note (not a blocker)

The optimizer currently lands the correct-but-modest **1×8 = 8-core** shard, whereas the
workaround hand-places **64 cores (8×8)**. Reaching 64 needs explicit-rectangular placement in
`LegalTensorLayoutAnalysis` (canonical row-major placement can only form a full-bbox *even*
divisor of 128 up to 8 cores on an 11-wide grid). This is a throughput refinement on a
correct, passing baseline.

## Goal

Today `DistributedRMSNormWidthShardInputRewritePattern` (a TTNN workaround) hand-picks the
fused norm's width-shard grid + `LayerNormShardedMultiCoreProgramConfig` at **all** opt
levels. We want:

- **opt-level < 2**: keep the workaround (hand-rolled grid).
- **opt-level 2**: the optimizer owns the layout — choose the width-shard grid and program
  config via the beam search + a rulebook, like every other sharded op.

## Background (why this op is special)

`distributed_rms_norm` (fused `rms_allgather` / `fused_rms_minimal`) is width-sharded L1,
and its stats reduction runs over the shard grid's **bounding-box rectangle**. A
non-rectangular shard grid leaves "phantom" cells that the reduction reads as uninitialized
L1 → silent PCC≈0 on Blackhole (tenstorrent/tt-xla#5738; runtime guard added in tt-metal
via the validator PR, issue #50979). The tt-mlir fix already on this branch
(`4a2ff97`) makes the *workaround* emit a rectangular grid. The optimizer path must
preserve that invariant: **only full-bbox width-sharded L1 layouts are legal for this op.**

## The migration recipe (from `PagedUpdateCacheOp`, the precedent)

`PagedUpdateCacheOp` sits in the same `optimizationLevel < 2` workaround gate and was moved
to the optimizer with exactly three pieces. This op needs the same three:

1. **Rulebook** — `OpRuleBook` subclass: input/output layout filters + hints, program-config
   generation, score adjustment.  ✅ done here.
2. **Gating** — move the workaround pattern into the `optimizationLevel < 2` block.  ✅ done
   here (WIP).
3. **Op-model** — `OpModel<Op>` so the beam search can *validate* the sharded candidates.
   ✅ done here as a **proxy** to `::ttnn::rms_norm` (details below). This was the original
   blocker: `DistributedRMSNormOp` was `OpModelExempt` like **every** CCL op (`all_gather`,
   `all_reduce`, `reduce_scatter`, `point_to_point`, …) — none can be graph-captured on the
   fabric-less mock device (tt-metal#44748). `distributed_rms_norm` is special: its layout
   legality equals a *modeled single-device op* (`rms_norm`), so a proxy works. Pure
   data-movement CCL ops have no such local-compute equivalent, so this does not generalize
   to them.

## What is implemented on this branch

- `4a2ff97` — rectangular-grid fix in the workaround (correctness; keep for opt<2).
- `110886f0` — **`DistributedRMSNormRuleBook`** (`Analysis/OpRules/NormalizationRules.{h,cpp}`,
  registered in `OpRuleBook.cpp`):
  - `getInputLayoutFilter`: operand 0 (input) & 2 (residual) → full-bbox width-sharded L1
    (reuses `layout_filter_utils::isFullBboxSharded`); operand 1 (weight) → ROW_MAJOR.
  - `generatesRowMajorInputSiblings(1)` for the weight.
  - `getOutputHints`: keep only full-bbox width-sharded L1 outputs; attach the
    `LayerNormShardedMultiCoreProgramConfig` derived from the shard spec (grid = bbox,
    block_h/block_w from the shard shape — same math as the workaround).
  - `applyOpSpecificAttrs`: write the program config onto the op.
  - `adjustScore`: mirrors `RmsNormRuleBook` (core count = input shard-grid volume).
  - New `DistributedRMSNormAttrs` variant member in `OpConfig::OpSpecificAttrs`.
- `0f43258e` — **[WIP]** gate `DistributedRMSNormWidthShardInputRewritePattern` into the
  `optimizationLevel < 2` block (next to `PagedUpdateCacheOpRewritePattern`).

Everything compiles (`MLIRTTNNAnalysis`, `ttmlir-opt`).

## The blocker: no `OpModel<DistributedRMSNormOp>`

With the gating on, opt-2 does **not** work yet. Reproduced by running the pipeline on
`test/ttmlir/Dialect/TTNN/Transforms/Workarounds/distributed_rms_norm_ttir_l1_input.mlir`
at `optimization-level=2` (set `TT_METAL_RUNTIME_ROOT` to the tt-metal dir):

```
DistributedRMSNormOp::allocateSemaphores: Assertion `inputShardSpec.has_value()' failed
```

i.e. the optimizer left the norm's input **interleaved** and the always-on
`TTNNAllocateDistributedOpSemaphores` pass asserted. Reason: the beam search cannot
*validate* a width-sharded `distributed_rms_norm` candidate without an op-model, so it never
chooses one. The rulebook's filters/hints are necessary but not sufficient — the op-model is
what lets the beam search confirm a sharded candidate is legal.

## Can the backend answer `rms_allgather` constraints?

**Directly: no.** The op-model query is
`::ttnn::graph::query_op_constraints(op, mockDevice, …)` on a mock device with **fabric
DISABLED** (`SingletonDeviceContext.cpp:63`, blocked on tt-metal#44748), default mesh
`{1,1}`. `fused_rms_minimal` needs a mesh + fabric all-gather over `cluster_axis` + a global
semaphore, so it cannot be graph-captured there. No CCL/fabric op is modeled anywhere.

**Feasible path (recommended): proxy via the local norm.** The all-gather only exchanges the
tiny stats tile; it does not affect input/output layout legality. The **local** norm
compute *is* modeled on the single mock device — `OpModel<RMSNormOp>`,
`OpModel<RMSNormPreAllGatherOp>` (`TTNNOpModel.cpp:6635/6725`). So implement
`OpModel<DistributedRMSNormOp>::getOpConstraints` by querying the local sharded
`rms_norm` / `rms_norm_pre_all_gather` with the same input layout + program config. That
gives the beam search the compile-time answer it needs (is this width-shard layout legal)
without fabric.

Notes:
- This proxy does **not** require any tt-metal uplift — it uses the existing local `rms_norm`
  op-model, whose validator already enforces full-bbox.
- A *direct* fused-op query (which would run tt-metal's new full-bbox `validate()` from
  issue #50979) needs tt-metal#44748 (mock fabric) first — a later refinement, not required
  to unblock opt-2.

## Remaining work

1. Implement `OpModel<DistributedRMSNormOp>` as the local-norm proxy (above). Consider
   whether the runtime model should add the stats all-gather cost, or approximate with the
   local compute (layout legality is unaffected either way).
2. Enable the gating (drop the WIP marker) once (1) lands.
3. Tests (mirror `PagedUpdateCache`'s opt-2 lit tests):
   - opt-2 lit test: `distributed_rms_norm` → full-bbox width-sharded input + a
     `layernorm_sharded_multicore_program_config`, no DRAM spill, no workaround ToLayoutOp.
   - opt<2 lit test: the workaround still runs.
   - End-to-end: llama-3.1-70B 1-layer, `--pcc-decode`, opt-2 on qb2 (2×2 mesh).

## Open questions for optimizer devs

- Is the local-norm-proxy op-model an acceptable pattern, or do you prefer waiting for
  tt-metal#44748 so the fused op can be queried directly?
- The layout generator uses canonical (row-major) placement. See "Blackhole blocker" above:
  the immediate correctness fix is the divisibility constraint; explicit-rectangle placement
  in `LegalTensorLayoutAnalysis` is then wanted only to recover utilization (reach 8×8=64 on
  BH instead of the canonical divisor cap of 1×8). Is that placement change acceptable?

## References

- tt-xla#5738 (the decode PCC bug), tt-metal#50979 (runtime validator), tt-metal#44748
  (mock fabric for CCL op-models).
- Precedent: `PagedUpdateCacheOp` (op-model `TTNNOpModel.cpp:4834` + rulebook + opt<2 gating).
