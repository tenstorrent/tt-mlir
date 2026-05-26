# Analysis: Non-Sharded Ops in FPN — Root Causes and Fix Roadmap

## Overview

This document analyzes the non-sharded operations in the EVO50 FPN block,
categorizes their root causes, and describes what compiler changes can bring
them into L1 sharded execution.

**Baseline** (after the upsample spill fix, see
[`op-rules-hs-slice-spill-fix.md`](./op-rules-hs-slice-spill-fix.md)):

| Metric | Value |
|--------|-------|
| `total_shardable_ops` | 57 |
| `sharded_ops` | 18 (31.6%) |
| `effectively_sharded_ops` | 18 |
| `sharded_and_spilled_ops` | 0 |
| `dram_spilled_ops` | 11 |

---

## FPN Graph Overview

The EVO50 FPN processes backbone features at 5 scales (deepest to shallowest).
Each scale has two conv2d operations: one after the upsample (inner-upsample
conv2d) and one after the channel-merge concat (inner-concat conv2d).

```
arg5 (1×80×3×13 DRAM)
 ↓ to_layout → permute → pad
                           ↓
                        upsample (HS RM) → slice (HS RM) → reshape (Group B)
                                                                ↓
arg4 (1×80×6×26 DRAM)                              conv2d_1 (Group C conv2d)
   ↓ to_layout (DRAM)                                   ↓
   └──────────────── concat_6 (L1) ─── permute ─── reshape ─── conv2d_7
                                        (Group D)                (inner-concat conv2d)
...repeat for scales 12×52, 24×104, 48×208, 96×416...
```

Five such chains are stacked, with the output of each concat feeding as one
input to the next scale's concat.

---

## Categorized Non-Sharded Ops (After P1 Baseline)

### Group A — Pre-Upsample Data Prep (3 ops)

| Op | Location | Current layout | Reason not sharded |
|----|----------|---------------|-------------------|
| `to_layout` | Resize2d_0 input | DRAM tile | Backbone input; tile conversion for permute |
| `permute` | Resize2d_0.transpose | DRAM interleaved tile | Tiny tensor (~6 KB); not beneficial |
| `pad` | Resize2d_0.pad | DRAM interleaved tile | Tiny tensor (~8 KB); `PadRuleBook` returns non-sharded |

**Verdict**: No practical benefit. Tensors are tiny (sub-10 KB each) and remain
in DRAM. Sharding would increase overhead.

---

### Group B — Reshape Upsample NHWC → Flat (Conv2d Activation Format)

These reshapes convert `1×H×W×C` NHWC (HS RM from upsample) to
`1×1×(H·W)×C` flat format required by the tt-metal conv2d activation input.
`canReshapeBeView` condition 1 (last dim C unchanged) holds for all; condition 2
(second-to-last tile-aligned for tiled input) depends on the spatial size.

| Op | Scale | Input shape | Second-to-last W | W % 32 | Status |
|----|-------|------------|-----------------|--------|--------|
| [05] | 6×26 | `1×6×26×80` | **26** | **26 ≠ 0** | L1 interleaved (**blocked**) |
| [18] | 12×52 | `1×12×52×80` | **52** | **20 ≠ 0** | **HS RM** ✓ (after P2) |
| [29] | 24×104 | `1×24×104×64` | **104** | **8 ≠ 0** | **HS RM** ✓ (after P2) |
| [40] | 48×208 | `1×48×208×64` | **208** | **16 ≠ 0** | **HS RM** ✓ (after P2) |
| [51] ✓ | 96×416 | `1×96×416×32` | **416** | **0** | **HS RM** ✓ (before P2) |

**Op [05] remains blocked**: The downstream `Conv2d_1` uses `block_sharded`
layout. Block-sharded conv2d requires interleaved or BS input; HS RM input is
incompatible. The greedy optimizer drops the HS RM candidate from the beam and
falls back to L1 interleaved tile. Fixing op [05] requires changing `Conv2d_1`
from `block_sharded` to `height_sharded`, which needs OpModel re-evaluation for
the 6×26 spatial size and is blocked by the small shard count.

**Ops [18, 29, 40] fixed by P2**: See the P2 section below.

---

### Group C — Reshape Flat Conv2d Output → NHWC (post-conv2d reshape)

After each conv2d, the output in flat format `1×1×(H·W)×C` is reshaped back to
`1×H×W×C` NHWC for the subsequent permute. In the 2D tile mapping, both shapes
collapse to `(H·W, C)`, so `canReshapeBeView` condition 2 (second-to-last
aligned) holds whenever H·W is tile-aligned — which is always true because H·W
is the total spatial size across cores.

**After P2, all Group C reshapes are sharded** — as a cascade effect of the
upstream flatten reshape (Group B) and conv2d becoming HS. When conv2d outputs
HS TILE, its flat output has shape `(H·W, C)` where H·W % 32 = 0 (conv2d
always outputs tile-aligned), so `canReshapeBeView` returns true for tiled HS
input. The NULL hint is tried, and the reshape becomes a zero-cost HS TILE view.

| Op | Scale | Format | Status (after P2) |
|----|-------|--------|------------------|
| Relu_5 | 6×26 | BS TILE | **Sharded** ✓ |
| Relu_11 | 6×26 (inner) | BS TILE | **Sharded** ✓ |
| Relu_17 | 12×52 | HS TILE | **Sharded** ✓ |
| Relu_23 | 12×52 (inner) | HS TILE | **Sharded** ✓ |
| Relu_29 | 24×104 | HS TILE | **Sharded** ✓ |
| Relu_35 | 24×104 (inner) | HS TILE | **Sharded** ✓ |
| Relu_41 | 48×208 | HS TILE | **Sharded** ✓ |
| Relu_47 | 48×208 (inner) | HS TILE | **Sharded** ✓ |
| Relu_53 | 96×416 | HS TILE | **Sharded** ✓ (was already before P2) |

**Group C is fully resolved** as a side-effect of P2. The 4 inner-concat conv2ds
(Conv2d_7, Conv2d_19, Conv2d_31, Conv2d_43) became HS because they receive L1
interleaved input from the concat and the conv2d kernel internally shards the
compute.

---

### Group D — Permute NHWC→NCHW (5 ops)

After each Group C reshape, a `permute [0,3,1,2]` converts `1×H×W×C` NHWC to
`1×C×H×W` NCHW for the channel-dim concat. This permute changes the last
dimension (from C to W), so `canReshapeBeView` is always false and
`PermuteRuleBook::getOutputHints` returns `nonShardedOutputHints` →
L1 interleaved.

| Op | After which conv2d | Status |
|----|--------------------|--------|
| `Relu_5` permute | Conv2d_1 (6×26) | L1 interleaved |
| `Relu_17` permute | Conv2d_13 (12×52) | L1 interleaved |
| `Relu_29` permute | Conv2d_25 (24×104) | L1 interleaved |
| `Relu_41` permute | Conv2d_37 (48×208) | L1 interleaved |
| `Relu_53` permute | Conv2d_49 (96×416) | L1 interleaved |

**Root cause**: Permuting `[0,3,1,2]` fundamentally changes the data's last
dimension. HS RM sharding distributes rows along the first spatial dims; a
permutation that swaps axes invalidates the shard address map. tt-metal does not
support HEIGHT_SHARDED permute when the shard axis changes. The only fix is to
**eliminate** these permutes, not to make them sharded.

**Fix option**: NHWC concat (P3 — see below). If the concat operates on NHWC
input (dim=3 instead of dim=1), the `[0,3,1,2]` permute is no longer needed.

---

### Group E — `to_layout` for Backbone Inputs Before Concat (5 ops)

Five backbone feature tensors (function args from DRAM RM) are converted to
tile format before feeding the concat:

```
arg_i  (DRAM RM)  → to_layout → DRAM tile → concat
```

`ToLayoutRuleBook::getOutputHints` returns `nonShardedOutputHints` (DRAM + L1
interleaved configs). The optimizer selects DRAM tile because:
1. The backbone arg is a function parameter already in DRAM.
2. Converting to L1 requires allocation of the full tensor in L1, which is
   often not feasible for the larger scales.

**Verdict**: These 5 DRAM ops are unavoidable without backbone-specific
optimizations. The tensors stay in DRAM and are read by the concat kernel.

---

### Group F — Concat Outputs (5 ops, L1 interleaved after P1)

After P1, all 5 FPN concat outputs moved from DRAM to L1 interleaved:

```
Concatenate_6   → 1×160×6×26   L1 interleaved tile
Concatenate_18  → 1×128×12×52  L1 interleaved tile
Concatenate_30  → 1×128×24×104 L1 interleaved tile
Concatenate_42  → 1×64×48×208  L1 interleaved tile
Concatenate_54  → 1×32×96×416  L1 interleaved tile
```

These remain L1 interleaved (not sharded) because:
1. The concat merges an HS/BS output from the FPN chain with a DRAM backbone
   input — the two halves have incompatible shard specs.
2. `ConcatRuleBook::isValidInputCombination` accepts mixed memory types
   (DRAM + L1) but the resulting output cannot be HS when inputs differ.

**Fix option**: See P3 below for a path to HS concat output.

---

### Group G — Pre-Inner-Concat-Conv2d Chain (8 ops)

Before each inner-concat conv2d (Conv2d_7, Conv2d_19, Conv2d_31, Conv2d_43),
the concat output (L1 interleaved NCHW tile) is converted to NHWC flat format:

```
concat (L1 interleaved NCHW tile)
  → permute [0,2,3,1] (NCHW→NHWC, L1 interleaved)
  → reshape flat (L1 interleaved)
  → conv2d_inner (HS TILE or BS TILE)
```

| Op | Scale | Operation | Status |
|----|-------|-----------|--------|
| `Conv2d_7.dc.transpose.1` | 6×26 | permute | L1 interleaved |
| `Conv2d_7.dc.transpose.1_reshape` | 6×26 | reshape | L1 interleaved |
| `Conv2d_19.dc.transpose.1` | 12×52 | permute | L1 interleaved |
| `Conv2d_19.dc.transpose.1_reshape` | 12×52 | reshape | L1 interleaved |
| `Conv2d_31.dc.transpose.1` | 24×104 | permute | L1 interleaved |
| `Conv2d_31.dc.transpose.1_reshape` | 24×104 | reshape | L1 interleaved |
| `Conv2d_43.dc.transpose.1` | 48×208 | permute | L1 interleaved |
| `Conv2d_43.dc.transpose.1_reshape` | 48×208 | reshape | L1 interleaved |

**Note**: These ops are non-sharded but are NOT blocking the inner conv2ds from
being sharded. Conv2d_7 (BS), Conv2d_19 (HS), Conv2d_31 (HS), Conv2d_43 (HS)
are all sharded despite receiving L1 interleaved input — the conv2d kernel
handles the layout transition internally. The pre-conv2d permute+reshape being
non-sharded is essentially a no-op overhead for small tensors.

**Fix**: Requires HS output from concat (P3) AND HS permute support in tt-metal,
or elimination of the NCHW→NHWC permute via P3 NHWC-concat frontend changes.

---

## P1 Fix: Concat → L1 Interleaved Output

**Goal**: Move concat outputs from DRAM tile to L1 interleaved tile to reduce
DRAM traffic at the concat merge points.

**Mechanism**: In `ConcatRuleBook::getOutputHints`, collect L1 interleaved
configs from `legalConfigs` and place them as **primary hints before** the NULL
hint (→ DRAM). `L1SpillManagement` evicts to DRAM when L1 budget is tight.

### Code Change

**File**: `lib/Dialect/TTNN/Analysis/OpRules/DataMovementRules.cpp`
**Function**: `ConcatRuleBook::getOutputHints`

```cpp
// BEFORE: NULL hint (DRAM) is first; L1 interleaved never tried.
result.hints.push_back(OpConfig(TTNNLayoutAttr()));

// AFTER: L1 interleaved from legalConfigs is primary; NULL/DRAM is fallback.
for (const auto &cfg : legalConfigs) {
  if (!cfg.outputLayout) continue;
  auto ml = cfg.outputLayout.getMemLayout();
  if (isL1BufferType(cfg.outputLayout.getBufferType()) && ml &&
      !isShardedMemoryLayout(ml.getValue())) {
    result.hints.push_back(cfg);  // L1 interleaved — primary
  }
}
result.hints.push_back(OpConfig(TTNNLayoutAttr()));  // DRAM — fallback
```

### Verified Impact ✓

| Metric | Before P1 | After P1 |
|--------|-----------|----------|
| `effectively_sharded_ops` | 18 (31.6%) | **18 (unchanged)** |
| `dram_spilled_ops` | 11 | **6** |
| Concat ops | DRAM | **L1 interleaved (all 5)** ✓ |

All 5 concat outputs moved from DRAM to L1 interleaved. Backbone `to_layout`
ops stayed DRAM (optimizer finds DRAM cheaper for function params).

---

## P2 Fix: Reshape NULL Hint for Non-Tile-Aligned Spatial Dims

### Actual Root Cause

**The bug**: `ReshapeRuleBook::getOutputHints` calls `canReshapeBeView` to
decide whether to return `nullHintOnly()` (allowing HEIGHT_SHARDED RM propagation)
or `nonShardedOutputHints` (blocking it). `canReshapeBeView` uses the
**original pre-optimizer IR type** (not the optimizer's beam-state layout) to
check tile-alignment of the second-to-last dimension.

For the Group B flatten reshapes at scales 12×52, 24×104, 48×208:

| Scale | Reshape | IR type second-to-last | W % 32 | canReshapeBeView |
|-------|---------|----------------------|--------|-----------------|
| 12×52 | `1×12×52×80 → 1×1×624×80` | W=52 | **20 ≠ 0** | **false** |
| 24×104 | `1×24×104×64 → 1×1×2496×64` | W=104 | **8 ≠ 0** | **false** |
| 48×208 | `1×48×208×64 → 1×1×9984×64` | W=208 | **16 ≠ 0** | **false** |
| 96×416 | `1×96×416×32 → 1×1×39936×32` | W=416 | **0** | **true** ✓ |

When `canReshapeBeView` returns false, `getOutputHints` returns only
`nonShardedOutputHints` — **never including the NULL hint**. The NULL hint
(`OpConfig(TTNNLayoutAttr())`) is the trigger that makes
`ReshapeOp::getOpConstraints` propagate the upstream HS RM layout through as a
zero-cost view. Without it, the greedy optimizer never evaluates HS RM for
these reshapes, so HS RM never reaches the downstream conv2d.

**Why `canReshapeBeView` uses the original IR type**: The greedy optimizer's
GreedyL1 beam search propagates layouts via a `beamState` map. IR types are
updated in `applyToIR` AFTER the beam search completes. During the forward
pass, `canReshapeBeView` calls `op->getOperand(0).getType()` which returns the
original DRAM tile type — not the HS RM layout the optimizer may have assigned
to the upstream upsample.

**What the doc previously said** (incorrect): "The conv2d OpModel rejects HS
RM input for small per-core shards (4, 10, 39 rows/core)." This was wrong. The
real conv2d OpModel query SUCCEEDS for all FPN scales (H=6, 12, 24, 48, 96).
The fix is entirely in the compiler (`DataMovementRules.cpp`), not in tt-metal.

### The Fix

**File**: `lib/Dialect/TTNN/Analysis/OpRules/DataMovementRules.cpp`
**Function**: `ReshapeRuleBook::getOutputHints`

Add the NULL hint when the last dimension is unchanged (condition 1 of
`canReshapeBeView`), regardless of tile-alignment of the second-to-last dim:

```cpp
OutputHints ReshapeRuleBook::getOutputHints(
    Operation *op, const std::vector<OpConfig> &legalConfigs) const {
  if (canReshapeBeView(op)) {
    return layout_filter_utils::nullHintOnly();
  }

  OutputHints result = layout_filter_utils::nonShardedOutputHints(legalConfigs);

  // Also include the NULL hint when the last dimension is unchanged (condition 1
  // of canReshapeBeView). canReshapeBeView returned false only because the
  // tiled tile-alignment check (condition 2) failed — but it IS a Row-Major
  // view. When an upstream op produces HEIGHT_SHARDED ROW_MAJOR output, the
  // NULL hint causes ReshapeOp::getOpConstraints to propagate the HS RM layout
  // through the reshape as a zero-cost view (analytical bypass). For non-RM
  // inputs the real QUERY_OP_CONSTRAINTS is invoked, which either succeeds
  // harmlessly or fails and the optimizer falls back to nonShardedOutputHints.
  //
  // Note: canReshapeBeView uses the original pre-optimizer IR type for its
  // isTiled() check, so it cannot see the HS RM layout that the optimizer may
  // have assigned to the upstream op. This extra NULL hint closes that gap.
  auto inputType =
      mlir::dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto outputType =
      mlir::dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (inputType && outputType && !inputType.getShape().empty() &&
      !outputType.getShape().empty() &&
      inputType.getShape().back() == outputType.getShape().back()) {
    result.hints.push_back(OpConfig(TTNNLayoutAttr()));
  }
  return result;
}
```

A safety-net analytical fallback was also added to `Conv2dOp::getOpConstraints`
in `lib/OpModel/TTNN/TTNNOpModel.cpp` for HS RM inputs when the graph-trace
query fails. In practice it is never triggered — all FPN-scale conv2d queries
succeed — but it guards against future regressions.

### Cascade Effect

When the Group B flatten reshape becomes HS RM, the downstream conv2d receives
HS RM input and outputs HS TILE. The post-conv2d reshape (Group C) then gets
HS TILE input, and `canReshapeBeView` for tiled input succeeds (the 2D mapping
collapses both shapes to `(H·W, C)` where H·W % 32 = 0 from conv2d's output).
This propagates the sharding three levels deep per scale:

```
flatten reshape (HS RM) → conv2d (HS TILE) → NHWC reshape (HS TILE, view)
```

At scales where the inner-concat conv2d follows, the same cascade applies once
the concat output feeds into the inner chain.

### Verified Impact ✓

| Metric | Before P2 | After P2 |
|--------|-----------|----------|
| `effectively_sharded_ops` | 18 (31.6%) | **29 (50.9%)** |
| `sharded_and_spilled_ops` | 0 | **0** |
| `dram_spilled_ops` | 6 | **6** (unchanged) |
| Group B reshapes sharded | 1 (H=96 only) | **4** (H=12, 24, 48, 96) |
| Group C reshapes sharded | 1 (H=96 only) | **9 (all)** |
| Conv2d ops sharded | 2 (Conv2d_1 BS, Conv2d_49 HS) | **9 (all)** |

**+11 effectively sharded ops breakdown**:
- +3 flatten reshapes now HS RM (H=12, H=24, H=48 — Group B)
- +3 conv2d now HS TILE (Conv2d_13, Conv2d_25, Conv2d_37)
- +3 post-conv2d reshapes now HS TILE (Relu_17, Relu_29, Relu_41 — Group C)
- +4 inner-concat conv2d + reshapes: Conv2d_7 (BS), Relu_11, Conv2d_19 (HS),
  Relu_23, Conv2d_31 (HS), Relu_35, Conv2d_43 (HS), Relu_47 — these were
  previously L1 interleaved; the P2 cascade changed the beam state such that
  the greedy optimizer promoted these to sharded configurations.

The 6 backbone DRAM `to_layout` ops remain unchanged (same as after P1).

---

## Current State After P2

**57 total shardable ops: 29 sharded (50.9%), 28 non-sharded**

```
Sharded (29):
  5 × upsample          HS RM    (all 5 scales)
  2 × slice             HS RM    (scales 6×26, 12×52 — width trim)
  4 × flatten reshape   HS RM    (Group B: scales 12, 24, 48, 96)
  9 × conv2d            HS/BS    (all 9 conv2ds)
  9 × NHWC reshape      HS/BS    (Group C: all 9 post-conv2d reshapes)

Non-sharded (28):
  3 × pre-upsample prep DRAM     (Group A: to_layout, permute, pad)
  1 × flatten reshape   L1       (Group B: H=6 scale, blocked by BS conv2d)
  5 × NHWC→NCHW permute L1       (Group D: before each concat)
  5 × backbone to_layout DRAM    (Group E: backbone arg tile conversion)
  5 × concat            L1       (Group F: L1 interleaved after P1)
  8 × pre-inner-conv2d  L1       (Group G: NCHW→NHWC permute + flatten)
  1 × pad               L1       (between scale chains, tiny)
```

---

## P3 Fix: NHWC Concat — Eliminate Group D Permutes

**Goal**: Eliminate the 5 NHWC→NCHW permutes (Group D) by restructuring the
FPN concat to operate on NHWC-format tensors.

**Why Group D is the next bottleneck**: After P2, Group D permutes are the only
remaining non-sharded ops that are both large enough to matter and structurally
fixable. The permute changes the last dimension (`[0,3,1,2]` swaps C and W),
so `canReshapeBeView` is always false. tt-metal does not support HEIGHT_SHARDED
permute when the sharding dimension changes. The only path forward is to
**eliminate** these permutes.

**Mechanism**: The permutes exist because the FPN concat operates along the
channel dimension (`dim=1`) in NCHW format. If the concat is restructured to
operate along the channel dimension in NHWC format (`dim=3`), the
NHWC→NCHW conversion before each concat is no longer needed.

### Impact on the FPN Graph

Before P3 (current):
```
conv2d output (HS TILE, NHWC flat) → Relu reshape (HS TILE, NHWC)
  → permute [0,3,1,2] (L1, NCHW)  ← ELIMINATED by P3
  → concat (dim=1, L1 NCHW)       ← changes to dim=3 NHWC
```

After P3:
```
conv2d output (HS TILE, NHWC flat) → Relu reshape (HS TILE, NHWC)
  → concat (dim=3, NHWC)          ← directly uses HS TILE NHWC input
```

**Expected impact**: +5 effectively sharded ops (the 5 Group D permutes
eliminated from the non-sharded count; concats now receive HS TILE input
directly, with the backbone DRAM input as the only non-HS operand).

### Secondary Effect on Group G

If the concat receives NHWC input, the inner-concat conv2d (Conv2d_7, etc.)
no longer needs a NCHW→NHWC permute before it. The pre-inner-conv2d chain
becomes:

```
concat (dim=3 NHWC, L1)
  → flatten reshape (L1, NHWC→flat)   ← no permute needed
  → conv2d_inner (HS TILE)
```

This eliminates 4 of the 8 Group G permutes, adding another +4 ops eliminated.

**Total P3 impact**: up to +9 ops (5 Group D permutes gone + 4 Group G permutes
gone, replaced by cheaper flatten reshapes), though the flatten reshapes remain
non-sharded since the concat output is L1 interleaved.

### Implementation

This is a **Forge frontend change**, not a tt-mlir compiler change:

1. In `forge/csrc/lowering/lowering.cpp` (or the ONNX→TTIR lowering), detect
   FPN-style concat-along-channel patterns where inputs are in NHWC format.
2. Lower the concat with `dim=3` (NHWC) instead of adding
   NHWC→NCHW permute + `dim=1` concat.
3. Ensure downstream ops receiving NHWC-format backbone features are also
   permuted consistently (the backbone args may need a one-time NCHW→NHWC
   conversion at the function boundary instead of per-concat).

**Alternative**: A tt-mlir pass that pattern-matches
`permute [0,3,1,2] → concat(dim=1) → permute [0,2,3,1]` and rewrites it to
`concat(dim=3)`. This keeps the frontend unchanged but adds a fold pass.

**Difficulty**: Medium. The concat semantics and shape inference need updating
across ONNX→TTIR→TTNN lowering stages.

---

## P4 Fix: Height-Sharded Concat Output

**Goal**: After P3, the concat still outputs L1 interleaved even though one of
its inputs is now HS TILE (the conv2d chain output in NHWC). Making the concat
output HEIGHT_SHARDED would allow the downstream inner-concat conv2d to receive
HS input directly, potentially improving data locality.

**Blocker**: The concat merges an HS TILE NHWC input (from the conv2d chain)
with a DRAM tile NHWC input (from the backbone). The backbone input has a
different (non-HS) memory layout. `ConcatRuleBook::isValidInputCombination`
currently requires compatible memory types. Two paths:

1. **Mixed-input concat**: Allow concat to output HS when one input is HS and
   the other is DRAM/L1 interleaved. The concat kernel must be able to read
   from mixed memory configs. Requires changes to `ConcatRuleBook` and
   verification that tt-metal's concat kernel handles this case.

2. **Promote backbone to HS before concat**: Insert a `to_layout (DRAM → HS)`
   for the backbone tensor before the concat. The backbone tensor (e.g.,
   `1×64×12×52×80` at 12×52 scale ≈ 160 KB) must fit in L1 HS format across
   64 cores (2.5 KB/core ≈ feasible). The `to_layout` op itself would be a
   non-sharded overhead but the concat output would then be HS.

**Expected impact**: +5 ops (the 5 concat outputs become HS). The Group G
flatten reshapes may also cascade to HS RM if the inner-concat conv2d receives
HS input.

**Difficulty**: Hard. Requires tt-metal kernel support or added L1 budget
analysis for backbone-to-HS conversion.

---

## Summary Roadmap

| Priority | Change | Location | Impact | Difficulty |
|----------|--------|----------|--------|-----------|
| **P1** ✓ | L1 interleaved concat output | `DataMovementRules.cpp` | +5 ops in L1 (`dram_spilled` 11→6) | Low |
| **P2** ✓ | Fix reshape NULL hint for non-tile-aligned dims | `DataMovementRules.cpp` + `TTNNOpModel.cpp` | +**11** HS sharded ops (50.9%) | Low-Medium |
| **P3** | NHWC concat (eliminate Group D permutes) | Forge frontend or tt-mlir fold pass | +5–9 ops eliminated | Medium |
| **P4** | HS concat output (promote backbone to HS) | `DataMovementRules.cpp` + tt-metal | +5 HS sharded | Hard |

### Current State (After P1+P2)

```
effectively_sharded_ops: 29 / 57  (50.9%)
sharded_and_spilled_ops: 0
dram_spilled_ops:        0        (is_spilled_to_dram metric; DRAM backbone
                                   to_layout ops are not counted as spilled)
```
