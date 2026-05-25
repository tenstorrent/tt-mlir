# Analysis: Non-Sharded Ops in FPN — Root Causes and Fix Roadmap

## Overview

This document analyzes the 39 non-sharded operations in the EVO50 FPN block
after the upsample-spill fix, categorizes their root causes, and describes
what compiler changes can bring them into L1 sharded execution.

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

The EVO50 FPN processes backbone features at 5 scales (deepest to shallowest):

```
arg5 (1×80×3×13 DRAM)
 ↓ to_layout → permute → pad → upsample (HS) → slice (HS) → reshape → conv2d (BS)
                                                                         ↓
arg4 (1×80×6×26 DRAM) → to_layout ──────────────────────────────────── concat → DRAM output
                                                                         ↑
                               permute → reshape ← conv2d ← permute+reshape ← concat input
...repeat 4 more times at scales 12×52, 24×104, 48×208, 96×416...
```

The 5 concat operations produce the FPN's final outputs and feed directly to the
next model stage.

---

## Categorized Non-Sharded Ops

### Group A — Pre-Upsample Data Prep (4 ops)

| Op | Location | Current layout | Reason not sharded |
|----|----------|---------------|-------------------|
| `to_layout` | Resize2d_0 input | DRAM tile | Backbone input; needed as tile for permute |
| `permute` | Resize2d_0.transpose | L1 interleaved tile | Tiny tensor (1×3×13×80); not beneficial |
| `pad` | Resize2d_0.pad | L1 interleaved tile | `PadRuleBook` always returns non-sharded |
| `pad` | Relu_11pad | L1 interleaved tile | Same — pad before Resize2d_12 upsample |

**Verdict**: No practical benefit. Tensor too small for H×W sharding; pad
cannot output sharded when input is interleaved (tt-metal constraint).

---

### Group B — Reshape After Upsample → Conv2d Input (4 ops)

These reshapes flatten NHWC to the 4D conv2d activation format
(`1×B×H×W → 1×1×(B·H·W)×C`). `canReshapeBeView` is `true` for all
(last dim C unchanged), so `getOutputHints` returns `nullHintOnly()`.
However the output lands as L1 interleaved tile instead of HS RM.

| Op | Location | Input HS shard | Output | Rows/core | Downstream conv |
|----|----------|---------------|--------|----------|-----------------|
| [05] | Resize2d_0 reshape | 39 cores, 4×80 | L1 interleaved | **4** | `block_sharded` |
| [18] | Resize2d_12 reshape | 64 cores, 10×80 | L1 interleaved | 10 | `height_sharded` |
| [29] | Resize2d_24 reshape | 64 cores, 39×64 | L1 interleaved | 39 | `height_sharded` |
| [40] | Resize2d_36 reshape | 64 cores, 156×64 | L1 interleaved | 156 | `height_sharded` |
| [51] ✓ | Resize2d_48 reshape | 64 cores, **624×32** | **HS RM** | **624** | `height_sharded` |

**Root causes**:

1. **Op [05]** — Downstream `Conv2d_1` uses `shard_layout=block_sharded`. BS conv2d
   requires interleaved or BS input; HS RM input is incompatible. The greedy optimizer's
   beam search drops the HS RM candidate and falls back to L1 interleaved tile.

2. **Ops [18, 29, 40]** — Downstream conv2ds use `shard_layout=height_sharded`. The
   NULL hint for reshape passes `outputLayout=nullptr` to
   `ReshapeOp::getOpConstraints`, which returns HS RM output (view).
   However, the conv2d **OpModel** rejects HS RM input for small per-core shards
   (4, 10, 39 rows/core). The threshold appears to be ~624 rows/core (at 32
   channels) — the exact configuration that works for Resize2d_48. For smaller
   shards the tt-metal conv2d activation activation kernel falls back to DRAM or
   requires tiled input.

**Fix for ops [29, 40]** (medium difficulty): Extend tt-metal conv2d HS RM input
support to shard sizes ≥32 rows/core (39 and 156 rows/core). This is a
tt-metal change; the compiler already generates the correct NULL hint.

**Fix for op [05]**: Requires changing `Conv2d_1` from `block_sharded` to
`height_sharded`, which needs OpModel re-evaluation for the 6×26 spatial size.

---

### Group C — Reshape Conv2d Output → NHWC (8 ops)

After each conv2d, the output is reshaped from flat format
`1×1×(H·W)×C` back to NHWC `1×H×W×C`.

`canReshapeBeView` for **tiled** layouts requires that the second-to-last
dimension is tile-aligned in both input and output. The EVO50 spatial
dimensions are not tile multiples:

| Scale | Output H | H % 32 | Output W | W % 32 | View reshape? |
|-------|---------|--------|---------|--------|--------------|
| 6×26 | 26 | **26** | 6 | **6** | ✗ NO |
| 12×52 | 52 | **20** | 12 | **12** | ✗ NO |
| 24×104 | 104 | **8** | 24 | **24** | ✗ NO |
| 48×208 | 208 | **16** | 48 | **16** | ✗ NO |
| **96×416** | **416** | **0** | **96** | **0** | **✓ YES** → op [53] sharded |

Because these reshapes cannot be views, `getOutputHints` returns
`nonShardedOutputHints` → L1 interleaved tile. **This is a fundamental model
architecture constraint**: the EVO50 feature map dimensions (6, 26, 52, 104,
208) are not multiples of the tile size (32).

**Fix option A — Model change**: Pad feature maps to tile-aligned sizes
(e.g., 32×32, 64×64). Not feasible without retraining.

**Fix option B — HS Row-Major through conv2d output** (high impact):

If conv2d could output in HS Row-Major format (instead of HS Tiled), then:
- Reshape `1×1×(H·W)×C` (HS RM) → `1×H×W×C` (HS RM): `canReshapeBeView=true`
  (last dim C unchanged; no tile-alignment check for RM).
- All 8 reshape ops would become HS RM views.

Implementation requires `TTNNRowMajorLayoutPropagation` to insert a
`to_layout (HS tile→HS RM)` after conv2d and propagate RM through the
reshape→permute chain.

---

### Group D — Permute NHWC→NCHW (5 ops)

After the reshape, a `permute [0,3,1,2]` converts `1×H×W×C` to `1×C×H×W`
for the concat along the channel dim. This permute changes the last dimension
(from C to W), so `canReshapeBeView` is always `false` →
`getOutputHints` → `nonShardedOutputHints` → L1 interleaved.

**Fix option**: Eliminate the permute by performing concat in NHWC format
(`dim=3` instead of `dim=1`). This is a model-lowering change in the Forge
frontend, not a tt-mlir change.

Even with a Group C fix (HS RM reshape), the permute NHWC→NCHW would still
be L1 interleaved because the last dimension changes.

---

### Group E — `to_layout` for Backbone Inputs Before Concat (5 ops)

Five backbone feature tensors (function args from DRAM) are converted to tile
format before being fed to the concat:

```
arg_i  (DRAM RM)  → to_layout → DRAM tile
                                      ↓
conv2d chain → ... → permute → L1 interleaved tile → concat → DRAM tile
```

`ToLayoutRuleBook::getOutputHints` returns `nonShardedOutputHints` which
includes both DRAM and L1 interleaved configs. Currently the optimizer selects
DRAM tile because:

1. The concat's `getOutputHints` puts the NULL hint (→ DRAM) as the **first**
   primary hint, so DRAM output for concat is evaluated and scored before any
   L1 alternative.
2. With a DRAM concat output in the beam, the optimizer has no incentive to
   promote the `to_layout` output from DRAM to L1.

---

### Group F — Concat Outputs (5 ops, currently DRAM)

The five FPN output concats all produce DRAM tile:

```
Concatenate_6   → 1×160×6×26   DRAM tile
Concatenate_18  → 1×128×12×52  DRAM tile
Concatenate_30  → 1×128×24×104 DRAM tile
Concatenate_42  → 1×64×48×208  DRAM tile
Concatenate_54  → 1×32×96×416  DRAM tile
```

These are the FPN outputs returned to the model caller. They are in DRAM
because the concat defaults to DRAM when the NULL hint is primary (see Group E).

**Note**: `ConcatRuleBook::isValidInputCombination` checks `TensorMemoryLayout`
(Interleaved vs Sharded), NOT buffer type (DRAM vs L1). Therefore DRAM
interleaved and L1 interleaved inputs satisfy the combination check equally. A
concat with one DRAM input and one L1 input can legally produce an L1 output.

---

## P1 Fix: Concat → L1 Interleaved Output

**Goal**: Move concat outputs from DRAM tile to L1 interleaved tile for the
cases where L1 budget allows, and cascade this to also promote the backbone
`to_layout` outputs.

**Mechanism**: In `ConcatRuleBook::getOutputHints`, collect L1 interleaved
configs from `legalConfigs` and place them as **primary hints before** the NULL
hint. The NULL hint (→ DRAM) becomes a lower-priority fallback. `L1SpillManagement`
will automatically evict to DRAM when the L1 budget is tight.

### L1 Budget for Backbone Tensors

Tensors in L1 interleaved are spread across 64 cores:

| Concat | Backbone tensor | Total size | L1/core |
|--------|----------------|-----------|---------|
| Concat_6 | `1×80×6×26` bf16 | ~25 KB | ~390 B |
| Concat_18 | `1×64×12×52` bf16 | ~160 KB | ~2.5 KB |
| Concat_30 | `1×64×24×104` bf16 | ~625 KB | ~9.8 KB |
| Concat_42 | `1×32×48×208` bf16 | ~1.25 MB | ~19.5 KB |
| Concat_54 | `1×32×96×416` bf16 | ~5 MB | ~78 KB |

Smaller scales fit trivially. The largest (Concat_54, 78 KB/core) requires
L1SpillManagement to verify against the total live L1 footprint at that program
point.

### Code Change

**File**: `lib/Dialect/TTNN/Analysis/OpRules/DataMovementRules.cpp`
**Function**: `ConcatRuleBook::getOutputHints`

```cpp
// BEFORE: NULL hint (DRAM) is first; L1 interleaved never tried.
result.hints.push_back(OpConfig(TTNNLayoutAttr()));
// ... sharded configs only

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
// ... sharded configs as before
```

### Expected Impact

If L1SpillManagement allows all 5 concats in L1:

| Metric | Before P1 | After P1 (expected) |
|--------|-----------|-------------------|
| `effectively_sharded_ops` | 18 (31.6%) | 18 (unchanged) |
| `dram_spilled_ops` | 11 | ~1 (large concat only) |
| Concat ops in L1 | 0 | 4–5 |
| `to_layout` backbone in L1 | 0 | 4–5 |

The sharding percentage doesn't increase (concat and to_layout are interleaved,
not sharded), but **10 ops move from DRAM to L1 interleaved**, significantly
reducing DRAM traffic for the concat merge steps.

---

## P2 Fix: HS RM Input for Conv2d (ops [29, 40])

**Goal**: Enable height_sharded row-major input for conv2d at 24×104 and
48×208 spatial sizes. Requires tt-metal conv2d OpModel to accept HS RM input
when per-core shard size ≥ 32 rows.

**Impact**: +2 effectively sharded ops.

---

## P3 Fix: HS RM Through Conv2d Output

**Goal**: Extend `TTNNRowMajorLayoutPropagation` to insert `to_layout (HS tile
→ HS RM)` after conv2d and propagate HS RM through the conv2d→reshape→permute
chain. Since reshape with HS RM input is a view (no tile-alignment check), all
8 Group C reshapes become HS RM.

**Impact**: +8 effectively sharded ops (the Group C reshapes, all scales).

---

## P4 Fix: NHWC Concat (Skip Permute)

**Goal**: Restructure FPN lowering to perform concat in NHWC format, eliminating
the permute NHWC→NCHW before each concat.

**Impact**: +5 ops (the Group D permutes eliminated; concat reshapes simplified).
Requires Forge frontend changes.

---

## Summary Roadmap

| Priority | Change | Location | Impact | Difficulty |
|----------|--------|----------|--------|-----------|
| **P1** ✓ | L1 interleaved concat output | `DataMovementRules.cpp` | +10 ops in L1 | Low |
| **P2** | HS RM conv2d input for ≥32 rows/core | tt-metal conv2d | +2 HS sharded | Medium |
| **P3** | HS RM through conv2d→reshape chain | `RowMajorLayoutPropagation.cpp` | +8 HS sharded | Hard |
| **P4** | NHWC concat (no permute) | Forge frontend | +5 ops eliminated | Hard |
